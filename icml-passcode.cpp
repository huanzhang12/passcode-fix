#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)
static void solve_l2r_l1l2_svc_lock( 
	const problem *prob, double *w, double eps,
	double Cp, double Cn, int solver_type, function *fun_obj = NULL, int max_iter = 1000)
{ // {{{
	int l = prob->l;
	int w_size = prob->n;
	//int i, s, iter = 0;
	//double C, d, G;
	int iter = 0;
	double *QD = new double[l];
	int *index = new int[l];
	double *alpha = new double[l];
	double *tg = new double[w_size];
	schar *y = new schar[l];

	//int active_size = l;

	std::vector<int> active_set(l);
	//omp_lock_t writelock;
	//omp_lock_t *lockaaa = (omp_lock_t*)malloc(sizeof(omp_lock_t)*w_size);
	//vector<omp_lock_t > lockaaa(w_size);
	//
	aligned_vector<omp_lock_t> wlocks(w_size);
	for(auto &lock: wlocks) 
		omp_init_lock(&lock);

	int nr_threads = omp_get_max_threads();

#ifdef SHRINKING
	aligned_vector<std::vector<int>> next_active_set(nr_threads);
	// PG: projected gradient, for shrinking and stopping
	double PGmax_old = INF;
	double PGmin_old = -INF;
	aligned_vector<double> PGmax_new(nr_threads);
	aligned_vector<double> PGmin_new(nr_threads);
#pragma omp parallel 
	{
		int tid = omp_get_thread_num();
		next_active_set[tid].reserve(l);
	}
#endif

	double tmp_start = omp_get_wtime();

	// default solver_type: L2R_L2LOSS_SVC_DUAL_LOCK
	double diag[3] = {0.5/Cn, 0, 0.5/Cp};
	double upper_bound[3] = {INF, 0, INF};
	if(solver_type == L2R_L1LOSS_SVC_DUAL_LOCK)
	{
		diag[0] = 0;
		diag[2] = 0;
		upper_bound[0] = Cn;
		upper_bound[2] = Cp;
	}

#pragma omp parallel for
	for(int i=0; i<l; i++)
	{
		if(prob->y[i] > 0)
			y[i] = +1;
		else
			y[i] = -1;
		active_set[i] = i;
		// Initial alpha can be set here. Note that
		// 0 <= alpha[i] <= upper_bound[GETI(i)]
		alpha[i] = 0;
	}

#pragma omp parallel for
	for(int i=0; i<w_size; i++)
		w[i] = 0;
#pragma omp parallel for
	for(int i=0; i<l; i++)
	{
		QD[i] = diag[GETI(i)];

		feature_node *xi = prob->x[i];
		while (xi->index != -1)
		{
			double val = xi->value;
			QD[i] += val*val;
			/*
			val*=y[i]*alpha[i];
			double &wi = w[xi->index-1];
#pragma omp atomic
			wi += val;
			*/
			xi++;
		}
		index[i] = i;
	}

	for (size_t i=0; i<l; i++) {
		size_t j = i+rand()%(l-i);
		std::swap(active_set[i], active_set[j]);
	}
	double inittime = omp_get_wtime() - tmp_start;
	std::vector<unsigned> seeds(nr_threads);
	for(int th = 0; th < nr_threads; th++)
		seeds[th] = th;

	double totaltime = 0, starttime = 0;
	while (iter < max_iter)
	{
		starttime = omp_get_wtime();
#ifdef SHRINKING
		//PGmax_new = -INF; PGmin_new = INF;
#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			PGmax_new[tid] = -INF;
			PGmin_new[tid] = INF;
			next_active_set[tid].clear();
		}
#endif 
		size_t active_size = active_set.size();

		if(0) {
			for (size_t i=0; i<active_size; i++)
			{
				int tid = omp_get_thread_num();
				size_t j = i+rand_r(&seeds[tid])%(active_size-i);
				//size_t j = i+rand()%(active_size-i);
				std::swap(active_set[i], active_set[j]);
			}
			/*
		} else if (nr_threads == 1) {
			for (size_t i=0; i<active_size; i++)
			{
				int tid = omp_get_thread_num();
				size_t j = i+rand_r(&seeds[tid])%(active_size-i);
				std::swap(active_set[i], active_set[j]);
			}*/
		} else {
#pragma omp parallel
			{
				int tid = omp_get_thread_num();
				size_t len = (active_size/nr_threads)+(active_size%nr_threads?1:0);
				size_t start = tid*len, end = std::min((tid+1)*len, active_size);
				for (size_t i=start; i<end; i++)
				{
					size_t j = i+rand_r(&seeds[tid])%(end-i);
					std::swap(active_set[i], active_set[j]);
				}
			}
		}


#pragma omp parallel for schedule(dynamic,8)
		for (size_t s=0; s<active_size; s++) {
			int i = active_set[s];
			double G = 0;
			schar yi = y[i];

			// lock all the variables I need or just wait
			feature_node *xi = prob->x[i];
			while(xi->index!= -1)
			{
				omp_set_lock(&wlocks[xi->index-1]);
				G += w[xi->index-1]*(xi->value);
				xi++;
			}
			G = G*yi-1;

			double C = upper_bound[GETI(i)];
			G += alpha[i]*diag[GETI(i)];

#ifdef SHRINKING
			double PG = 0;
			if (alpha[i] == 0)
			{
				if (G > PGmax_old)
				{
				//	is_shrunken[i] = 1;
				//	active_size--;
				//	swap(index[s], index[active_size]);
				//  s--;
					xi = prob->x[i];
					while(xi->index != -1) {
						omp_unset_lock(&wlocks[xi->index-1]);
						xi++;
					}
					continue;
				}
				else if (G < 0)
					PG = G;
			}
			else if (alpha[i] == C)
			{
				if (G < PGmin_old)
				{
				//	is_shrunken[i] = 1;
				//	active_size--;
				//	swap(index[s], index[active_size]);
				//  s--;
					xi = prob->x[i];
					while(xi->index != -1) {
						omp_unset_lock(&wlocks[xi->index-1]);
						xi++;
					}
					continue;
				}
				else if (G > 0)
					PG = G;
			}
			else
				PG = G;

			int tid = omp_get_thread_num();
			//next_active_set.local().push_back(i);
			//PGmin_new.update(PG);
			//PGmax_new.update(PG);
			next_active_set[tid].push_back(i);
			PGmin_new[tid] = std::min(PGmin_new[tid], PG);
			PGmax_new[tid] = std::max(PGmax_new[tid], PG);
			
			if(fabs(PG) > 1.0e-12)
#endif
			{
				double alpha_old = alpha[i];
				alpha[i] = min(max(alpha[i] - G/QD[i], 0.0), C);

				double d = (alpha[i] - alpha_old)*yi;
				if(d != 0) {
					xi = prob->x[i];
					while (xi->index != -1)
					{
						int ind = xi->index-1;
						double val = d*xi->value;
//#pragma omp atomic
						w[ind] += val;
						xi++;
					}
				}
			}
			xi = prob->x[i];
			while(xi->index != -1) {
				omp_unset_lock(&wlocks[xi->index-1]);
				xi++;
			}
		}

		iter++;
		//if(iter % 10 == 0) { info("."); }

#ifdef SHRINKING

		//double PGmax_new_global = PGmax_new.reduce();
		//double PGmax_new_global = PGmax_new.reduce();
		double PGmin_new_global = PGmin_new.min();
		double PGmax_new_global = PGmax_new.max();

		if(PGmax_new_global - PGmin_new_global <= eps)
		{
			/*
			if(active_size == l)
				break;
			else
			*/
			{
				//	active_size = l;
				//	info("*");
				//std::fill(is_shrunken.begin(), is_shrunken.end(), 0);
				active_set.resize(l);
				for(int ss = 0; ss < l; ss++) active_set[ss] = ss;
				PGmax_old = INF;
				PGmin_old = -INF;
				//	continue;
			}
		}
		else 
		{
			active_set.clear();
			/*
			for(auto &bag: next_active_set.buffer) {
				for(auto &ss : bag)
					active_set.push_back(ss);
			}
			*/
			for(auto &bag: next_active_set) {
				for(auto &ss: bag)
					active_set.push_back(ss);
			}

			PGmax_old = PGmax_new_global;
			PGmin_old = PGmin_new_global;
			if (PGmax_old <= 0)
				PGmax_old = INF;
			if (PGmin_old >= 0)
				PGmin_old = -INF;
		}
#endif
		double itertime = omp_get_wtime() - starttime;
		totaltime += itertime;
		//tesing after each step
		double primal_obj = 0, true_primal_obj = 0;
		double acc = 0, true_acc = 0;
		double err = 0;
		std::vector<double> tmpw(w_size, 0);
#pragma omp parallel for
		for(int ggg = 0; ggg < l; ggg++) {
			double yialpha = prob->y[ggg] * alpha[ggg];
			feature_node *xi = prob->x[ggg];
			while (xi->index != -1) {
#pragma omp atomic
				tmpw[xi->index-1] += xi->value*yialpha;
				xi++;
			}
		}
		double dual_obj = 0, true_dual_obj = 0;
		for(int ggg = 0; ggg < w_size; ggg++) {
			err += (tmpw[ggg]-w[ggg])*(tmpw[ggg]-w[ggg]);
			true_dual_obj += tmpw[ggg]*tmpw[ggg];
			dual_obj += w[ggg]*w[ggg];
		}
		for(int ggg = 0; ggg < l; ggg++) {
			true_dual_obj += alpha[ggg]*(alpha[ggg]*diag[GETI(ggg)]-2);
			dual_obj += alpha[ggg]*(alpha[ggg]*diag[GETI(ggg)]-2);
		}
		dual_obj *= 0.5;
		true_dual_obj *= 0.5;

		if(fun_obj) {
			primal_obj = fun_obj->fun(w);
			acc = fun_obj->testing(w)*100;
			true_primal_obj = fun_obj->fun(&(tmpw[0]));
			true_acc = fun_obj->testing(&(tmpw[0]))*100;
		}

		printf("iter %d walltime %lf itertime %lf f %.3lf d %.3lf acc %.1lf true-f %.3lf true-d %.3lf true-acc %.1lf err %.3g inittime %lf active_size %ld\n", 
				iter, totaltime, itertime, primal_obj, dual_obj, acc, true_primal_obj, true_dual_obj, true_acc, err, inittime, active_set.size());		
		fflush(stdout);
	}


	/*
	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\nUsing -s 2 may be faster (also see FAQ)\n\n");
		*/

	// calculate objective value

	/*
	double v = 0;
	int nSV = 0;
	for(int i=0; i<w_size; i++)
		v += w[i]*w[i];
	for(int i=0; i<l; i++)
	{
		v += alpha[i]*(alpha[i]*diag[GETI(i)] - 2);
		if(alpha[i] > 0)
			++nSV;
	}
	info("Objective value = %lf\n",v/2);
	info("nSV = %d\n",nSV);
	*/

	for(auto &lock: wlocks) 
		omp_destroy_lock(&lock);
	delete [] QD;
	delete [] alpha;
	delete [] tg;
	delete [] y;
	delete [] index;
} // }}} 

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)
static void solve_l2r_l1l2_svc_atomic( 
	const problem *prob, double *w, double eps,
	double Cp, double Cn, int solver_type, function *fun_obj = NULL, int max_iter = 1000)
{ // {{{
	int l = prob->l;
	int w_size = prob->n;
	//int i, s, iter = 0;
	//double C, d, G;
	int iter = 0;
	double *QD = new double[l];
	int *index = new int[l];
	double *alpha = new double[l];
	double *tg = new double[w_size];
	schar *y = new schar[l];

	//int active_size = l;

	std::vector<int> active_set(l);
	//omp_lock_t writelock;
	//omp_lock_t *lockaaa = (omp_lock_t*)malloc(sizeof(omp_lock_t)*w_size);
//	vector<omp_lock_t > lockaaa(w_size);
//
	int nr_threads = omp_get_max_threads();

#ifdef SHRINKING
	aligned_vector<std::vector<int>> next_active_set(nr_threads);
	// PG: projected gradient, for shrinking and stopping
	double PGmax_old = INF;
	double PGmin_old = -INF;
	aligned_vector<double> PGmax_new(nr_threads);
	aligned_vector<double> PGmin_new(nr_threads);
#pragma omp parallel 
	{
		int tid = omp_get_thread_num();
		next_active_set[tid].reserve(l);
	}
#endif

	double tmp_start = omp_get_wtime();

	// default solver_type: L2R_L2LOSS_SVC_DUAL_ATOMIC
	double diag[3] = {0.5/Cn, 0, 0.5/Cp};
	double upper_bound[3] = {INF, 0, INF};
	if(solver_type == L2R_L1LOSS_SVC_DUAL_ATOMIC)
	{
		diag[0] = 0;
		diag[2] = 0;
		upper_bound[0] = Cn;
		upper_bound[2] = Cp;
	}

#pragma omp parallel for
	for(int i=0; i<l; i++)
	{
		if(prob->y[i] > 0)
			y[i] = +1;
		else
			y[i] = -1;
		active_set[i] = i;
		// Initial alpha can be set here. Note that
		// 0 <= alpha[i] <= upper_bound[GETI(i)]
		alpha[i] = 0;
	}

#pragma omp parallel for
	for(int i=0; i<w_size; i++)
		w[i] = 0;
#pragma omp parallel for
	for(int i=0; i<l; i++)
	{
		QD[i] = diag[GETI(i)];

		feature_node *xi = prob->x[i];
		while (xi->index != -1)
		{
			double val = xi->value;
			QD[i] += val*val;
			/*
			val*=y[i]*alpha[i];
			double &wi = w[xi->index-1];
#pragma omp atomic
			wi += val;
			*/
			xi++;
		}
		index[i] = i;
	}
	for (size_t i=0; i<l; i++) {
		size_t j = i+rand()%(l-i);
		std::swap(active_set[i], active_set[j]);
	}
	double inittime = omp_get_wtime() - tmp_start;
	std::vector<unsigned> seeds(nr_threads);
	for(int th = 0; th < nr_threads; th++)
		seeds[th] = th;

	double totaltime = 0, starttime = 0;
	while (iter < max_iter)
	{
		starttime = omp_get_wtime();
#ifdef SHRINKING
		//PGmax_new = -INF; PGmin_new = INF;
#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			PGmax_new[tid] = -INF;
			PGmin_new[tid] = INF;
			next_active_set[tid].clear();
		}
#endif 
		size_t active_size = active_set.size();

		if(0) {
			for (size_t i=0; i<active_size; i++)
			{
				int tid = omp_get_thread_num();
				size_t j = i+rand_r(&seeds[tid])%(active_size-i);
				//size_t j = i+rand()%(active_size-i);
				std::swap(active_set[i], active_set[j]);
			}
			/*
		} else if (nr_threads == 1) {
			for (size_t i=0; i<active_size; i++)
			{
				int tid = omp_get_thread_num();
				size_t j = i+rand_r(&seeds[tid])%(active_size-i);
				std::swap(active_set[i], active_set[j]);
			}*/
		} else {
#pragma omp parallel
			{
				int tid = omp_get_thread_num();
				size_t len = (active_size/nr_threads)+(active_size%nr_threads?1:0);
				size_t start = tid*len, end = std::min((tid+1)*len, active_size);
				for (size_t i=start; i<end; i++)
				{
					size_t j = i+rand_r(&seeds[tid])%(end-i);
					std::swap(active_set[i], active_set[j]);
				}
			}
		}


#pragma omp parallel for schedule(dynamic,8)
		for (size_t s=0; s<active_size; s++)
		{
			int i = active_set[s];
			double G = 0;
			schar yi = y[i];

			feature_node *xi = prob->x[i];
			while(xi->index!= -1)
			{
				G += w[xi->index-1]*(xi->value);
				xi++;
			}
			G = G*yi-1;

			double C = upper_bound[GETI(i)];
			G += alpha[i]*diag[GETI(i)];

#ifdef SHRINKING
			double PG = 0;
			if (alpha[i] == 0)
			{
				if (G > PGmax_old)
				{
				//	is_shrunken[i] = 1;
				//	active_size--;
				//	swap(index[s], index[active_size]);
				//  s--;
					continue;
				}
				else if (G < 0)
					PG = G;
			}
			else if (alpha[i] == C)
			{
				if (G < PGmin_old)
				{
				//	is_shrunken[i] = 1;
				//	active_size--;
				//	swap(index[s], index[active_size]);
				//  s--;
					continue;
				}
				else if (G > 0)
					PG = G;
			}
			else
				PG = G;

			int tid = omp_get_thread_num();
			//next_active_set.local().push_back(i);
			//PGmin_new.update(PG);
			//PGmax_new.update(PG);
			next_active_set[tid].push_back(i);
			PGmin_new[tid] = std::min(PGmin_new[tid], PG);
			PGmax_new[tid] = std::max(PGmax_new[tid], PG);
			
			if(fabs(PG) > 1.0e-12)
#endif
			{
				double alpha_old = alpha[i];
				alpha[i] = min(max(alpha[i] - G/QD[i], 0.0), C);

				double d = (alpha[i] - alpha_old)*yi;
				if(d != 0) {
					xi = prob->x[i];
					while (xi->index != -1)
					{
						int ind = xi->index-1;
						double val = d*xi->value;
#pragma omp atomic
						w[ind] += val;

						xi++;
					}
				}
			}
		}

		iter++;
		//if(iter % 10 == 0) { info("."); }

#ifdef SHRINKING

		//double PGmax_new_global = PGmax_new.reduce();
		//double PGmax_new_global = PGmax_new.reduce();
		double PGmin_new_global = PGmin_new.min();
		double PGmax_new_global = PGmax_new.max();

		if(PGmax_new_global - PGmin_new_global <= eps)
		{
			/*
			if(active_size == l)
				break;
			else
			*/
			{
				//	active_size = l;
				//	info("*");
				//std::fill(is_shrunken.begin(), is_shrunken.end(), 0);
				active_set.resize(l);
				for(int ss = 0; ss < l; ss++) active_set[ss] = ss;
				PGmax_old = INF;
				PGmin_old = -INF;
				eps = eps * 0.5;
				//	continue;
			}
		}
		else 
		{
			active_set.clear();
			/*
			for(auto &bag: next_active_set.buffer) {
				for(auto &ss : bag)
					active_set.push_back(ss);
			}
			*/
			for(auto &bag: next_active_set) {
				for(auto &ss: bag)
					active_set.push_back(ss);
			}

			PGmax_old = PGmax_new_global;
			PGmin_old = PGmin_new_global;
			if (PGmax_old <= 0)
				PGmax_old = INF;
			if (PGmin_old >= 0)
				PGmin_old = -INF;
		}
#endif
		double itertime = omp_get_wtime() - starttime;
		totaltime += itertime;
		//tesing after each step
		double primal_obj = 0, true_primal_obj = 0;
		double acc = 0, true_acc = 0;
		double err = 0;
		std::vector<double> tmpw(w_size, 0);
#pragma omp parallel for
		for(int ggg = 0; ggg < l; ggg++) {
			double yialpha = prob->y[ggg] * alpha[ggg];
			feature_node *xi = prob->x[ggg];
			while (xi->index != -1) {
#pragma omp atomic
				tmpw[xi->index-1] += xi->value*yialpha;
				xi++;
			}
		}
		double dual_obj = 0, true_dual_obj = 0;
		for(int ggg = 0; ggg < w_size; ggg++) {
			err += (tmpw[ggg]-w[ggg])*(tmpw[ggg]-w[ggg]);
			true_dual_obj += tmpw[ggg]*tmpw[ggg];
			dual_obj += w[ggg]*w[ggg];
		}
		for(int ggg = 0; ggg < l; ggg++) {
			true_dual_obj += alpha[ggg]*(alpha[ggg]*diag[GETI(ggg)]-2);
			dual_obj += alpha[ggg]*(alpha[ggg]*diag[GETI(ggg)]-2);
		}
		dual_obj *= 0.5;
		true_dual_obj *= 0.5;

		if(fun_obj) {
			primal_obj = fun_obj->fun(w);
			acc = fun_obj->testing(w)*100;
			true_primal_obj = fun_obj->fun(&(tmpw[0]));
			true_acc = fun_obj->testing(&(tmpw[0]))*100;
		}

		printf("iter %d walltime %lf itertime %lf f %.3lf d %.3lf acc %.1lf true-f %.3lf true-d %.3lf true-acc %.1lf err %.3g inittime %lf active_size %ld\n", 
				iter, totaltime, itertime, primal_obj, dual_obj, acc, true_primal_obj, true_dual_obj, true_acc, err, inittime, active_set.size());		

		fflush(stdout);
	}


	/*
	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\nUsing -s 2 may be faster (also see FAQ)\n\n");
		*/

	// calculate objective value

	/*
	double v = 0;
	int nSV = 0;
	for(int i=0; i<w_size; i++)
		v += w[i]*w[i];
	for(int i=0; i<l; i++)
	{
		v += alpha[i]*(alpha[i]*diag[GETI(i)] - 2);
		if(alpha[i] > 0)
			++nSV;
	}
	info("Objective value = %lf\n",v/2);
	info("nSV = %d\n",nSV);
	*/

	delete [] QD;
	delete [] alpha;
	delete [] tg;
	delete [] y;
	delete [] index;
} // }}} 

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)
static void solve_l2r_l1l2_svc_rf( 
	const problem *prob, double *w, double eps,
	double Cp, double Cn, int solver_type, function *fun_obj = NULL, int max_iter = 1000)
{ // {{{
	int l = prob->l;
	int w_size = prob->n;
	//int i, s, iter = 0;
	//double C, d, G;
	int iter = 0;
	double *QD = new double[l];
	int *index = new int[l];
	double *alpha = new double[l];
	double *tg = new double[w_size];
	schar *y = new schar[l];

	//int active_size = l;

	std::vector<int> active_set(l);
	//omp_lock_t writelock;
	//omp_lock_t *lockaaa = (omp_lock_t*)malloc(sizeof(omp_lock_t)*w_size);
//	vector<omp_lock_t > lockaaa(w_size);
//
	int nr_threads = omp_get_max_threads();

#ifdef SHRINKING
	aligned_vector<std::vector<int>> next_active_set(nr_threads);
	// PG: projected gradient, for shrinking and stopping
	double PGmax_old = INF;
	double PGmin_old = -INF;
	aligned_vector<double> PGmax_new(nr_threads);
	aligned_vector<double> PGmin_new(nr_threads);
#pragma omp parallel 
	{
		int tid = omp_get_thread_num();
		next_active_set[tid].reserve(l);
	}
#endif

	double tmp_start = omp_get_wtime();

	// default solver_type: L2R_L2LOSS_SVC_DUAL_RF
	double diag[3] = {0.5/Cn, 0, 0.5/Cp};
	double upper_bound[3] = {INF, 0, INF};
	if(solver_type == L2R_L1LOSS_SVC_DUAL_RF)
	{
		diag[0] = 0;
		diag[2] = 0;
		upper_bound[0] = Cn;
		upper_bound[2] = Cp;
	}

#pragma omp parallel for
	for(int i=0; i<l; i++)
	{
		if(prob->y[i] > 0)
			y[i] = +1;
		else
			y[i] = -1;
		active_set[i] = i;
		// Initial alpha can be set here. Note that
		// 0 <= alpha[i] <= upper_bound[GETI(i)]
		alpha[i] = 0;
	}

#pragma omp parallel for
	for(int i=0; i<w_size; i++)
		w[i] = 0;
#pragma omp parallel for
	for(int i=0; i<l; i++)
	{
		QD[i] = diag[GETI(i)];

		feature_node *xi = prob->x[i];
		while (xi->index != -1)
		{
			double val = xi->value;
			QD[i] += val*val;
			/*
			val*=y[i]*alpha[i];
			double &wi = w[xi->index-1];
#pragma omp atomic
			wi += val;
			*/
			xi++;
		}
		index[i] = i;
	}
	for (size_t i=0; i<l; i++) {
		size_t j = i+rand()%(l-i);
		std::swap(active_set[i], active_set[j]);
	}
	double inittime = omp_get_wtime() - tmp_start;
	std::vector<unsigned> seeds(nr_threads);
	for(int th = 0; th < nr_threads; th++)
		seeds[th] = th;

	double totaltime = 0, starttime = 0;
	while (iter < max_iter)
	{
		starttime = omp_get_wtime();
#ifdef SHRINKING
		//PGmax_new = -INF; PGmin_new = INF;
#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			PGmax_new[tid] = -INF;
			PGmin_new[tid] = INF;
			next_active_set[tid].clear();
		}
#endif 
		size_t active_size = active_set.size();

		if(0) {
			for (size_t i=0; i<active_size; i++)
			{
				int tid = omp_get_thread_num();
				size_t j = i+rand_r(&seeds[tid])%(active_size-i);
				//size_t j = i+rand()%(active_size-i);
				std::swap(active_set[i], active_set[j]);
			}
			/*
		} else if (nr_threads == 1) {
			for (size_t i=0; i<active_size; i++)
			{
				int tid = omp_get_thread_num();
				size_t j = i+rand_r(&seeds[tid])%(active_size-i);
				std::swap(active_set[i], active_set[j]);
			}*/
		} else {
#pragma omp parallel
			{
				int tid = omp_get_thread_num();
				size_t len = (active_size/nr_threads)+(active_size%nr_threads?1:0);
				size_t start = tid*len, end = std::min((tid+1)*len, active_size);
				for (size_t i=start; i<end; i++)
				{
					size_t j = i+rand_r(&seeds[tid])%(end-i);
					std::swap(active_set[i], active_set[j]);
				}
			}
		}


#pragma omp parallel for schedule(dynamic,8)
		for (size_t s=0; s<active_size; s++)
		{
			int i = active_set[s];
			double G = 0;
			schar yi = y[i];

			feature_node *xi = prob->x[i];
			while(xi->index!= -1)
			{
				G += w[xi->index-1]*(xi->value);
				xi++;
			}
			G = G*yi-1;

			double C = upper_bound[GETI(i)];
			G += alpha[i]*diag[GETI(i)];

#ifdef SHRINKING
			double PG = 0;
			if (alpha[i] == 0)
			{
				if (G > PGmax_old)
				{
				//	is_shrunken[i] = 1;
				//	active_size--;
				//	swap(index[s], index[active_size]);
				//  s--;
					continue;
				}
				else if (G < 0)
					PG = G;
			}
			else if (alpha[i] == C)
			{
				if (G < PGmin_old)
				{
				//	is_shrunken[i] = 1;
				//	active_size--;
				//	swap(index[s], index[active_size]);
				//  s--;
					continue;
				}
				else if (G > 0)
					PG = G;
			}
			else
				PG = G;

			int tid = omp_get_thread_num();
			//next_active_set.local().push_back(i);
			//PGmin_new.update(PG);
			//PGmax_new.update(PG);
			next_active_set[tid].push_back(i);
			PGmin_new[tid] = std::min(PGmin_new[tid], PG);
			PGmax_new[tid] = std::max(PGmax_new[tid], PG);
			
			if(fabs(PG) > 1.0e-12)
#endif
			{
				double alpha_old = alpha[i];
				alpha[i] = min(max(alpha[i] - G/QD[i], 0.0), C);

				double d = (alpha[i] - alpha_old)*yi;
				if(d != 0) {
					xi = prob->x[i];
					while (xi->index != -1)
					{
						int ind = xi->index-1;
						double val = d*xi->value;
//#pragma omp atomic
						w[ind] += val;

						xi++;
					}
				}
			}
		}

		iter++;
		//if(iter % 10 == 0) { info("."); }

#ifdef SHRINKING

		//double PGmax_new_global = PGmax_new.reduce();
		//double PGmax_new_global = PGmax_new.reduce();
		double PGmin_new_global = PGmin_new.min();
		double PGmax_new_global = PGmax_new.max();

		//printf("PGmax %g PGmin %g eps %g\n", PGmax_new_global, PGmin_new_global, eps);

		if(PGmax_new_global - PGmin_new_global <= eps)
		{
			/*
			if(active_size == l)
				break;
			else
			*/
			{
				//	active_size = l;
				//	info("*");
				//std::fill(is_shrunken.begin(), is_shrunken.end(), 0);
				active_set.resize(l);
				for(int ss = 0; ss < l; ss++) active_set[ss] = ss;
				PGmax_old = INF;
				PGmin_old = -INF;
				eps = eps*0.5;
				//	continue;
			}
		}
		else 
		{
			active_set.clear();
			/*
			for(auto &bag: next_active_set.buffer) {
				for(auto &ss : bag)
					active_set.push_back(ss);
			}
			*/
			for(auto &bag: next_active_set) {
				for(auto &ss: bag)
					active_set.push_back(ss);
			}

			PGmax_old = PGmax_new_global;
			PGmin_old = PGmin_new_global;
			if (PGmax_old <= 0)
				PGmax_old = INF;
			if (PGmin_old >= 0)
				PGmin_old = -INF;
		}
#endif
		double itertime = omp_get_wtime() - starttime;
		totaltime += itertime;
		//tesing after each step
		double primal_obj = 0, true_primal_obj = 0;
		double acc = 0, true_acc = 0;
		double err = 0;
		std::vector<double> tmpw(w_size, 0);
#pragma omp parallel for
		for(int ggg = 0; ggg < l; ggg++) {
			double yialpha = prob->y[ggg] * alpha[ggg];
			feature_node *xi = prob->x[ggg];
			while (xi->index != -1) {
#pragma omp atomic
				tmpw[xi->index-1] += xi->value*yialpha;
				xi++;
			}
		}
		double dual_obj = 0, true_dual_obj = 0;
		for(int ggg = 0; ggg < w_size; ggg++) {
			err += (tmpw[ggg]-w[ggg])*(tmpw[ggg]-w[ggg]);
			true_dual_obj += tmpw[ggg]*tmpw[ggg];
			dual_obj += w[ggg]*w[ggg];
		}
		for(int ggg = 0; ggg < l; ggg++) {
			true_dual_obj += alpha[ggg]*(alpha[ggg]*diag[GETI(ggg)]-2);
			dual_obj += alpha[ggg]*(alpha[ggg]*diag[GETI(ggg)]-2);
		}
		dual_obj *= 0.5;
		true_dual_obj *= 0.5;

		if(fun_obj) {
			primal_obj = fun_obj->fun(w);
			acc = fun_obj->testing(w)*100;
			true_primal_obj = fun_obj->fun(&(tmpw[0]));
			true_acc = fun_obj->testing(&(tmpw[0]))*100;
		}

		printf("iter %d walltime %lf itertime %lf f %.3lf d %.3lf acc %.1lf true-f %.3lf true-d %.3lf true-acc %.1lf err %.3g inittime %lf active_size %ld\n", 
				iter, totaltime, itertime, primal_obj, dual_obj, acc, true_primal_obj, true_dual_obj, true_acc, err, inittime, active_set.size());		

		fflush(stdout);
	}


	/*
	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\nUsing -s 2 may be faster (also see FAQ)\n\n");
		*/

	// calculate objective value

	/*
	double v = 0;
	int nSV = 0;
	for(int i=0; i<w_size; i++)
		v += w[i]*w[i];
	for(int i=0; i<l; i++)
	{
		v += alpha[i]*(alpha[i]*diag[GETI(i)] - 2);
		if(alpha[i] > 0)
			++nSV;
	}
	info("Objective value = %lf\n",v/2);
	info("nSV = %d\n",nSV);
	*/

	delete [] QD;
	delete [] alpha;
	delete [] tg;
	delete [] y;
	delete [] index;
} // }}} 

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)
static void solve_l2r_l1l2_svc_cocoa( 
	const problem *prob, double *w, double eps,
	double Cp, double Cn, int solver_type, function *fun_obj = NULL, int max_iter = 1000)
{ // {{{
	cpu_affinity_t cpu_info;
	int l = prob->l;
	int w_size = prob->n;
	//int i, s, iter = 0;
	//double C, d, G;
	int iter = 0;
	double *QD = new double[l];
	int *index = new int[l];
	double *alpha = new double[l];
	double *tg = new double[w_size];
	schar *y = new schar[l];

	double *global_w = new double[w_size];
	double *last_alpha = new double[l];

	//int active_size = l;
	std::vector<int> active_set(l);
	int nr_threads = omp_get_max_threads();

	// allocate per-package and per-thread storage
	std::vector<double*> local_w(nr_threads);
	std::vector<int> local_begin(nr_threads), local_end(nr_threads);
#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		local_w[tid] = (double*)cpu_info.thread_malloc(sizeof(double)*w_size, tid);
		local_begin[tid] = l;
		local_end[tid] = 0;
	}

	double tmp_start = omp_get_wtime();

	// default solver_type: L2R_L2LOSS_SVC_DUAL
	double diag[3] = {0.5/Cn, 0, 0.5/Cp};
	double upper_bound[3] = {INF, 0, INF};
	if(solver_type == L2R_L1LOSS_SVC_DUAL_COCOA)
	{
		diag[0] = 0;
		diag[2] = 0;
		upper_bound[0] = Cn;
		upper_bound[2] = Cp;
	}

#pragma omp parallel for schedule(static)
	for(int i=0; i<l; i++)
	{
		if(prob->y[i] > 0)
			y[i] = +1;
		else
			y[i] = -1;
		active_set[i] = i;
		// Initial alpha can be set here. Note that
		// 0 <= alpha[i] <= upper_bound[GETI(i)]
		alpha[i] = 0;
		last_alpha[i] = 0;
	}

	// initialize w and local_w
#pragma omp parallel for schedule(static)
	for(int i=0; i<w_size; i++)
		global_w[i] = 0;
#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		auto *w = local_w[tid];
		for(int i=0; i<w_size; i++)
			w[i] = 0;
	}
#pragma omp parallel for schedule(static)
	for(int i=0; i<l; i++)
	{
		QD[i] = diag[GETI(i)];

		feature_node *xi = prob->x[i];
		while (xi->index != -1)
		{
			double val = xi->value;
			QD[i] += val*val;
			/*
			val*=y[i]*alpha[i];
			double &wi = global_w[xi->index-1];
#pragma omp atomic
			wi += val;
			*/
			xi++;
		}
		index[i] = i;
		int tid = omp_get_thread_num();
		if(i < local_begin[tid]) local_begin[tid] = i;
		if(i >= local_end[tid]) local_end[tid] = i+1;
	}

	for (size_t i=0; i<l; i++) {
		size_t j = i+rand()%(l-i);
		std::swap(active_set[i], active_set[j]);
	}
	double inittime = omp_get_wtime() - tmp_start;

	std::vector<unsigned> seeds(nr_threads);
	for(int th = 0; th < nr_threads; th++)
		seeds[th] = th;

	double totaltime = 0, starttime = 0;
	while (iter < max_iter)
	{
		starttime = omp_get_wtime();
#pragma omp parallel 
		{
			int tid = omp_get_thread_num();
			double *w = local_w[tid];
			int begin = local_begin[tid], end = local_end[tid];
			size_t active_size = end-begin;
			size_t H = active_size; 
			for(size_t s=0;s<H;s++) {
				int i = index[begin+rand_r(&seeds[tid])%active_size];
				double G = 0;
				schar yi = y[i];

				feature_node *xi = prob->x[i];
				while(xi->index!= -1) {
					G += w[xi->index-1]*(xi->value);
					xi++;
				}
				G = G*yi-1;

				double C = upper_bound[GETI(i)];
				G += alpha[i]*diag[GETI(i)];

				double alpha_old = alpha[i];
				alpha[i] = min(max(alpha[i] - G/QD[i], 0.0), C);

				double d = (alpha[i] - alpha_old)*yi;
				if(d != 0) {
					xi = prob->x[i];
					while (xi->index != -1) {
						int ind = xi->index-1;
						double val = d*xi->value;
						w[ind] += val;
						xi++;
					}
				}
			}
		}
		
#pragma omp parallel for schedule(static)
		for(int i = 0; i < w_size; i++) {
			double sum = 0;
			for(int tid = 0; tid < nr_threads; tid++)
				sum += local_w[tid][i];
			sum /= nr_threads;
			for(int tid = 0; tid < nr_threads; tid++)
				local_w[tid][i] = sum;
		}

#pragma omp parallel for schedule(static)
		for(int i = 0; i < l; i++)
			alpha[i] = last_alpha[i] = last_alpha[i]+(alpha[i]-last_alpha[i])/nr_threads;

		iter++;
		totaltime += omp_get_wtime() - starttime;

		//testing after each iterations
		double primal_obj = 0, true_primal_obj = 0;
		double acc = 0, true_acc = 0;
		double err = 0;
		std::vector<double> tmpw(w_size, 0);
#pragma omp parallel for
		for(int ggg = 0; ggg < l; ggg++) {
			double yialpha = prob->y[ggg] * alpha[ggg];
			feature_node *xi = prob->x[ggg];
			while (xi->index != -1) {
#pragma omp atomic
				tmpw[xi->index-1] += xi->value*yialpha;
				xi++;
			}
		}
		double dual_obj = 0, true_dual_obj = 0;
		double *w = local_w[0];
		for(int ggg = 0; ggg < w_size; ggg++) {
			err += (tmpw[ggg]-w[ggg])*(tmpw[ggg]-w[ggg]);
			true_dual_obj += tmpw[ggg]*tmpw[ggg];
			//dual_obj += tmpw[ggg]*(tmpw[ggg]-w[ggg]);
			dual_obj += w[ggg]*w[ggg]; //(tmpw[ggg]-w[ggg]);
		}
		for(int ggg = 0; ggg < l; ggg++) {
			true_dual_obj += alpha[ggg]*(alpha[ggg]*diag[GETI(ggg)]-2);
			dual_obj += alpha[ggg]*(alpha[ggg]*diag[GETI(ggg)]-2);
		}
		dual_obj *= 0.5;
		true_dual_obj *= 0.5;

		if(fun_obj) {
			primal_obj = fun_obj->fun(w);
			acc = fun_obj->testing(w)*100;
			true_primal_obj = fun_obj->fun(&(tmpw[0]));
			true_acc = fun_obj->testing(&(tmpw[0]))*100;
		}

		printf("iter %d walltime %lf f %.3lf d %.3lf acc %.1lf true-f %.3lf true-d %.3lf true-acc %.1lf err %.3g inittime %lf active_size %ld\n", 
				iter, totaltime, primal_obj, dual_obj, acc, true_primal_obj, true_dual_obj, true_acc, err, inittime, active_set.size());		

		fflush(stdout);
	}


	/*
	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\nUsing -s 2 may be faster (also see FAQ)\n\n");
		*/

	// calculate objective value

	/*
	double v = 0;
	int nSV = 0;
	for(int i=0; i<w_size; i++)
		v += w[i]*w[i];
	for(int i=0; i<l; i++)
	{
		v += alpha[i]*(alpha[i]*diag[GETI(i)] - 2);
		if(alpha[i] > 0)
			++nSV;
	}
	info("Objective value = %lf\n",v/2);
	info("nSV = %d\n",nSV);
	*/

	for(int tid = 0; tid < nr_threads; tid++)
		free(local_w[tid]);
	delete [] QD;
	delete [] alpha;
	delete [] tg;
	delete [] y;
	delete [] index;
} // }}} 

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)
static void solve_l2r_l1l2_svc_ascd( 
	const problem *prob, double *w, double eps,
	double Cp, double Cn, int solver_type, function *fun_obj = NULL, int max_iter = 1000)
{ // {{{
	cpu_affinity_t cpu_info;
	int l = prob->l;
	int w_size = prob->n;
	//int i, s, iter = 0;
	//double C, d, G;
	int iter = 0;
	double *QD = new double[l];
	int *index = new int[l];
	double *alpha = new double[l];
	double *tg = new double[w_size];
	schar *y = new schar[l];
	double **Q = Malloc(double*, l);
	for(int i = 0; i < l; i++) 
		Q[i] = Malloc(double, l);

	int nr_threads = omp_get_max_threads();

	std::vector<int> local_begin(nr_threads, l), local_end(nr_threads, 0);

	double tmp_start = omp_get_wtime();
	// default solver_type: L2R_L2LOSS_SVC_DUAL_ASCD
	double diag[3] = {0.5/Cn, 0, 0.5/Cp};
	double upper_bound[3] = {INF, 0, INF};
	double Lmax = 0, r = 0.5; // step size 
	if(solver_type == L2R_L1LOSS_SVC_DUAL_ASCD) {
		diag[0] = 0;
		diag[2] = 0;
		upper_bound[0] = Cn;
		upper_bound[2] = Cp;
	}


	//int active_size = l;

	std::vector<int> active_set(l);

#pragma omp parallel for schedule(static)
	for(int i=0; i<l; i++) {
		if(prob->y[i] > 0)
			y[i] = +1;
		else
			y[i] = -1;
		active_set[i] = i;
		// Initial alpha can be set here. Note that
		// 0 <= alpha[i] <= upper_bound[GETI(i)]
		alpha[i] = 0;
		index[i] = i;
		int tid = omp_get_thread_num();
		if(i < local_begin[tid]) local_begin[tid] = i;
		if(i >= local_end[tid]) local_end[tid] = i+1;
	}

	// construction of Q
	for(int i = 0; i < l; i++) {
#pragma omp parallel for schedule(static)
		for(int j = 0; j < i; j++) 
			Q[i][j] = Q[j][i];
#pragma omp parallel for schedule(dynamic,32)
		for(int j = i; j < l; j++) {
			feature_node *xi = prob->x[i];
			feature_node *xj = prob->x[j];
			double dot = 0;
			while(xi->index != -1 && xj->index != -1) {
				if(xi->index == xj->index) {
					dot += xi->value * xj->value;
					xi++; 
					xj++;
				} else {
					if (xi->index > xj->index) xj++;
					else xi++;
				}
			}
			Q[i][j] = dot*y[i]*y[j];
			if(i == j)
				Q[i][j] += diag[GETI(i)];
		}
	}

	// See the first Eq. in page 4 of http://arxiv.org/pdf/1311.1873.pdf
	for(int i = 0 ; i < l; i++) 
		if(Q[i][i] > Lmax) 
			Lmax = Q[i][i];

	double inittime = omp_get_wtime() - tmp_start;

	std::vector<unsigned> seeds(nr_threads);
	for(int th = 0; th < nr_threads; th++)
		seeds[th] = th;

	double totaltime = 0, starttime = 0;
	while (iter < max_iter) {
		starttime = omp_get_wtime();
		// global permutation after 10 iters
		if((iter+1) % 10 == 0) {
			for(int i = 0; i < l; i++) {
				int j = i + rand()%(l-i);
				std::swap(index[i], index[j]);
			}
		}
#pragma omp parallel 
		{
			int tid = omp_get_thread_num();
			int begin = local_begin[tid];
			int end = local_end[tid];
			unsigned &seed = seeds[tid];
			double stepsize = r/Lmax;

			// local permutation
			for(int i = begin; i < end; i++) {
				int j = i+rand_r(&seed)%end-i;
				std::swap(index[i], index[j]);
			}

			for(int s = begin; s < end; s++) {
				int i = index[s];
				double G = -1;
				for(int t = 0; t < l; t++)
					G += Q[i][t]*alpha[t];
				double alpha_tmp = alpha[i]-stepsize*G;
				if(alpha_tmp < 0) 
					alpha[i] = 0;
				else if (alpha_tmp > upper_bound[GETI(i)])
					alpha[i] = upper_bound[GETI(i)];
				else 
					alpha[i] = alpha_tmp;
			}
		}
		iter++;
		totaltime += omp_get_wtime() - starttime;
		//testing after each step
		double primal_obj = 0, true_primal_obj = 0;
		double acc = 0, true_acc = 0;
		double err = 0;
		std::vector<double> tmpw(w_size, 0);
#pragma omp parallel for
		for(int ggg = 0; ggg < l; ggg++) {
			double yialpha = prob->y[ggg] * alpha[ggg];
			feature_node *xi = prob->x[ggg];
			while (xi->index != -1) {
#pragma omp atomic
				tmpw[xi->index-1] += xi->value*yialpha;
				xi++;
			}
		}
		double dual_obj = 0, true_dual_obj = 0;
		for(int ggg = 0; ggg < w_size; ggg++)
			dual_obj += tmpw[ggg]*tmpw[ggg];
		for(int ggg = 0; ggg < l; ggg++)
			dual_obj += alpha[ggg]*(alpha[ggg]*diag[GETI(ggg)]-2);
		dual_obj *= 0.5;
		true_dual_obj = dual_obj;

		if(fun_obj) {
			primal_obj = fun_obj->fun(&tmpw[0]);
			acc = fun_obj->testing(&tmpw[0])*100;
			true_primal_obj = primal_obj; 
			true_acc = acc;
		}

		printf("iter %d walltime %lf f %.3lf d %.3lf acc %.1lf true-f %.3lf true-d %.3lf true-acc %.1lf err %.3g inittime %lf active_size %ld\n", 
				iter, totaltime, primal_obj, dual_obj, acc, true_primal_obj, true_dual_obj, true_acc, err, inittime, active_set.size());		

		fflush(stdout);
	}


	/*
	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\nUsing -s 2 may be faster (also see FAQ)\n\n");
		*/

	// calculate objective value

	/*
	double v = 0;
	int nSV = 0;
	for(int i=0; i<w_size; i++)
		v += w[i]*w[i];
	for(int i=0; i<l; i++)
	{
		v += alpha[i]*(alpha[i]*diag[GETI(i)] - 2);
		if(alpha[i] > 0)
			++nSV;
	}
	info("Objective value = %lf\n",v/2);
	info("nSV = %d\n",nSV);
	*/

	for(int i = 0; i < l; i++) 
		free(Q[i]);
	free(Q);
	delete [] QD;
	delete [] alpha;
	delete [] tg;
	delete [] y;
	delete [] index;
} // }}} 
