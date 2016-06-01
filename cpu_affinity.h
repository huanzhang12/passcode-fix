/*
 * All Copyright Reserved 
 *
 * Author: Hsiang-Fu Yu (rofuyu@cs.utexas.edu)
 *
 * */

#ifndef CPU_AFFINITY_H
#define CPU_AFFINITY_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cstdio>
#include <sched.h>    // for sched_setaffinity/sched_getaffinity
#include <unistd.h>   // for sysconf
#include <vector>
#include <map>
#include <algorithm>

#include "aligned_storage.h"

class cpu_affinity_t { // {{{
	public:
	int num_procs;
	int num_pkgs;
	int num_real_procs;
	size_t page_size;
	
	std::vector<int> proc_id;
	std::vector<int> pkg_id;
	std::vector<int> pkg_leader;

	cpu_affinity_t() { parse_cpuinfo(); get_pagesize();}
	bool thread_binding(int tid) const { // {{{
		int pid = proc_id[tid%num_procs];
		cpu_set_t mask;
		CPU_ZERO(&mask);
		CPU_SET(pid, &mask);
		if(sched_setaffinity(0, sizeof(cpu_set_t), &mask) == -1) {
			fprintf(stdout,"Warning: Could not get affinity for thread %d (binding to proc-id %d", tid, pid);
			return false;
		}
		return true;
	}// }}}
	int get_proc_id(int tid) const {return proc_id[tid%num_procs];}
	int get_pkg_id(int tid) const {return pkg_id[tid%num_procs];}
	int get_leader(int tid) const {return pkg_leader[pkg_id[tid%num_procs]];}
	bool is_leader(int tid) const {return get_leader(tid)==tid;}
	int active_pkgs(int nr_threads) { // {{{
		std::vector<int> active(num_pkgs);
		for(int th=0; th < nr_threads; th++) 
			active[pkg_id[th]] = 1;
		int num_active_pkgs = 0;
		for(int pkg=0; pkg < num_pkgs; pkg++)
			num_active_pkgs += active[pkg];
		return num_active_pkgs;
	} // }}}

	void *pagein_malloc(size_t bytes) const { // {{{
		void *ret = malloc(bytes);
		if(ret) {
			volatile char* ptr = (char*)ret;
			for(register size_t i = 0; i < bytes; i += page_size) 
				ptr[i] = 0;
		}
		return ret;
	}// }}}
	void* thread_malloc(size_t bytes, int tid) const { // {{{
		cpu_set_t old_mask;
		sched_getaffinity(0, sizeof(cpu_set_t), &old_mask);
		thread_binding(tid);
		void *ret = pagein_malloc(bytes);
		sched_setaffinity(0, sizeof(cpu_set_t), &old_mask);
		return ret;
	} // }}}
	void* package_malloc(size_t bytes, int pkg_id) const { // {{{
		return thread_malloc(bytes, pkg_leader[pkg_id]);
	} // }}}
	void* interleaved_malloc(size_t bytes, int active_pkgs=0) const { // {{{
		if(active_pkgs==0) active_pkgs = num_pkgs;
		active_pkgs = std::max(active_pkgs, num_pkgs);
		cpu_set_t old_mask;
		void *ret = malloc(bytes);
		sched_getaffinity(0, sizeof(cpu_set_t), &old_mask);
		for(int i = 0; i < active_pkgs; i++) {
			thread_binding(pkg_leader[i]);
			size_t start = i*page_size;
			size_t stride = page_size*active_pkgs;
			register char *ptr = (char*)ret;
			for(size_t j = start; j < bytes; j += stride) 
				ptr[j] = 0;
		}
		sched_setaffinity(0, sizeof(cpu_set_t), &old_mask);
		return ret;
	} // }}}

	private:
	size_t get_pagesize() { return page_size = sysconf(_SC_PAGESIZE); }
	struct cpuinfo_t{ // {{{
		int proc_id, phy_id, sib, core_id, cpu_cores, ht;
		void clear() {proc_id=phy_id=sib=core_id=cpu_cores=ht=0;}
		void print(FILE *fp) {
			fprintf(fp, "proc(%d), phy(%d), core_id(%d) ht(%d)\n", proc_id, phy_id, core_id, ht);
		}
		bool operator<(const cpuinfo_t &m) const {
			if(ht != m.ht) return ht < m.ht;
			if(phy_id != m.phy_id) return phy_id < m.phy_id;
			if(core_id != m.core_id) return core_id < m.core_id;
			return proc_id < m.proc_id;
		}
		// test wheather hyperthreaded with another logical processor
		bool ht_with(const cpuinfo_t &m) const {
			return (phy_id == m.phy_id and core_id == m.core_id);
		}
	}; // }}}
	void parse_cpuinfo() { // {{{
		FILE *fp = fopen("/proc/cpuinfo","r");
		const size_t len = 1024;
		char buf[len];
		cpuinfo_t cpuinfo;
		std::vector<cpuinfo_t> proc_list;
		std::map<int,int>pkg_mapping;
		num_pkgs = 0;
		bool first = true;
		while(fgets(buf, len, fp)) { // {{{
			int val;
			if(sscanf(buf,"processor\t: %d", &val) == 1) {
				if(first) first = false;
				else proc_list.push_back(cpuinfo);
				cpuinfo.clear();
				cpuinfo.proc_id = val;
			} else if(sscanf(buf, "physical id\t: %d", &val) == 1) {
				std::map<int,int>::iterator it = pkg_mapping.find(val);
				if(it == pkg_mapping.end()) pkg_mapping[val] = num_pkgs++;
				cpuinfo.phy_id = pkg_mapping[val];
			} else if(sscanf(buf, "siblings\t: %d", &val) == 1) {
				cpuinfo.sib = val;
			} else if(sscanf(buf, "core id\t: %d", &val) == 1) {
				cpuinfo.core_id = val;
			} else if(sscanf(buf, "cores\t: %d", &val) == 1) {
				cpuinfo.cpu_cores = val;
			}
		} // }}}
		proc_list.push_back(cpuinfo);
		num_procs = static_cast<int>(proc_list.size());
		std::sort(proc_list.begin(), proc_list.end());
		// handling hyperthreading.
		for(int i = 1; i < num_procs; i++) {
			if(proc_list[i].ht_with(proc_list[i-1])) 
				proc_list[i].ht = proc_list[i-1].ht+1;
			else 
				proc_list[i].ht = 0;
		}
		// sort it again to move all hyperthreads to the bottom
		std::sort(proc_list.begin(), proc_list.end());
		proc_id.resize(num_procs, 0);
		pkg_id.resize(num_procs, 0);
		pkg_leader.resize(num_pkgs, -1);
		num_real_procs = 0;
		for(int i = 0; i < num_procs; i++) {
			proc_id[i] = proc_list[i].proc_id;
			pkg_id[i] = proc_list[i].phy_id;
			if(proc_list[i].ht==0) 
				num_real_procs++;
			if(pkg_leader[pkg_id[i]] == -1) 
				pkg_leader[pkg_id[i]] = i;
		}
	} // }}}
}; // }}}

#endif // CPU_AFFINITY_H
