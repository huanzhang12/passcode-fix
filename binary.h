
#include "linear.h"

struct binary_problem {
	struct problem *prob;
	void *buf;
};


struct binary_problem read_binary(const char *file_name, double bias);
void destroy_binary_problem(struct binary_problem *binprob);
