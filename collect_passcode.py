#!/usr/bin/env python2

import os, sys, time, subprocess
from subprocess import check_call

dryrun = True

dataset = ["a9a", "music", "covtype", "webspam", "rcv1", "news20", "epsilon"]
nthreads = [1, 5, 10, 20]
algs = [31, 35, 51, 55]
outputdir = "passcode_" + time.strftime("%m%d-%H%M%S")
if not dryrun:
	check_call("mkdir -p {}/".format(outputdir), shell=True)

if len(sys.argv) > 1:
	if sys.argv[1] == "-n":
		dryrun = True

for d in dataset:
	for n in nthreads:
		for alg in algs:
			cmdline = "./train -s {} -t 100 -n {} data/{}_train.cbin data/{}_test.cbin".format(alg, n, d, d)
			result_name = os.path.join(outputdir, "{}_{}_{}.txt".format(alg, d, n))
			print "Executing {}\nResults at {}".format(cmdline, result_name)
			if not dryrun:
					result = subprocess.Popen(cmdline, shell=True, stdout=subprocess.PIPE).stdout.read()
					with open(result_name, "w") as f:
						f.write(result)



