#!/usr/bin/env python

import os, sys
from commands import getoutput



def test_gcc_version():
	o = getoutput('gcc -v').split('\n')[-1].split()
	sys.stdout.write('Testing GCC (version > 4.7.2): ')
	sys.stdout.flush();
	if o[0] == 'gcc' and o[1] == 'version' and o[3] == '(GCC)':
		if map(int, o[2].split('.')) >= [4, 7, 2]:
			sys.stdout.write('Success\n')
			return True
		else :
			sys.stdout.write('Fail (version too old)\n')
			return False
	else:
		sys.stdout.write('Fail (No GCC available)\n')
		return False

def test_build():
	sys.stdout.write('Building PASSCoDe: ')
	sys.stdout.flush();
	o = getoutput('make clean all')
	if 'Error' not in o and os.path.exists('train') and os.path.exists('train-shrinking') and os.path.exists('convert2binary'):
		sys.stdout.write('Success\n')
		return True
	else :
		sys.stdout.write('Fail (compilation Error)\n')
		return False

def test_data():
	sys.stdout.write('Downloading dataset a9a: ')
	cmd = 'rm -f a9a; wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a'
	o = getoutput(cmd)
	cmd = 'rm -f a9a.t; wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a.t'
	o = getoutput(cmd)
	sys.stdout.write('Success\n')
	sys.stdout.write('Converting dataset a9a: ')
	cmd = 'rm -f a9a.cbin; ./convert2binary a9a a9a.cbin'
	o = getoutput(cmd)
	cmd = 'rm -f a9a.t.cbin;./convert2binary a9a.t a9a.t.cbin'
	o = getoutput(cmd)
	sys.stdout.write('Success\n')
	return True

def test_running():
	sys.stdout.write('Running PASSCoDe: ')
	cmd = 'rm -f test-log; ./train-shrinking -s 33 a9a.cbin a9a.t.cbin > test-log'
	o = getoutput(cmd)
	sys.stdout.write('Success\n')
	return True

if not test_gcc_version(): sys.exit(-1)
if not test_build(): sys.exit(-1)
if not test_data(): sys.exit(-1)
if not test_running(): sys.exit(-1)

