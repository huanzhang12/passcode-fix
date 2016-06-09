CXX=g++
CC=gcc
CFLAGS = -Wall -Wconversion -Wno-sign-compare -Wno-unused-function -O3 -fPIC -std=c++11 -fopenmp 
CFLAGS = -Wall -Wconversion -Wno-sign-compare -Wno-unused-function -O2 -march=native -fPIC -std=c++11 -fopenmp 
#CFLAGS = -Wall -Wconversion -O2 -march=native -fPIC -openmp 
LIBS = blas/blas.a zlib/libz.a 
SHVER = 1
OS = $(shell uname)
#LIBS = -lblas

all: train train-shrink convert2binary

lib: linear.o tron.o blas/blas.a zlib/libz.a
	if [ "$(OS)" = "Darwin" ]; then \
		SHARED_LIB_FLAG="-dynamiclib -Wl,-install_name,liblinear.so.$(SHVER)"; \
	else \
		SHARED_LIB_FLAG="-shared -Wl,-soname,liblinear.so.$(SHVER)"; \
	fi; \
	$(CXX) $${SHARED_LIB_FLAG} linear.o tron.o blas/blas.a -o liblinear.so.$(SHVER)

train: tron.o linear.o train.c blas/blas.a binary.o zlib/libz.a
	$(CXX) $(CFLAGS) -o train train.c tron.o linear.o binary.o $(LIBS)

train-shrink: tron.o linear-shrinking.o train.c blas/blas.a binary.o
	$(CXX) $(CFLAGS) -o train-shrinking train.c tron.o linear-shrinking.o binary.o $(LIBS)

convert2binary: convert2binary.cpp
	$(CXX) $(CXXFLAGS) -o convert2binary convert2binary.cpp zlib/libz.a

predict: tron.o linear.o predict.c blas/blas.a
	$(CXX) $(CFLAGS) -o predict predict.c tron.o linear.o $(LIBS)

tron.o: tron.cpp tron.h
	$(CXX) $(CFLAGS) -c -o tron.o tron.cpp

linear.o: linear.cpp linear.h icml-passcode.cpp
	$(CXX) $(CFLAGS) -c -o linear.o linear.cpp

linear-shrinking.o: linear.cpp linear.h icml-passcode.cpp
	$(CXX) $(CFLAGS) -DSHRINKING -c -o linear-shrinking.o linear.cpp

binary.o: binary.cpp linear.h binary.h 
	$(CXX) $(CFLAGS) -c -o binary.o binary.cpp

zlib/libz.a:
	cd zlib; make libz.a;

blas/blas.a: blas/*.c blas/*.h
	make -C blas OPTFLAGS='$(CFLAGS)' CC='$(CC)';

clean:
	make -C blas clean
	make -C zlib clean
	rm -f *~ tron.o linear.o train predict liblinear.so.$(SHVER) binary.o convert2binary
	rm -f train-shrinking linear-shrinking.o test
