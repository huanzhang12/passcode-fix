PASSCoDe-fix
========================

PASSCoDe-fix is based on the PASSCoDe (Parallel ASynchronous Stochastic dual
Co-ordinate Descent) algorithm, and we aim to fix the divergence problem of
PASSCoDe in highly parallel situations.  Please note that the current version
only supports binary classification (with label +1 and -1), and only
implemented limited dual solvers in the LIBLINEAR (see Usage below). For more
details about this algorithm please refer to the following papers:

```
Fixing the Convergence Problems in Parallel Asynchronous Dual Coordinate Descent,
Huan Zhang and Cho-Jui Hsieh, 2016. 

PASSCoDe: Parallel ASynchronous Stochastic dual Co-ordinate Descent, 
C.-J. Hsieh, H.-F. Yu, and I. S. Dhillon, 2015. 
```

Build
---------------

We require the following environment to build PASSCoDe-fix:

- GNU Compiler Collection (GCC) 4.7.1 or newer versions, with C++11 and OpenMP
  support
- Unix Systems (If you are in Mac OS, please install GCC instead of the LLVM
  compiler shipped with the Xcode command line tools.)

To build the program, simply run `make`. Two binaries, `train` (for training
without shrinking) and `train-shrink` (for training with shrinking) will be
built.  In the PASSCoDe-fix paper, we did not use shrinking in our experiments
and analysis.

Data Preparation 
----------------

Please download the datasets from LIBSVM datasets
http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
and convert them to the binary format used for PASSCoDe-fix. 

```
./convert2binary training_set_file [training_binary]
```

You can also use LIBSVM format directly by adding the argument "-b 0".
However data loading will be slower.

We have prepared binary files used in the experiments of our paper.
You can download these datasets here:

[http://jaina.cs.ucdavis.edu/datasets/classification_compressed/](http://jaina.cs.ucdavis.edu/datasets/classification_compressed/)

You only need to download .cbin.xz files. To save downloading time these files
are compressed. Please decompress them using the `xz` utility before use.

Usage
----------------

The new solvers added in this version are PASSCoDe-Atomic-fix and
PASSCoDe-Wild-fix for L1-loss and L2-loss support vector classifications.  They
can be invoked by set the type of solver to 55, 57, 35 and 37, respectively.
Please note that in our paper only ATOMIC-fix based algorithms are analyzed,
however Wild-fix might give you best performance for certain datasets.

```
./train[-shink] [options] training_set_file test_set_file
options:
-s type : set type of solver (default 31)
	31 -- L2-regularized L2-loss support vector classification PASSCoDe-Wild (dual)
	33 -- L2-regularized L1-loss support vector classification PASSCoDe-Wild (dual)
	35 -- L2-regularized L2-loss support vector classification PASSCoDe-Wild-fix (dual)
	37 -- L2-regularized L1-loss support vector classification PASSCoDe-Wild-fix (dual)
	41 -- L2-regularized L2-loss support vector classification PASSCoDe-LOCK (dual)
	43 -- L2-regularized L1-loss support vector classification PASSCoDe-LOCK (dual)
	51 -- L2-regularized L2-loss support vector classification PASSCoDe-ATOMIC (dual)
	53 -- L2-regularized L1-loss support vector classification PASSCoDe-ATOMIC (dual)
	55 -- L2-regularized L2-loss support vector classification PASSCoDe-ATOMIC-fix (dual)
	57 -- L2-regularized L1-loss support vector classification PASSCoDe-ATOMIC-fix (dual)
	61 -- L2-regularized L2-loss support vector classification CoCoA (dual)
	63 -- L2-regularized L1-loss support vector classification CoCoA (dual)
	71 -- L2-regularized L2-loss support vector classification ASCD (dual)
	73 -- L2-regularized L1-loss support vector classification ASCD (dual)
-c cost : set the parameter C (default 1)
-n nr_threads : the number of threads
-t max_iterations: the max number of iterations (default 100)
-e epsilon : set tolerance of termination criterion
-b binary_mode : if binary_mode = 1, read binary format (default 1)
```


Additional Information
----------------------

If your have any questions or comments, please open an issue on Github,
or send an email to ecezhang@ucdavis.edu. We appreciate your feedback.

