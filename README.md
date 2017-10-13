# Matrix Multiply

This c++ package demonstrate the multiplication of two random matrices(n x n) using sequential method and parallel method. It measure the average time taken to complete the process for each n value(200 - 2000).

```mat_multi_seq.cpp``` - Sequential matrix multiplication<br>
```mat_multi_para.cpp``` - Parallel matrix multiplication using OpenMP<br>
```mat_multi_seq_optimized_1``` - Optimized sequential program using transpose<br>
```mat_multi_para_optimized_1``` - Optimized parallel program using transpose<br>
```mat_multi_para_optimized_1.1``` - Optimized parallel program using transpose and pointer arrays<br>
```mat_multi_seq_para_optimized_2``` - Optimized program using Strassen algorithm and Transpose<br>
```mat_multi_seq_para_optimized_3``` - Optimized program using tiled algorithm and Transpose<br>

# Dependencies

This package relies on C / C++ compiler (gcc) and openMP.

## Linux

The GNU Compiler Collection may come with your distribution. Run `which gcc g++` to find out.<br>
If that command does not output as below, you will require to install it.
```shell
/usr/bin/gcc
/usr/bin/g++
```

For RHEL-based distros, run `sudo dnf install gcc gcc-c++`.

For Debian-based distros, run `sudo apt install gcc g++`.

For Arch-based distros, run `sudo pacman -S gcc`.

## Windows

You will require to install [MinGW](http://www.mingw.org/) and [add it to your PATH](https://www.howtogeek.com/118594/how-to-edit-your-system-path-for-easy-command-line-access/).

## Mac

You will require to install [XCode](https://developer.apple.com/xcode/).

# How to execute?

## Linux
Sequential versions :
```shell
$ g++ mat_multi_seq.cpp -o mat_multi_seq
$ ./mat_multi_seq
```
Parallel versions:
```shell
$ g++ -fopenmp mat_multi_seq.cpp -o mat_multi_seq
$ ./mat_multi_seq
```

## Windows
```shell
g++ -fopenmp matMultiply.cpp -o matMultiply
matMultiply
```

Both parallelized and optimized versions can be executed as above.
Upon execution, the execution times and the calculated speed up values will be displayed.

The ```mat_multi_seq_para_optimized_2.cpp``` file containing optimizing using Strassen algorithm can be supplied with a suitable integer threshold value at runtime as follows
```shell
mat_multi_seq_para_optimized_2 threshold-value
```

The default threshold value is 128.

