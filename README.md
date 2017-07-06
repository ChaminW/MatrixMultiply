# Matrix Multiply

This c++ package demonstrate the multiplication of two random matrices(n x n) using sequential method and parallel method. It measure the average time taken to complete the process for each n value(200 - 2000).

```matMultiply.cpp``` - Sequential matrix multiplication and its parallelization using OpenMP<br>
```matMultiplyOpt1.cpp``` - Optimized program using transpose<br>
```matMultiplyOpt2.cpp``` - Optimized program using Strassen

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
```shell
$ g++ -fopenmp matMultiply.cpp -o matMultiply<br>
$ ./matMultiply
```

## Windows
```shell
g++ -fopenmp matMultiply.cpp -o matMultiply<br>
matMultiply
```

Both parallelized and optimized versions can be executed as above.
Upon execution, the execution times and the calculated speed up values will be displayed.

The 'matMultiplyOpt2' file containing optimizing using Strassen algorithm can be supplied with a suitable integer threshold value as follows:
```shell
g++ -fopenmp matMultiply.cpp -o matMultiply<br>
matMultiply threshold-value
```

The default threshold value is 128.

