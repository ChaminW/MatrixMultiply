# Matrix Multiply

This c++ package demostrate the multiplication of two random matrices(n x n) using sequential method and parallel method. It measure the time taken to complete the process for each n value(200 - 2000).

# Dependencies

This package relies on a C / C++ compiler (gcc).

## Linux

The GNU Compiler Collection may come with your distribution. Run `which gcc g++` to find out.<br>
If that command does not output
```shell
/usr/bin/gcc
/usr/bin/g++
```

you will need to install it.

For RHEL-based distros, run `sudo dnf install gcc gcc-c++`.

For Debian-based distros, run `sudo apt install gcc g++`.

For Arch-based distros, run `sudo pacman -S gcc`.

## Windows

You'll need to install [MinGW](http://www.mingw.org/) and [add it to your PATH](https://www.howtogeek.com/118594/how-to-edit-your-system-path-for-easy-command-line-access/).

## Mac

You'll need to install [XCode](https://developer.apple.com/xcode/).

# How to execute?

$ g++ -fopenmp matMultiply.cpp -o matMultiply<br>
$ ./matMultiply
