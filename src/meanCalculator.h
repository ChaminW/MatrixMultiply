
//functions for computing distribution parameters

#include <stddef.h>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <cmath>

using namespace std; 

//calculates mean of a given vector
double calculateMean(vector<double> data, int size) {
    double sum = 0.0, mean = 0.0;
    for (int i = 0; i < size; ++i) {
        sum += data[i];
    }

    mean = sum / size;
    return mean;
}

//calculates standard deviation of a given vector
double calculateSD(vector<double> data, int size, double mean) {
    double sum = 0.0, standardDeviation = 0.0;

    for (int i = 0; i < size; ++i)
        standardDeviation += pow(data[i] - mean, 2);

    return sqrt(standardDeviation / size);
}

//calculates sample count
double calculateSampleCount(double mean, double sd) {
    double sample_count = 100 * 1.96 * sd / (5 * mean);
    return sample_count;
}