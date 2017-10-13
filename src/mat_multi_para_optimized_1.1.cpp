/*
 * Parallel Matrix Multiplication using openMP with transpose - Optimization 1 - with pointer arrays
 * Compile with -fopenmp flag
 * Author: Chamin Wickramarathna
 */

#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <omp.h>
#include <cmath>

using namespace std;

const int NUM_THREADS = 4;

void initMat(double **a,double **b,int n){
		// Initialize arrays.
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				a[i][j] = (double)rand()/RAND_MAX*10;
				b[i][j] = (double)rand()/RAND_MAX*10;
				
			}
		}
	}

void multiplyMatParallel(double **a,double **b,double **c,int n){
		// Compute matrix multiplication.
		// C <- C + A x B
		// Use omp parrelle for loop

		double **btrans = (double **)malloc(n * sizeof(double *));
    	for (int i=0; i<n; i++){
         	btrans[i] = (double *)malloc(n * sizeof(double));
    	}

		for(int i = 0; i < n; ++i){
			for(int j = 0; j < n; ++j){
				btrans[j][i]=b[i][j];
			}
		}

		#pragma omp parallel for
		for (int i = 0; i < n; ++i) {
			#pragma omp parallel for
			for (int j = 0; j < n; ++j) {	
				double temp  = 0;
				for (int k = 0; k < n; ++k) {
					temp += a[i][k] * btrans[j][k];
				}
				c[i][j]=temp;
			}
		}
		
	}
	
double calculateMean(vector<double> data, int size) {
    double sum = 0.0, mean = 0.0;
    for (int i = 0; i < size; ++i) {
        sum += data[i];
    }

    mean = sum / size;
    return mean;
}

double calculateSD(vector<double> data, int size, double mean) {
    double sum = 0.0, standardDeviation = 0.0;

    for (int i = 0; i < size; ++i)
        standardDeviation += pow(data[i] - mean, 2);

    return sqrt(standardDeviation / size);
}

double calculateSampleCount(double mean, double sd) {
    double sample_count = 100 * 1.96 * sd / (5 * mean);
    return sample_count;
}
	
	
int main()
{
	const char *filename = "results/mat_multi_para_optimized_1.txt";    //file to store results of execution 
	ofstream f;

	srand(time(0));   //seed for random number generation
	const int sampleSize = 10;      // Number of sample size consider to evaluate average time taken
	const int maxSize = 2000;       // maximum size of the 2d matrix
	double dtime;
	double parMean;
	double sd;
	double sampleCount;
	
	//open file to append
	f.open(filename, ios::trunc);
    if (!f.is_open()) {
        cout << "Unable to open the file" << endl;
        exit(1);
    }

	cout << "Parallel multiplication using openMP - Optimized - v1"<< endl;
	f << "Parallel multiplication using openMP - Optimized - v1\n";

	for (int n = 200; n <= maxSize; n+=200) {

		//vectors storing execution time values
		vector<double> seqTime(sampleSize);      
		vector<double> parTime(sampleSize);
		
		for (int k = 0; k < sampleSize; k++) {
			double **a = (double **)malloc(n * sizeof(double *));
			double **b = (double **)malloc(n * sizeof(double *));
			double **c = (double **)malloc(n * sizeof(double *));

    		for (int i=0; i<n; i++){
         		a[i] = (double *)malloc(n * sizeof(double));
         		b[i] = (double *)malloc(n * sizeof(double));
         		c[i] = (double *)malloc(n * sizeof(double));
    		}

			initMat(a,b,n);
		
			//parallel execution
			dtime = 0;			
			dtime = omp_get_wtime();
			multiplyMatParallel(a,b,c,n);
			dtime = omp_get_wtime() - dtime;
			parTime[k] = dtime;

			//free memory
		    for(int i = 0; i< n; i++){   
		    	free(a[i]);
		    	free(b[i]);
		    	free(c[i]);
		    }
		    free(a);
		    free(b);
		    free(c);
		}
		
		parMean = calculateMean(parTime, sampleSize);
		sd = calculateSD(parTime, sampleSize, parMean);
		sampleCount = calculateSampleCount(parMean, sd);
		
		cout << "Average time taken to execute in n-" << n << " : " << parMean << endl;
		cout << "Standard deviation for execution in n-" << n << " : " << sd << endl;
		cout << "Sample count for n-" << n << " : " << sampleCount << endl;
		cout << endl;

		f << "\n--- n : " << n << " ---\n";
		f << "Average time taken to execute in n-" << n << " : " << parMean << " seconds\n\n";		
		
	}
	f.close();
    return 0;
}

