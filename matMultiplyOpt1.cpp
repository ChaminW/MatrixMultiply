/*
 * Parallel Matrix Multiplication using openMP with transpose - Optimization 1
 * Compile with -fopenmp flag
 * Author: Chamin Wickramarathna
 */

#include <iostream>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <omp.h>
#include <cmath>

using namespace std;

const int NUM_THREADS = 8;


void initMat(double **a,double **b,int n){
		// Initialize arrays.
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				a[i][j] = (double)rand()/RAND_MAX*10;
				b[i][j] = (double)rand()/RAND_MAX*10;
				
			}
		}
	}
void transpose(double **b,double **btrans,int n) {
		// Finding transpose of matrix b[][] and storing it in matrix btrans[][]
		int i,j;
		#pragma omp parallel num_threads(NUM_THREADS) shared(b,btrans) private(i,j)
		//#pragma omp parallel
		for(i = 0; i < n; ++i)
			#pragma omp for nowait
			for(j = 0; j < n; ++j)
			{
				btrans[j][i]=b[i][j];
			}
	}
	
void multiplyTMatSeq(double **a,double **b,double **c,int n){
		// Compute matrix multiplication.
		// Transpose the B matrix
		// C <- C + A x Btrans
		double **btrans = (double **)malloc(n * sizeof(double *));
    	for (int i=0; i<n; i++){
         	btrans[i] = (double *)malloc(n * sizeof(double));
    	}
    	
    	//getting transpose of the matrix b
    	int i,j;
		for(i = 0; i < n; ++i){
			for(j = 0; j < n; ++j)
			{
				btrans[j][i]=b[i][j];
			}
		}

		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				double temp  = 0;
				for (int k = 0; k < n; ++k) {
					temp += a[i][k] * btrans[j][k];
				}
				c[i][j]=temp;
			}
		}
		for(int i = 0; i< n; i++){   
		    free(btrans[i]);
		}
		free(btrans);
	}

	void multiplyTMatParallel(double **a,double **b,double **c,int n){
		// Compute matrix multiplication.
		// Transpose the B matrix
		// C <- C + A x Btrans
		// Use omp parrelle for loop
		double **btrans = (double **)malloc(n * sizeof(double *));
    	for (int i=0; i<n; i++){
         	btrans[i] = (double *)malloc(n * sizeof(double));
    	}

		transpose(b,btrans,n);
		int i,j,k;
		#pragma omp parallel num_threads(NUM_THREADS) default(none) shared(a,btrans,c,n) private(i,j,k)
		//#pragma omp parallel for
		
			//#pragma omp for schedule(static)
			
			for (i = 0; i < n; ++i) {
				#pragma omp for nowait
				for (j = 0; j < n; ++j) {	
					double temp  = 0;
					for (k = 0; k < n; ++k) {
						temp += a[i][k] * btrans[j][k];
					}
					c[i][j]=temp;
				}
			}
		
		for(int i = 0; i< n; i++){   
		    free(btrans[i]);
		}
		free(btrans);
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
	srand(time(0));   //seed for random number generation
	
	const int sampleSize = 20;      // Number of sample size consider to evaluate average time taken
	const int maxSize = 2000;       // maximum size of the 2d matrix
	double dtime;
	double seqMean;
	double parMean;
	double sd;
	double sampleCount;
	
	for (int n = 200; n <= maxSize; n+=200) {

		//vectors storing execution time values
		vector<double> seqTime(sampleSize);      
		vector<double> parTime(sampleSize);


		
		for (int k = 0; k < sampleSize; k++) {
			//vector< vector<double> > a(n,vector<double>(n)),b(n,vector<double>(n)),c(n,vector<double>(n));	//c = a * b, c is the result matrix
			double **a = (double **)malloc(n * sizeof(double *));
			double **b = (double **)malloc(n * sizeof(double *));
			double **c = (double **)malloc(n * sizeof(double *));

    		for (int i=0; i<n; i++){
         		a[i] = (double *)malloc(n * sizeof(double));
         		b[i] = (double *)malloc(n * sizeof(double));
         		c[i] = (double *)malloc(n * sizeof(double));
    		}

			initMat(a,b,n);
			
			//sequential execution		
			dtime = omp_get_wtime();
			multiplyTMatSeq(a,b,c,n);
			dtime = omp_get_wtime() - dtime;
			//cout << "Time taken to execute in n-"<< n << " : "<< dtime << endl;
			seqTime[k] = dtime;
			
			//parallel execution
			dtime = 0;			
			dtime = omp_get_wtime();
			multiplyTMatParallel(a,b,c,n);
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
		cout << "Sequential multiplication"<< endl;
		seqMean = calculateMean(seqTime, sampleSize);
		sd = calculateSD(seqTime, sampleSize, seqMean);
		sampleCount = calculateSampleCount(seqMean, sd);
		
		cout << "Average time taken to execute in n-" << n << " : " << seqMean << endl;
		cout << "Standard deviation for execution in n-" << n << " : " << sd << endl;
		cout << "Sample count for n-" << n << " : " << sampleCount << endl;
		cout << endl;
		
		cout << "Parallel multiplication using openMP"<< endl;
		parMean = calculateMean(parTime, sampleSize);
		sd = calculateSD(parTime, sampleSize, parMean);
		sampleCount = calculateSampleCount(parMean, sd);
		
		cout << "Average time taken to execute in n-" << n << " : " << parMean << endl;
		cout << "Standard deviation for execution in n-" << n << " : " << sd << endl;
		cout << "Sample count for n-" << n << " : " << sampleCount << endl;
		cout << endl;
		
		cout << "Speed up after Parallelizing for n-" << n << " : " << seqMean/parMean << endl;
		cout << endl;
	}
	
    return 0;
}


	
