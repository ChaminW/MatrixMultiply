/*
 * Parallel matrix multiplication using openMP
 * Author: Chamin Wickramarathna
 */
 
#include <stddef.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <omp.h>
#include <cmath>
#include "meanCalculator.h"
 
using namespace std;

const char *filename = "results/parallel.txt";    //file to store results of execution 
//const int NUM_THREADS = 2;		// number of threads for omp to run parallel for 
const int sampleSize = 50;      // Number of sample size considered to evaluate average time taken
const int maxSize = 2000;       // maximum size of the 2d matrix
double startTime;
double elapsedTime;
double seqMean;
double parMean;
double sd;
double sampleCount;
ofstream f;

//initialize matrices
void initMat(vector< vector<double> > &a,vector< vector<double> > &b,int n);
//compute standard matrix multiplication
void multiplyMatSeq(vector< vector<double> > &a,vector< vector<double> > &b, vector< vector<double> > &c, int n);
//compute parallel matrix multiplication
void multiplyMatParallel(vector< vector<double> > &a,vector< vector<double> > &b, vector< vector<double> > &c, int n);

int main()
{
	srand(time(0));   //seed for random number generation
	
	//open file to append
	f.open(filename, ios::trunc);
    if (!f.is_open()) {
        cout << "Unable to open the file" << endl;
        exit(1);
    }
	
	//executing for each matrix size
	for (int n = 200; n <= maxSize; n+=200) {

		//vectors storing execution time values
		vector<double> seqTime(sampleSize);      
		vector<double> parTime(sampleSize);
		
		//executing for each sample 
		for (int k = 0; k < sampleSize; k++) {
						
			vector< vector<double> > a(n,vector<double>(n)),b(n,vector<double>(n)),c(n,vector<double>(n));	
			//c = a * b, c is the result matrix				
			initMat(a,b,n);
			
			//sequential execution		
			startTime = time(0);
			multiplyMatSeq(a,b,c,n);
			elapsedTime = time(0)-startTime;
			seqTime[k] = elapsedTime;
			
			//parallel execution
			startTime = time(0);
			multiplyMatParallel(a,b,c,n);
			elapsedTime = time(0)-startTime;
			parTime[k] = elapsedTime;			
		}
		
		cout << "--- n : " << n << " ---" << endl;
		cout << "Sequential multiplication"<< endl;
		seqMean = calculateMean(seqTime, sampleSize);
		sd = calculateSD(seqTime, sampleSize, seqMean);
		sampleCount = calculateSampleCount(seqMean, sd);
		
		cout << "Average time taken to execute in n-" << n << " : " << seqMean << " seconds" << endl;		
		cout << "Sample count for n-" << n << " : " << sampleCount << endl;
		cout << endl;
		
		f << "\n--- n : " << n << " ---\n";
		f << "Sequential multiplication\n";
		f << "Average time taken to execute in n-" << n << " : " << seqMean << " seconds\n";		
		f << "Sample count for n-" << n << " : " << sampleCount << "\n\n";
		
		cout << "Parallel multiplication using openMP"<< endl;
		parMean = calculateMean(parTime, sampleSize);
		sd = calculateSD(parTime, sampleSize, parMean);
		sampleCount = calculateSampleCount(parMean, sd);
		
		cout << "Average time taken to execute in n-" << n << " : " << parMean << " seconds" << endl;		
		cout << "Sample count for n-" << n << " : " << sampleCount << endl;
		cout << endl;
		
		f << "Parallel multiplication using openMP\n";
		f << "Average time taken to execute in n-" << n << " : " << parMean << " seconds\n";		
		f << "Sample count for n-" << n << " : " << sampleCount << "\n\n";
		
		cout << "Speed up after Parallelizing for n-" << n << " : " << seqMean/parMean << endl;
		cout << endl;
		
		f << "Speed up after Parallelizing for n-" << n << " : " << seqMean/parMean << "\n\n";		
	}
	f.close();
	
    return 0;
}

//initialize matrices
void initMat(vector< vector<double> > &a,vector< vector<double> > &b,int n){
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			a[i][j] = (double)rand()/RAND_MAX*10;
			b[i][j] = (double)rand()/RAND_MAX*10;			
		}
	}
}

//compute standard matrix multiplication
void multiplyMatSeq(vector< vector<double> > &a,vector< vector<double> > &b, vector< vector<double> > &c, int n){		
	// C <- C + A x B
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			double temp  = 0;
			for (int k = 0; k < n; ++k) {
				temp += a[i][k] * b[k][j];
			}
			c[i][j]=temp;
		}
	}
}	

//compute parallel matrix multiplication using openMP parallel for
void multiplyMatParallel(vector< vector<double> > &a,vector< vector<double> > &b, vector< vector<double> > &c, int n){
	 int i,j,k;
	#pragma omp parallel shared(a,b,c) private(i,j,k) 
	{
		#pragma omp for schedule(static)
		for (i = 0; i < n; ++i) {
			for (j = 0; j < n; ++j) {	
				double temp  = 0;
				for (k = 0; k < n; ++k) {
					temp += a[i][k] * b[k][j];
				}
				c[i][j]=temp;
			}
		}
	}
}