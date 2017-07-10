/*
 * Optimized parallel matrix multiplication 
 * optimized using transpose, cache blocking and loop unrolling
 * parallelized using openMP
 * Authors: Malsha Ranawaka
 */
 
#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <omp.h>
#include <cmath>
#include "meanCalculator.h"
 
using namespace std;

const char *filename = "results/optimized2-blocksize-256.txt";    //file to store results of execution 
//const int NUM_THREADS = 2;		// number of threads for omp to run parallel for 
int blockSize = 256;				// number of blocks to use when cache blocking 
const int sampleSize = 50;      // Number of sample size considered to evaluate average time taken
const int maxSize = 2000;       // maximum size of the 2d matrix
double startTime;
double elapsedTime;
double seqMean;
double optMean;
double sd;
double sampleCount;
ofstream f;

// initialize matrices
void initMat(vector< vector<double> > &a,vector< vector<double> > &b,int n);
// compute standard matrix multiplication
void multiplyMatSeq(vector< vector<double> > &a,vector< vector<double> > &b, vector< vector<double> > &c, int n);
// finding transpose of matrix b[][] and storing it in matrix btrans[][]
vector< vector<double> > transpose(vector< vector<double> > &b,vector< vector<double> > &btrans, int n);
// multiply the transpose with cache blocking (tiling) and loop unrolling 
void multiplyMatTiled(vector< vector<double> > &a,vector< vector<double> > &b, vector< vector<double> > &c, int n);


int main(int argc, char *argv[])
{
	srand(time(0));   //seed for random number generation
	
	//open file to append
	f.open(filename, ios::trunc);
    if (!f.is_open()) {
        cout << "Unable to open the file" << endl;
        exit(1);
    }
	
	if(argc>1){
		blockSize = atoi(argv[1]);  //set blockSize if given by user
	}
	cout << "Block size used for cache blocking: "<< blockSize <<endl;
	cout << endl;
	
	f << "Block size used for cache blocking: "<< blockSize << "\n";
	
	//executing for each matrix size
	for (int n = 200; n <= maxSize; n+=200) {
			
		//vectors storing execution time values
		vector<double> seqTime(sampleSize);       
		vector<double> optTime(sampleSize);
		
		//executing for each sample 
		for (int k = 0; k < sampleSize; k++) {
			
			vector< vector<double> > a(n,vector<double>(n)),b(n,vector<double>(n)),c(n,vector<double>(n));	//c = a * b, c is the result matrix			
			initMat(a,b,n);
			
			vector< vector<double> > btrans(n,vector<double>(n));
			btrans = transpose(b, btrans, n);
			
			//sequential execution
			startTime = time(0);
			multiplyMatSeq(a,b,c,n);
			elapsedTime = time(0)-startTime;
			seqTime[k] = elapsedTime;
			
			//optimized execution 	
			startTime = time(0);
			multiplyMatTiled(a,btrans,c,n);
			elapsedTime = time(0)-startTime;
			optTime[k] = elapsedTime;			
			
		}
		
		cout << "--- n : " << n << " ---" << endl;
		cout << "Sequential multiplication"<< endl;
		seqMean = calculateMean(seqTime, sampleSize);
		sd = calculateSD(seqTime, sampleSize, seqMean);
		sampleCount = calculateSampleCount(seqMean, sd);
		
		cout << "Average time taken to execute in n-" << n << " : " << seqMean << " seconds" << endl;		
		cout << "Sample count for n-" << n << " : " << sampleCount << endl;
		cout << endl;
		
		f << "--- n : " << n << " ---\n";
		f << "Sequential multiplication\n";
		f << "Average time taken to execute in n-" << n << " : " << seqMean << " seconds\n";		
		f << "Sample count for n-" << n << " : " << sampleCount << "\n\n";
		
		cout << "Optimized parallel multiplication"<< endl;		
		optMean = calculateMean(optTime, sampleSize);
		sd = calculateSD(optTime, sampleSize, optMean);
		sampleCount = calculateSampleCount(optMean, sd);
		
		cout << "Average time taken to execute in n-" << n << " : " << optMean << " seconds" << endl;		
		cout << "Sample count for n-" << n << " : " << sampleCount << endl;
		cout << endl;
		
		f << "Optimized parallel multiplication\n";
		f << "Average time taken to execute in n-" << n << " : " << optMean << " seconds\n";		
		f << "Sample count for n-" << n << " : " << sampleCount << "\n\n";
		
		cout << "Speed up after parallelized optimization for n-" << n << " : " << seqMean/optMean << endl;
		cout << endl;
		
		f << "Speed up after Parallelizing for n-" << n << " : " << seqMean/optMean << "\n\n";	
		
	}
	f.close();
    
	return 0;
}

// initialize matrices
void initMat(vector< vector<double> > &a,vector< vector<double> > &b,int n){
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			a[i][j] = (double)rand()/RAND_MAX*10;
			b[i][j] = (double)rand()/RAND_MAX*10;			
		}
	}
}

// compute standard matrix multiplication
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

// finding transpose of matrix b[][] and storing it in matrix btrans[][]
vector< vector<double> > transpose(vector< vector<double> > &b,vector< vector<double> > &btrans, int n) {	
	int i,j;
	#pragma omp parallel shared(b,btrans) private(i,j)
	{
		#pragma omp for schedule(static)
		for(i = 0; i < n; ++i){	
			for(j = 0; j < n; ++j){
				btrans[j][i]=b[i][j];
			}
		}
	}
	return btrans;
}

// multiply the transpose with cache blocking (tiling) and loop unrolling 
void multiplyMatTiled(vector< vector<double> > &a,vector< vector<double> > &b, vector< vector<double> > &c, int n){	
	int ii,jj,kk,j,k;
	#pragma omp parallel shared(a,b,c,blockSize) private(ii,jj,kk,j,k)
	{
		#pragma omp for schedule(static)		
		 for(int ii=0; ii<n; ii += blockSize){
			for(int jj=0; jj<n; jj += blockSize){
				for(int kk=0; kk<n; kk += blockSize){
					for (int i = ii; i < min(ii+blockSize, n); ++i) {
						for (int j = jj; j < min(jj+blockSize, n); ++j) {										
							for (int k = kk; k < min(kk+blockSize, n); k+=8) {
								//do loop unrolling 
								c[i][j] +=
									 + a[i][k]*b[j][k]
									 + a[i][k+1]*b[j][k+1]
									 + a[i][k+2]*b[j][k+2]
									 + a[i][k+3]*b[j][k+3]
									 + a[i][k+4]*b[j][k+4]
									 + a[i][k+5]*b[j][k+5]
									 + a[i][k+6]*b[j][k+6]
									 + a[i][k+7]*b[j][k+7];
							}							
						}
					}
				}
			}
		}
	}
}

