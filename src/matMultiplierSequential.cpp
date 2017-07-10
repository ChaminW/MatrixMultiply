/*
 * Sequential matrix multiplication 
 * Author: Chamin Wickramarathna
 */
 
#include <stddef.h>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <omp.h>
#include <cmath>
#include "meanCalculator.h"
 
using namespace std;

const int sampleSize = 20;      // Number of sample size considered to evaluate average time taken
const int maxSize = 2000;       // maximum size of the 2d matrix
double startTime;
double elapsedTime;
double seqMean;
double sd;
double sampleCount;

//initialize matrices
void initMat(vector< vector<double> > &a,vector< vector<double> > &b,int n);
//compute standard matrix multiplication
void multiplyMatSeq(vector< vector<double> > &a,vector< vector<double> > &b, vector< vector<double> > &c, int n);
 
int main()
{
	srand(time(0));   //seed for random number generation
	
	//executing for each matrix size
	for (int n = 200; n <= maxSize; n+=200) {

		//vector storing execution time values
		vector<double> seqTime(sampleSize);      
		
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
		}
		
		cout << "--- n : " << n << " ---" << endl;
		cout << "Sequential multiplication"<< endl;
		seqMean = calculateMean(seqTime, sampleSize);
		sd = calculateSD(seqTime, sampleSize, seqMean);
		sampleCount = calculateSampleCount(seqMean, sd);
		
		cout << "Average time taken to execute in n-" << n << " : " << seqMean << " seconds" << endl;		
		cout << "Sample count for n-" << n << " : " << sampleCount << endl;
		cout << endl;
		
	}
	
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