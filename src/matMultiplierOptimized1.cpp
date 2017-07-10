/*
 * Optimized parallel matrix multiplication 
 * optimized using transpose and Strassen's algorithm
 * parallelized using openMP
 * Authors: Chamin Wickramarathna, Malsha Ranawaka
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

const char *filename = "results/optimized1-threshold-256.txt";    //file to store results of execution 
//const int NUM_THREADS = 2;		// number of threads for omp to run parallel for 
int thresholdSize = 256;  		// size at which the sequential multiplication is used instead of recursive Strassen
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
//compute parallel matrix multiplication using openMP parallel for
void multiplyMatParallel(vector< vector<double> > &a,vector< vector<double> > &b, vector< vector<double> > &c, int n);
// finding transpose of matrix b[][] and storing it in matrix btrans[][]
vector< vector<double> > transpose(vector< vector<double> > &b,vector< vector<double> > &btrans, int n);
// get the next smallest integer that is a power of two
int getNextPowerOfTwo(int n);
// pad with zeros if matrix size is not a power of two
void fillZeros(vector< vector<double> > &newA, vector< vector<double> > &newB, vector< vector<double> > &a, vector< vector<double> > &b, int n);
// adds two matrices
void add(vector< vector<double> > &a, vector< vector<double> > &b, vector< vector<double> > &resultMatrix, int n);
// subtracts two matrices
void subtract(vector< vector<double> > &a, vector< vector<double> > &b, vector< vector<double> > &resultMatrix, int n);
//parallelized Strassen's algorithm
void multiplyStrassenParallel(vector< vector<double> > &a,vector< vector<double> > &b, vector< vector<double> > &c, int n);
//sequential Strassen's algorithm
void multiplyStrassen(vector< vector<double> > &a,vector< vector<double> > &b, vector< vector<double> > &c, int n);


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
		thresholdSize = atoi(argv[1]);  //set threshold value if given by user
	}
	cout << "Threshold used for Strassen's: "<< thresholdSize <<endl;
	cout << endl;
	
	f << "Threshold used for Strassen's: "<< thresholdSize << "\n";
	
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
			multiplyStrassenParallel(a,b,c,n);
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

// get the next smallest integer that is a power of two
int getNextPowerOfTwo(int n){
	return pow(2, int(ceil(log2(n))));
}

// pad with zeros if matrix size is not a power of two
void fillZeros(vector< vector<double> > &newA, vector< vector<double> > &newB, vector< vector<double> > &a, vector< vector<double> > &b, int n){
	for (int i=0; i<n; i++){
		for (int j=0; j<n; j++){
			newA[i][j] = a[i][j];
			newB[i][j] = b[i][j];
		}
	}
}

// adds two matrices
void add(vector< vector<double> > &a, vector< vector<double> > &b, vector< vector<double> > &resultMatrix, int n){
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			resultMatrix[i][j] = a[i][j] + b[i][j];
		}
	}
}

// subtracts two matrices
void subtract(vector< vector<double> > &a, vector< vector<double> > &b, vector< vector<double> > &resultMatrix, int n){
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			resultMatrix[i][j] = a[i][j] - b[i][j];
		}
	}
}

//parallelized Strassen's algorithm
void multiplyStrassenParallel(vector< vector<double> > &a,vector< vector<double> > &b, vector< vector<double> > &c, int n){
	if(n<=thresholdSize){
		multiplyMatParallel(a, b, c, n);
	}
	else{
		//expand and fill with zeros if matrix size is not a power of two
		int newSize = getNextPowerOfTwo(n);
		vector< vector<double> > newA(newSize, vector<double>(newSize)), newB(newSize, vector<double>(newSize)), newC(newSize, vector<double>(newSize));
		if(n==newSize){   //matrix size is already a power of two
			newA = a;
			newB = b;
		}
		else{
			fillZeros(newA, newB, a, b, n);
		}
		
		//initialize submatrices
		int blockSize = newSize/2;  //size for a partition matrix
		vector<double> block (blockSize);
		vector< vector<double> > a11(blockSize, block), a12(blockSize, block), a21(blockSize, block), a22(blockSize, block), /*partitions of newA*/
			b11(blockSize, block), b12(blockSize, block), b21(blockSize, block), b22(blockSize, block), /*partitions of newB*/
			c11(blockSize, block), c12(blockSize, block), c21(blockSize, block), c22(blockSize, block), /*partitions of newC*/
			aBlock(blockSize, block), bBlock(blockSize, block),  /*matrices storing intermediate results*/
			m1(blockSize, block), m2(blockSize, block), m3(blockSize, block), m4(blockSize, block),  
			m5(blockSize, block), m6(blockSize, block), m7(blockSize, block);  /*set of submatrices derived from partitions*/
		
		//partition matrices
		int i, j;
		#pragma omp parallel shared(newA, newB, a11, a12, a21, a22, b11, b12, b21, b22) private(i,j)
		{
			#pragma omp for schedule(static)
			for (i=0; i<blockSize; i++){
				for (j=0; j<blockSize; j++){
					a11[i][j] = newA[i][j];
					a12[i][j] = newA[i][j+blockSize];
					a21[i][j] = newA[i+blockSize][j];
					a22[i][j] = newA[i+blockSize][j+blockSize];
					b11[i][j] = newB[i][j];
					b12[i][j] = newB[i][j+blockSize];
					b21[i][j] = newB[i+blockSize][j];
					b22[i][j] = newB[i+blockSize][j+blockSize];
				}
			}
		}
		
		//compute submatrices
		#pragma omp parallel sections
		{
			#pragma omp section
			{
				//m1 = (a11+a22)(b11+b22)
				add(a11, a22, aBlock, blockSize);
				add(b11, b22, bBlock, blockSize);
				multiplyStrassenParallel(aBlock, bBlock, m1, blockSize);
			}

			#pragma omp section
			{
				//m2 = (a21+a22)b11
				add(a21, a22, aBlock, blockSize);
				multiplyStrassenParallel(aBlock, b11, m2, blockSize);
			}
		
			#pragma omp section
			{
				//m3 = a11(b12-b22)
				subtract(b12, b22, bBlock, blockSize);
				multiplyStrassenParallel(a11, bBlock, m3, blockSize);
			}
			
			#pragma omp section
			{
				//m4 = a22(b21-b11)
				subtract(b21, b11, bBlock, blockSize);
				multiplyStrassenParallel(a22, bBlock, m4, blockSize);
			}
			
			#pragma omp section
			{
				//m5 = (a11+a12)b22
				add(a11, a12, aBlock, blockSize);
				multiplyStrassenParallel(aBlock, b22, m5, blockSize);
			}
			
			#pragma omp section
			{
				//m6 = (a21-a11)(b11+b12)
				subtract(a21, a11, aBlock, blockSize);
				add(b11, b12, bBlock, blockSize);
				multiplyStrassenParallel(aBlock, bBlock, m6, blockSize);
			}
			
			#pragma omp section
			{
				//m7 = (a12-a22)(b12+b22)
				subtract(a12, a22, aBlock, blockSize);
				add(b12, b22, bBlock, blockSize);
				multiplyStrassenParallel(aBlock, bBlock, m7, blockSize);
			}
		}
		
		//calculate result submatrices
		#pragma omp parallel sections
		{
			#pragma omp section
			{
				//c11 = m1+m4-m5+m7
				add(m1, m4, aBlock, blockSize);
				subtract(aBlock, m5, bBlock, blockSize);
				add(bBlock, m7, c11, blockSize);
			}
			
			#pragma omp section
			{
				//c12 = m3+m5
				add(m3, m5, c12, blockSize);
			}
			
			#pragma omp section
			{
				//c21 = m2+m4
				add(m2, m4, c12, blockSize);
			}
			
			#pragma omp section
			{
				//c22 = m1-m2+m3+m6
				subtract(m1, m2, aBlock, blockSize);
				add(aBlock, m3, bBlock, blockSize);
				add(bBlock, m6, c22, blockSize);
			}
		}
		
		//calculate final result matrix
		#pragma omp parallel shared(c11, c12, c21, c22, newC) private(i, j)
		{
			#pragma omp for schedule(static)
			for(i=0; i<blockSize; i++){
				for(j=0; j<blockSize; j++){
					newC[i][j] = c11[i][j];
					newC[i][blockSize+j] = c12[i][j];
					newC[blockSize+i][j] = c21[i][j];
					newC[blockSize+i][blockSize+j] = c22[i][j];
				}
			}
		}
		
		//remove additional values from expanded matrix
		#pragma omp parallel shared(c, newC) private(i, j)
		{
			#pragma omp for schedule(static)
			for(i=0; i<n; i++){
				for(j=0; j<n; j++){
					c[i][j] = newC[i][j];
				}
			}
		}
	}
}

//sequential Strassen's algorithm
void multiplyStrassen(vector< vector<double> > &a,vector< vector<double> > &b, vector< vector<double> > &c, int n){	
	if(n<=thresholdSize){
		multiplyMatSeq(a, b, c, n);
	}
	else{
		//expand and fill with zeros if matrix size is not a power of two
		int newSize = getNextPowerOfTwo(n);
		vector< vector<double> > newA(newSize, vector<double>(newSize)), newB(newSize, vector<double>(newSize)), newC(newSize, vector<double>(newSize));
		if(n==newSize){   //matrix size is already a power of two
			newA = a;
			newB = b;
		}
		else{
			fillZeros(newA, newB, a, b, n);
		}
		
		//initialize submatrices
		int blockSize = newSize/2;  //size for a partition matrix
		vector<double> block (blockSize);
		vector< vector<double> > a11(blockSize, block), a12(blockSize, block), a21(blockSize, block), a22(blockSize, block), /*partitions of newA*/
			b11(blockSize, block), b12(blockSize, block), b21(blockSize, block), b22(blockSize, block), /*partitions of newB*/
			c11(blockSize, block), c12(blockSize, block), c21(blockSize, block), c22(blockSize, block), /*partitions of newC*/
			aBlock(blockSize, block), bBlock(blockSize, block),  /*matrices storing intermediate results*/
			m1(blockSize, block), m2(blockSize, block), m3(blockSize, block), m4(blockSize, block),  
			m5(blockSize, block), m6(blockSize, block), m7(blockSize, block);  /*set of submatrices derived from partitions*/
		
		//partition matrices
		for (int i=0; i<blockSize; i++){
			for (int j=0; j<blockSize; j++){
				a11[i][j] = newA[i][j];
				a12[i][j] = newA[i][j+blockSize];
				a21[i][j] = newA[i+blockSize][j];
				a22[i][j] = newA[i+blockSize][j+blockSize];
				b11[i][j] = newB[i][j];
				b12[i][j] = newB[i][j+blockSize];
				b21[i][j] = newB[i+blockSize][j];
				b22[i][j] = newB[i+blockSize][j+blockSize];
			}
		}
		
		//compute submatrices
		//m1 = (a11+a22)(b11+b22)
		add(a11, a22, aBlock, blockSize);
		add(b11, b22, bBlock, blockSize);
		multiplyStrassen(aBlock, bBlock, m1, blockSize);
		
		//m2 = (a21+a22)b11
		add(a21, a22, aBlock, blockSize);
		multiplyStrassen(aBlock, b11, m2, blockSize);
		
		//m3 = a11(b12-b22)
		subtract(b12, b22, bBlock, blockSize);
		multiplyStrassen(a11, bBlock, m3, blockSize);
		
		//m4 = a22(b21-b11)
		subtract(b21, b11, bBlock, blockSize);
		multiplyStrassen(a22, bBlock, m4, blockSize);
		
		//m5 = (a11+a12)b22
		add(a11, a12, aBlock, blockSize);
		multiplyStrassen(aBlock, b22, m5, blockSize);
		
		//m6 = (a21-a11)(b11+b12)
		subtract(a21, a11, aBlock, blockSize);
		add(b11, b12, bBlock, blockSize);
		multiplyStrassen(aBlock, bBlock, m6, blockSize);
		
		//m7 = (a12-a22)(b12+b22)
		subtract(a12, a22, aBlock, blockSize);
		add(b12, b22, bBlock, blockSize);
		multiplyStrassen(aBlock, bBlock, m7, blockSize);
		
		//calculate result submatrices
		//c11 = m1+m4-m5+m7
		add(m1, m4, aBlock, blockSize);
		subtract(aBlock, m5, bBlock, blockSize);
		add(bBlock, m7, c11, blockSize);
		
		//c12 = m3+m5
		add(m3, m5, c12, blockSize);
		
		//c21 = m2+m4
		add(m2, m4, c12, blockSize);
		
		//c22 = m1-m2+m3+m6
		subtract(m1, m2, aBlock, blockSize);
		add(aBlock, m3, bBlock, blockSize);
		add(bBlock, m6, c22, blockSize);
		
		//calculate final result matrix
		for(int i=0; i<blockSize; i++){
			for(int j=0; j<blockSize; j++){
				newC[i][j] = c11[i][j];
				newC[i][blockSize+j] = c12[i][j];
				newC[blockSize+i][j] = c21[i][j];
				newC[blockSize+i][blockSize+j] = c22[i][j];
			}
		}
		
		//remove additional values from expanded matrix
		for(int i=0; i<n; i++){
			for(int j=0; j<n; j++){
				c[i][j] = newC[i][j];
			}
		}
	}
}