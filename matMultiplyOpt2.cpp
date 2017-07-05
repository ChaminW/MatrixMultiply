/*
 * Parallel Matrix Multiplication using openMP
 * Compile with -fopenmp flag
 * Author: Malsha Ranawaka
 */

 //optimized matrix multiplication using Strassen algorithm
 
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <omp.h>
#include <cmath>

using namespace std;

double dtime;
int thresholdSize = 128;  //size at which the sequential multiplication is used instead of recursive Strassen

void initMat(vector< vector<double> > &a,vector< vector<double> > &b,int n){
		// Initialize arrays.
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				a[i][j] = (double)rand()/RAND_MAX*10;
				b[i][j] = (double)rand()/RAND_MAX*10;
				
			}
		}
	}
	
void multiplyMatSeq(vector< vector<double> > &a,vector< vector<double> > &b, vector< vector<double> > &c, int n){
		// Compute matrix multiplication.
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

void multiplyMatParallel(vector< vector<double> > &a,vector< vector<double> > &b, vector< vector<double> > &c, int n){
		// Compute matrix multiplication.
		// C <- C + A x B
		// Use omp parrelle for loop
		 int i,j,k;
		#pragma omp parallel shared(a,b,c) private(i,j,k) //spawn threads
		{
			#pragma omp for schedule(static)   //divide loop iterations among threads
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
	
int getNextPowerOfTwo(int n){
	return pow(2, int(ceil(log2(n))));
}

void fillZeros(vector< vector<double> > &newA, vector< vector<double> > &newB, vector< vector<double> > &a, vector< vector<double> > &b, int n){
	for (int i=0; i<n; i++){
		for (int j=0; j<n; j++){
			newA[i][j] = a[i][j];
			newB[i][j] = b[i][j];
		}
	}
}

void fillZerosParallel(vector< vector<double> > &newA, vector< vector<double> > &newB, vector< vector<double> > &a, vector< vector<double> > &b, int n){
	int i, j;
	#pragma omp parallel shared(a,b,newA,newB) private(i,j)
	{
		#pragma omp for schedule(static)
		for (i=0; i<n; i++){
			for (j=0; j<n; j++){
				newA[i][j] = a[i][j];
				newB[i][j] = b[i][j];
			}
		}
	}
}

void add(vector< vector<double> > &a, vector< vector<double> > &b, vector< vector<double> > &resultMatrix, int n){
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			resultMatrix[i][j] = a[i][j] + b[i][j];
		}
	}
}

void subtract(vector< vector<double> > &a, vector< vector<double> > &b, vector< vector<double> > &resultMatrix, int n){
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			resultMatrix[i][j] = a[i][j] - b[i][j];
		}
	}
}
	
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
			fillZerosParallel(newA, newB, a, b, n);
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
				multiplyStrassen(aBlock, bBlock, m1, blockSize);
			}

			#pragma omp section
			{
				//m2 = (a21+a22)b11
				add(a21, a22, aBlock, blockSize);
				multiplyStrassen(aBlock, b11, m2, blockSize);
			}
		
			#pragma omp section
			{
				//m3 = a11(b12-b22)
				subtract(b12, b22, bBlock, blockSize);
				multiplyStrassen(a11, bBlock, m3, blockSize);
			}
			
			#pragma omp section
			{
				//m4 = a22(b21-b11)
				subtract(b21, b11, bBlock, blockSize);
				multiplyStrassen(a22, bBlock, m4, blockSize);
			}
			
			#pragma omp section
			{
				//m5 = (a11+a12)b22
				add(a11, a12, aBlock, blockSize);
				multiplyStrassen(aBlock, b22, m5, blockSize);
			}
			
			#pragma omp section
			{
				//m6 = (a21-a11)(b11+b12)
				subtract(a21, a11, aBlock, blockSize);
				add(b11, b12, bBlock, blockSize);
				multiplyStrassen(aBlock, bBlock, m6, blockSize);
			}
			
			#pragma omp section
			{
				//m7 = (a12-a22)(b12+b22)
				subtract(a12, a22, aBlock, blockSize);
				add(b12, b22, bBlock, blockSize);
				multiplyStrassen(aBlock, bBlock, m7, blockSize);
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
	
int main()
{
	srand(time(0));   //seed for random number generation
	
	const int matrixCount = 10;   //no of matrix sizes taken into account
	const int sampleSize = 20;    //no of samples to evaluate average time taken
	const int maxSize = 2000;     //maximum size of the matrix 
	
	//vectors storing execution time values
	vector<double> strTime(matrixCount);      
	vector<double> strParTime(matrixCount);
	
	int count = 0;
	
	cout << "Sequential multiplication with Strassen - Optimization 2"<< endl;
	count = 0;
	for (int n = 200; n <= maxSize; n+=200) {
		double total_time = 0;
		for (int k = 0; k < sampleSize; k++) {
			vector< vector<double> > a(n,vector<double>(n)),b(n,vector<double>(n)),c(n,vector<double>(n));	//c = a * b, c is the result matrix
			
			initMat(a,b,n);
			dtime = omp_get_wtime();  //get current time
			multiplyStrassen(a,b,c,n);
			dtime = omp_get_wtime() - dtime;  //get execution time
			//cout << "Time taken to execute in n-"<< n << " : "<< dtime << endl;
			total_time+= dtime;
		}
		cout << "Average time taken to execute in n-"<< n << " : "<< total_time/sampleSize << endl;
		strTime[count] = total_time/sampleSize;
		count++;
	}
	
	cout << "Parallel multiplication using openMP with Strassen - Optimization 2"<< endl;
	count = 0;
	for (int n = 200; n <= maxSize; n+=200) {
		double total_time = 0;
		for (int k = 0; n < sampleSize; n++) {
			vector< vector<double> > a(n,vector<double>(n)),b(n,vector<double>(n)),c(n,vector<double>(n));	//c = a * b, c is the result matrix
			
			initMat(a,b,n);
			
			dtime = omp_get_wtime();
			multiplyStrassenParallel(a,b,c,n);
			dtime = omp_get_wtime() - dtime;
			//cout << "Time taken to execute in n-"<< n << " : "<< dtime << endl;
			total_time+= dtime;
		}
		cout << "Average time taken to execute in n-"<< n << " : "<< total_time/sampleSize << endl;
		strParTime[count] = total_time/sampleSize;
		count++;
	}  
	
	cout << "Speed up calculations - Using Strassen" << endl;
	int n = 200;
	for(int i=0; i<matrixCount; i++){
		cout << "Speed up using Strassen for n-" << n << " : " << strTime[i]/strParTime[i] << endl;
		n+=200;
	}
    
	return 0;
}


	
