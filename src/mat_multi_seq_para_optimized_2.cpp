/*
 * Parallel Matrix Multiplication using openMP
 * Compile with -fopenmp flag
 * Author: Malsha Ranawaka
 */

 //optimized matrix multiplication using Strassen algorithm and Transpose
 
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <omp.h>
#include <cmath>

using namespace std;

int thresholdSize = 128;  //size at which the sequential multiplication is used instead of recursive Strassen
int NUM_THREADS = 1;

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
	
vector< vector<double> > transpose(vector< vector<double> > &b,vector< vector<double> > &btrans, int n) {
	// Finding transpose of matrix b[][] and storing it in matrix btrans[][]
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
	
int main(int argc, char *argv[])
{
	srand(time(0));   //seed for random number generation
	
	const int sampleSize = 10;      // Number of sample size consider to evaluate average time taken
	const int maxSize = 1000;       // maximum size of the 2d matrix
	double dtime;
	double seqMean;
	double strMean;
	double parMean;
	double sd;
	double sampleCount;
	
	if(argc>1){
		thresholdSize = atoi(argv[1]);  //set threshold value if given by user
	}
	
	//for (int n = 200; n <= maxSize; n+=200) {
		int n = 1000;		
		//vectors storing execution time values
		vector<double> seqTime(sampleSize);      
		vector<double> strTime(sampleSize);  
		vector<double> parTime(sampleSize);
		
		for (int k = 0; k < sampleSize; k++) {
			vector< vector<double> > a(n,vector<double>(n)),b(n,vector<double>(n)),c(n,vector<double>(n));	//c = a * b, c is the result matrix
			
			initMat(a,b,n);
			vector< vector<double> > btrans(n,vector<double>(n));
			btrans = transpose(b, btrans, n);
			
			//sequential execution
			dtime = 0;			
			dtime = omp_get_wtime();
			multiplyMatSeq(a,b,c,n);
			dtime = omp_get_wtime() - dtime;
			seqTime[k] = dtime;
			
			//strassen execution			
			dtime = 0;			
			dtime = omp_get_wtime();
			multiplyStrassen(a,btrans,c,n);
			dtime = omp_get_wtime() - dtime;			
			strTime[k] = dtime;
			
			//parallel execution
			dtime = 0;			
			dtime = omp_get_wtime();
			multiplyStrassenParallel(a,btrans,c,n);
			dtime = omp_get_wtime() - dtime;
			parTime[k] = dtime;
		}
		cout << endl;
		cout << "Sequential multiplication"<< endl;
		seqMean = calculateMean(seqTime, sampleSize);
		sd = calculateSD(seqTime, sampleSize, seqMean);
		sampleCount = calculateSampleCount(seqMean, sd);
		
		cout << "Average time taken to execute in n-" << n << " : " << seqMean << endl;
		cout << "Standard deviation for execution in n-" << n << " : " << sd << endl;
		cout << "Sample count for n-" << n << " : " << sampleCount << endl;
		cout << endl;
		
		cout << "Strassen multiplication"<< endl;
		cout << "Threshold used: "<< thresholdSize <<endl;
		strMean = calculateMean(strTime, sampleSize);
		sd = calculateSD(strTime, sampleSize, strMean);
		sampleCount = calculateSampleCount(strMean, sd);
		
		cout << "Average time taken to execute in n-" << n << " : " << strMean << endl;
		cout << "Standard deviation for execution in n-" << n << " : " << sd << endl;
		cout << "Sample count for n-" << n << " : " << sampleCount << endl;
		cout << "Speed up after Strassen with trnspose for n-" << n << " : " << seqMean/strMean << endl;
		cout << endl;
		
		cout << "Parallel strassen with transpose using openMP"<< endl;
		parMean = calculateMean(parTime, sampleSize);
		sd = calculateSD(parTime, sampleSize, parMean);
		sampleCount = calculateSampleCount(parMean, sd);
		
		cout << "Average time taken to execute in n-" << n << " : " << parMean << endl;
		cout << "Standard deviation for execution in n-" << n << " : " << sd << endl;
		cout << "Sample count for n-" << n << " : " << sampleCount << endl;
		cout << "Speed up after Parallelizing with strassen for n-" << n << " : " << seqMean/parMean << endl;
		cout << endl;
		
	//}//
    
	return 0;
}


	
