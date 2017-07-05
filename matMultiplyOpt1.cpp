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

using namespace std;

double dtime;

void initMat(vector< vector<double> > &a,vector< vector<double> > &b,int n){
		// Initialize arrays.
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				a[i][j] = (double)rand()/RAND_MAX*10;
				b[i][j] = (double)rand()/RAND_MAX*10;
				
			}
		}
	}
void transpose(vector< vector<double> > &b, vector< vector<double> > &btrans, int n) {
		// Finding transpose of matrix b[][] and storing it in matrix btrans[][]
		for(int i = 0; i < n; ++i)
			for(int j = 0; j < n; ++j)
			{
				btrans[j][i]=b[i][j];
			}
	}
	
void multiplyTMatSeq(vector< vector<double> > &a,vector< vector<double> > &b, vector< vector<double> > &c, int n){
		// Compute matrix multiplication.
		// Transpose the B matrix
		// C <- C + A x Btrans
		vector< vector<double> > btrans(n,vector<double>(n));
		transpose(b,btrans,n);
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				double temp  = 0;
				for (int k = 0; k < n; ++k) {
					temp += a[i][k] * btrans[j][k];
				}
				c[i][j]=temp;
			}
		}
	}

	void multiplyTMatParallel(vector< vector<double> > &a,vector< vector<double> > &b, vector< vector<double> > &c, int n){
		// Compute matrix multiplication.
		// Transpose the B matrix
		// C <- C + A x Btrans
		// Use omp parrelle for loop
		vector< vector<double> > btrans(n,vector<double>(n));
		transpose(b,btrans,n);
		 int i,j,k;
		#pragma omp parallel shared(a,b,c) private(i,j,k) 
		{
			#pragma omp for schedule(static)
			for (i = 0; i < n; ++i) {
				for (j = 0; j < n; ++j) {	
					double temp  = 0;
					for (k = 0; k < n; ++k) {
						temp += a[i][k] * btrans[j][k];
					}
					c[i][j]=temp;
				}
			}
		}
	}
	
	
int main()
{
	srand(time(0));   //seed for random number generation
	
	const int matrixCount = 10;     //no of matrix sizes taken into account
	const int sampleSize = 20;      // Number of sample size consider to evaluate average time taken
	const int maxSize = 2000;       // maximum size of the 2d matrix
	
	//vectors storing execution time values
	vector<double> trnTime(matrixCount);      
	vector<double> trnParTime(matrixCount);
	
	int count = 0;
	
	cout << "Sequential multiplication with transpose - Optimization 1"<< endl;
	count = 0;
	for (int n = 200; n <= maxSize; n+=200) {
		double total_time = 0;
		for (int k = 0; k < sampleSize; k++) {
			vector< vector<double> > a(n,vector<double>(n)),b(n,vector<double>(n)),btrans(n,vector<double>(n)),c(n,vector<double>(n));	//c = a * b, c is the result matrix
			
			initMat(a,b,n);
			
			dtime = omp_get_wtime();
			multiplyTMatSeq(a,b,c,n);
			dtime = omp_get_wtime() - dtime;
			//cout << "Time taken to execute in n-"<< n << " : "<< dtime << endl;
			total_time+= dtime;
		}
		cout << "Average time taken to execute in n-"<< n << " : "<< total_time/sampleSize << endl;
		trnTime[count] = total_time/sampleSize;
		count++;
	}
	
	cout << "Parallel multiplication using openMP with transpose - Optimization 1"<< endl;
	count = 0;
	for (int n = 200; n <= maxSize; n+=200) {
		double total_time = 0;
		for (int k = 0; k < sampleSize; k++) {
			vector< vector<double> > a(n,vector<double>(n)),b(n,vector<double>(n)),btrans(n,vector<double>(n)),c(n,vector<double>(n));	//c = a * b, c is the result matrix
			
			initMat(a,b,n);
			
			dtime = omp_get_wtime();
			multiplyTMatParallel(a,b,c,n);
			dtime = omp_get_wtime() - dtime;
			//cout << "Time taken to execute in n-"<< n << " : "<< dtime << endl;
			total_time+= dtime;
		}
		cout << "Average time taken to execute in n-"<< n << " : "<< total_time/sampleSize << endl;
		trnParTime[count] = total_time/sampleSize;
		count++;
	}
	
	cout << "Speed up calculations - Using transpose" << endl;
	int n = 200;
	for(int i=0; i<matrixCount; i++){
		cout << "Speed up after using transpose for n-" << n << " : " << trnTime[i]/trnParTime[i] << endl;
		n+=200;
	}
	
    return 0;
}


	
