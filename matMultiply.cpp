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
		#pragma omp parallel for
		for (int i = 0; i < n; ++i) {
			#pragma omp parallel for
			for (int j = 0; j < n; ++j) {
				double temp  = 0;
				#pragma omp parallel for
				for (int k = 0; k < n; ++k) {
					temp += a[i][k] * b[k][j];
				}
				c[i][j]=temp;
			}
		}
	}
int main()
{
	cout << "Sequential multiplication"<< endl;
	for (int n = 200; n < 2000; n+=200) {
		vector< vector<double> > a(n,vector<double>(n)),b(n,vector<double>(n)),c(n,vector<double>(n));	//c = a * b, c is the result matrix
		
		initMat(a,b,n);
		
		dtime = omp_get_wtime();
		multiplyMatSeq(a,b,c,n);
		dtime = omp_get_wtime() - dtime;
		cout << "Time taken to execute in n-"<< n << " : "<< dtime << endl;
	}
	
	cout << "Parallel multiplication using openMP"<< endl;
	for (int n = 200; n < 2000; n+=200) {
		vector< vector<double> > a(n,vector<double>(n)),b(n,vector<double>(n)),c(n,vector<double>(n));	//c = a * b, c is the result matrix
		
		initMat(a,b,n);
		
		dtime = omp_get_wtime();
		multiplyMatParallel(a,b,c,n);
		dtime = omp_get_wtime() - dtime;
		cout << "Time taken to execute in n-"<< n << " : "<< dtime << endl;
	}
    return 0;
}


	
