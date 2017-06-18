#include <iostream>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <omp.h>


using namespace std;


clock_t start, end;
double msecs;

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
				for (int k = 0; k < n; ++k) {
					c[i][j] += a[i][k] * b[k][j];
					//cout << c[i][j] << " ,";
					
				}
				//cout << endl;
			}
		}
	}

void multiplyMatParallel(vector< vector<double> > &a,vector< vector<double> > &b, vector< vector<double> > &c, int n){
		// Compute matrix multiplication.
		// C <- C + A x B
		#pragma omp parallel for
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				for (int k = 0; k < n; ++k) {
					c[i][j] += a[i][k] * b[k][j];
					//cout << c[i][j] << " ,";
					
				}
				//cout << endl;
			}
		}
	}
int main()
{
	
	cout << "Sequential multiplication"<< endl;
	for (int n = 200; n < 2000; n+=200) {
		vector< vector<double> > a(n,vector<double>(n)),b(n,vector<double>(n)),c(n,vector<double>(n));	//c = a * b, c is the result matrix
		
		initMat(a,b,n);
		
		start = clock();
		multiplyMatSeq(a,b,c,n);
		end = clock();
		msecs = ((double) (end - start)) * 1000 / CLOCKS_PER_SEC;
		cout << "Time taken to execute in n-"<< n << " : "<< msecs << endl;
	}
	
	cout << "Parallel multiplication using openMP"<< endl;
	for (int n = 200; n < 2000; n+=200) {
		vector< vector<double> > a(n,vector<double>(n)),b(n,vector<double>(n)),c(n,vector<double>(n));	//c = a * b, c is the result matrix
		
		initMat(a,b,n);
		
		start = clock();
		multiplyMatParallel(a,b,c,n);
		end = clock();
		msecs = ((double) (end - start)) * 1000 / CLOCKS_PER_SEC;
		cout << "Time taken to execute in n-"<< n << " : "<< msecs << endl;
	}
    return 0;
}


	
