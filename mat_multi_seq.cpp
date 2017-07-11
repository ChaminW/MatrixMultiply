/*
 * Matrix Multiplication sequentially
 * Compile with -fopenmp flag
 * Author: Chamin Wickramarathna
 */

#include <iostream>
#include <fstream>
#include <ctime>
#include <sys/time.h>
#include <cstdlib>
#include <vector>
#include <omp.h>
#include <cmath>

using namespace std;

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
	
	
int main()
{
	const char *filename = "results/mat_multi_seq.txt";    //file to store results of execution 
	ofstream f;

	srand(time(0));   //seed for random number generation
	const int sampleSize = 1;      // Number of sample size consider to evaluate average time taken
	const int maxSize = 2000;       // maximum size of the 2d matrix
	double dtime;
	struct timeval startingTime, endingTime;
	double seqMean;
	double sd;
	double sampleCount;

	//open file to append
	f.open(filename, ios::trunc);
    if (!f.is_open()) {
        cout << "Unable to open the file" << endl;
        exit(1);
    }

	cout << "Sequential multiplication"<< endl;
	f << "Sequential multiplication\n";

	
	for (int n = 200; n <= maxSize; n+=200) {

		//vectors storing execution time values
		vector<double> seqTime(sampleSize);      
		vector<double> parTime(sampleSize);
		
		for (int k = 0; k < sampleSize; k++) {
			vector< vector<double> > a(n,vector<double>(n)),b(n,vector<double>(n)),c(n,vector<double>(n));	//c = a * b, c is the result matrix
			
			initMat(a,b,n);
			
			//sequential execution		
			gettimeofday(&startingTime, NULL);		//Get the starting time of execution
			multiplyMatSeq(a,b,c,n);
			gettimeofday(&endingTime, NULL);		//Get the ending time of execution of multiply function

			//Calculate the time taken to execute multiply function
			double dtime = ((endingTime.tv_sec  - startingTime.tv_sec) * 1000000u + endingTime.tv_usec - startingTime.tv_usec) / 1.e6;
			seqTime[k] = dtime;
			
		}
		
		seqMean = calculateMean(seqTime, sampleSize);
		sd = calculateSD(seqTime, sampleSize, seqMean);
		sampleCount = calculateSampleCount(seqMean, sd);
		
		cout << "Average time taken to execute in n-" << n << " : " << seqMean << endl;
		cout << "Standard deviation for execution in n-" << n << " : " << sd << endl;
		cout << "Sample count for n-" << n << " : " << sampleCount << endl;
		cout << endl;

		f << "\n--- n : " << n << " ---\n";
		f << "Average time taken to execute in n-" << n << " : " << seqMean << " seconds\n\n";		
		
	}
	f.close();
    return 0;
}


	
