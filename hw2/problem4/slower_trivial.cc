#include <cstdio>
#include <ctime>
#include <iostream> 
#include <cstdlib>
#include <omp.h>

#define count 100000

using namespace std;

int main(int argc, char* argv[]){
  struct timespec start, finish;
  double elapsed;

  clock_gettime(CLOCK_MONOTONIC, &start);
  int nThreads=omp_get_max_threads();
#pragma omp parallel for
  for(int i=0;i<nThreads;i++){
    int temp=0;
    for(int j=0;j<nThreads*count;j++){
      temp++;
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &finish);
//  cerr<<endl;
  elapsed = (finish.tv_sec - start.tv_sec);
  elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

  cout<<"Time elapsed: "<<elapsed<<" seconds"<<endl;

  return 0;
}
