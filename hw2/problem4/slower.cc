#include <cstdio>
#include <ctime>
#include <iostream> 
#include <cstdlib>
#include <omp.h>

#define dim 10000
#define count 10 

using namespace std;

int main(int argc, char* argv[]){
  int a[dim];
  
  struct timespec start, finish;
  double elapsed;

  for(int i=0;i<dim;i++){
    a[i]=1;
  }
  clock_gettime(CLOCK_MONOTONIC, &start);
  for(int i=0;i<count;i++){
#pragma omp parallel for
    for(int j=0;j<dim;j++){
#pragma omp critical
      a[j]++;      
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &finish);

  elapsed = (finish.tv_sec - start.tv_sec);
  elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

  cout<<"Time elapsed: "<<elapsed<<" seconds"<<endl;

  return 0;
}
