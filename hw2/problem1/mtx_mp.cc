#include <ctime>
#include <iostream> 
#include <cstdlib>
#include <omp.h>

using namespace std;

int main(int argc, char* argv[]){
  volatile int a[100][100];
  volatile int b[100][100];
  volatile int c[100][100];
  
  struct timespec start, finish;
  double elapsed;

  for(int i=0;i<100;i++){
    for(int j=0;j<100;j++){
      a[i][j]=rand();
      b[i][j]=rand();
    }
  }
  
  clock_gettime(CLOCK_MONOTONIC, &start);
#pragma omp parallel for
  for(int i=0;i<100;i++){
    for(int j=0;j<100;j++){
      c[i][j]=a[i][j]*b[i][j];
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &finish);

  elapsed = (finish.tv_sec - start.tv_sec);
  elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

  cout<<"Time elapsed: "<<elapsed<<" seconds"<<endl;

  return 0;
}
