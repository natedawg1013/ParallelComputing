#include <ctime>
#include <iostream> 
#include <cstdlib>
#include <omp.h>

using namespace std;

#define MTX_DIM 100

//Matrices are row-major

int main(int argc, char* argv[]){
  volatile int a[MTX_DIM][MTX_DIM];
  volatile int b[MTX_DIM][MTX_DIM];
  volatile int c[MTX_DIM][MTX_DIM];
  
  struct timespec start, finish;
  double elapsed;

  for(int i=0;i<MTX_DIM;i++){
    for(int j=0;j<MTX_DIM;j++){
      a[i][j]=rand();
      b[i][j]=rand();
    }
  }
  
  clock_gettime(CLOCK_MONOTONIC, &start);
#pragma omp parallel for
  for(int i=0;i<MTX_DIM;i++){
#pragma omp parallel for
    for(int j=0;j<MTX_DIM;j++){
      for(int k=0;k<MTX_DIM;k++){
        c[i][j]+=a[i][k]*b[k][j];
      }
      c[i][j]=a[i][j]*b[i][j];
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &finish);

  elapsed = (finish.tv_sec - start.tv_sec);
  elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

  cout<<"Time elapsed: "<<elapsed<<" seconds"<<endl;

  return 0;
}
