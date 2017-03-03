#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <cuda_runtime.h>
#include "cublas_v2.h"

using namespace std;

#define blockSize 32
#define DIM 1000 

#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

__device__ __managed__ float A[DIM*DIM], B[DIM*DIM], C[DIM*DIM];

void init() {
  srand48(5L);
  for (int i=0; i<DIM; i++)
    for (int j=0; j<DIM; j++) {
      A[DIM*i+j] = drand48();
      B[DIM*i+j] = drand48();
    }
}

int main(int argc, char* argv[]){
  struct timespec start, finish;
  double elapsed;
  cout<<"generating random numbers"<<endl;
  init();
  cout<<"done"<<endl;
  
  cublasHandle_t handle;
  cublasCreate(&handle);
  const float a=1.0f,b=0.0f;
  
  clock_gettime(CLOCK_MONOTONIC, &start);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, DIM, DIM, DIM, &a, B, DIM, A, DIM, &b, C, DIM);
  clock_gettime(CLOCK_MONOTONIC, &finish);
  cublasDestroy(handle);
  
  cout<<" C[3][3]: "<<C[3*DIM+3]<<" C[100][200]: "<<C[100*DIM+200]<<endl;
  elapsed = (finish.tv_sec - start.tv_sec);
  elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
  cout<<"Time elapsed: "<<elapsed<<" seconds"<<endl;
  cudaDeviceReset();
  return 0;
}
