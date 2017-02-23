#include <iostream>
#include <ctime>
#include <cstdlib>

using namespace std;

#define MTX_DIM 100
#define BLOCK_SIZE 10

__device__ __managed__ float *A, *B, *C;

__global__ void calcGravity(const size_t n){
  int row = threadIdx.x + blockDim.x * blockIdx.x;
  int col = blockIdx.x*BLOCK_SIZE + threadIdx.x;
  if(row<n && col<n){
    float result = 0;
    for(int j=0;j<n;j++){
      result+=A[row*n+j]*B[j*n+col];
    }
    C[row*n+col]=result;
  }
}

int main(int argc, char* argv[]){
  cudaMallocManaged(&A,   (size_t) MTX_DIM*MTX_DIM*sizeof(float));
  cudaMallocManaged(&B,   (size_t) MTX_DIM*MTX_DIM*sizeof(float));
  cudaMallocManaged(&C,   (size_t) MTX_DIM*MTX_DIM*sizeof(float));
  
  struct timespec start, finish;
  double elapsed;

  for(int i=0;i<MTX_DIM;i++){
    for(int j=0;j<MTX_DIM;j++){
      A[i*MTX_DIM+j]=rand();
      B[i*MTX_DIM+j]=rand();
    }
  }
  
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((MTX_DIM+dimBlock.x-1) / dimBlock.x, (MTX_DIM+dimBlock.y-1) / dimBlock.y);
  cout<<"Sending to GPU"<<endl;
  // launch the kernel

  clock_gettime(CLOCK_MONOTONIC, &start);
  calcGravity<<<dimGrid, dimBlock>>>(MTX_DIM);
  cudaDeviceSynchronize();
  clock_gettime(CLOCK_MONOTONIC, &finish);

  elapsed = (finish.tv_sec - start.tv_sec);
  elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

  cout<<"Time elapsed: "<<elapsed<<" seconds"<<endl;

  return 0;
}
