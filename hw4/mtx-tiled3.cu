#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ctime>

using namespace std;

#define blockSize 32
#define DIM 1000 

__device__ __managed__ float A[DIM][DIM], B[DIM][DIM], C[DIM][DIM];

void init() {
  srand48(5L);
  for (int i=0; i<DIM; i++)
    for (int j=0; j<DIM; j++) {
      A[i][j] = drand48();
      B[i][j] = drand48();
    }
}

__global__ void MatMulKernel(int mtxDim, int grid){
    double CValue  = 0;
    int Row = blockIdx.y*blockSize + threadIdx.y;
    int Col = blockIdx.x*blockSize + threadIdx.x;
    const int &sRow = threadIdx.y;
    const int &sCol = threadIdx.x;
    const int zOff = blockIdx.z * blockSize;
    __shared__ float As[blockSize][blockSize];
    __shared__ float Bs[blockSize][blockSize];
    if(Row>=mtxDim || Col>=mtxDim) return;
    if(zOff+sCol < mtxDim)
      As[sRow][sCol]=A[Row][sCol+zOff];
    else{
      As[sRow][sCol]=0;
    }

    if(zOff+sRow < mtxDim)
      Bs[sRow][sCol]=B[sRow+zOff][Col];
    else{
      Bs[sRow][sCol]=0;
    }

    __syncthreads();
    for(int j=0;j<blockSize;j++){
      CValue+=As[sRow][j]*Bs[j][sCol];
    }
    atomicAdd(&(C[Row][Col]),CValue);
}

int main(int argc, char* argv[]){
  struct timespec start, finish;
  double elapsed;
  cout<<"generating random numbers"<<endl;
  init();
  cout<<"done"<<endl;
  int gridSize=(DIM+blockSize-1)/blockSize;
  dim3 dimBlock(blockSize,blockSize);
  dim3 dimGrid(gridSize, gridSize, gridSize);
  clock_gettime(CLOCK_MONOTONIC, &start);
  MatMulKernel<<<dimGrid, dimBlock>>>(DIM, gridSize);
  cudaDeviceSynchronize();
  clock_gettime(CLOCK_MONOTONIC, &finish);
  cout<<" C[3][3]: "<<C[3][3]<<" C[100][200]: "<<C[100][200]<<endl;
  elapsed = (finish.tv_sec - start.tv_sec);
  elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
  cout<<"Time elapsed: "<<elapsed<<" seconds"<<endl;
  cudaDeviceReset();
  return 0;
}
