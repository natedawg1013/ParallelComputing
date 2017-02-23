//#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <iostream>
#include <fcntl.h>
#include <cmath>
using namespace std;

__device__ __managed__ float *x, *y, *z, *res;

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
  int width;
  int height;
  int stride; 
  float* elements;
} Matrix;

// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
      return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
                               float value)
{
      A.elements[row * A.stride + col] = value;
}

// Thread block size
#define BLOCK_SIZE 16

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
 __device__ Matrix GetSubMatrixA(Matrix A, int row, int col) 
{
      Matrix Asub;
      Asub.width    = 3;
      Asub.height   = BLOCK_SIZE;
      Asub.stride   = A.stride;
      Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                  + A.width * col];
      return Asub;
}
 __device__ Matrix GetSubMatrixB(Matrix A, int row, int col) 
{
      Matrix Asub;
      Asub.width    = BLOCK_SIZE;
      Asub.height   = 3;
      Asub.stride   = A.stride;
      Asub.elements = &A.elements[A.stride * A.height * row
                                  + BLOCK_SIZE * col];
      return Asub;
}
 __device__ Matrix GetSubMatrixC(Matrix A, int row, int col) 
{
      Matrix Asub;
      Asub.width    = BLOCK_SIZE;
      Asub.height   = BLOCK_SIZE;
      Asub.stride   = A.stride;
      Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                  + BLOCK_SIZE * col];
      return Asub;
}


// Forward declaration of the matrix multiplication kernel
//__global__ void MatMulKernel(*float, *float, *float, int, int, int, int, int, int);


__global__ void MatMulKernel(float* A, float* B, float* C, int ARows, int ACols, int BRows,
    int BCols, int CRows, int CCols)
{
    float CValue = 0;

    int Row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
    int Col = blockIdx.x*BLOCK_SIZE + threadIdx.x;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    for (int k = 0; k < (BLOCK_SIZE + ACols - 1)/BLOCK_SIZE; k++) {

         if (k*BLOCK_SIZE + threadIdx.x < ACols && Row < ARows)
             As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*BLOCK_SIZE + threadIdx.x];
         else
             As[threadIdx.y][threadIdx.x] = 0.0;

         if (k*BLOCK_SIZE + threadIdx.y < BRows && Col < BCols)
             Bs[threadIdx.y][threadIdx.x] = B[(k*BLOCK_SIZE + threadIdx.y)*BCols + Col];
         else
             Bs[threadIdx.y][threadIdx.x] = 0.0;

         __syncthreads();
         for (int n = 0; n < BLOCK_SIZE; ++n)
             CValue += (As[threadIdx.y][n] - Bs[n][threadIdx.x])*(As[threadIdx.y][n] - Bs[n][threadIdx.x]);

         __syncthreads();
    }

    if (Row < CRows && Col < CCols){ 
         if(CValue>0)
           CValue=1/sqrt(CValue);
         else
           CValue=0;
        C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols) +
           (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;
    }
}

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
    cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((B.width+dimBlock.x-1) / dimBlock.x, (A.height+dimBlock.y-1) / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A.elements, d_B.elements, d_C.elements, d_A.height, d_A.width, d_B.height, d_B.width, d_C.height, d_C.width);

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}


/*__global__ void calcGravity(const size_t n){
  int row = blockIdx.y*blockDim.x + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;

  __shared__ float A[blockDim.x][3];
  __shared__ float B[3][blockDim.x];

  if(threadIdx.y==0){
    A[threadIdx.x][0]=B[0][threadIdx.x]; 
    A[threadIdx.x][1]=B[1][threadIdx.x]; 
    A[threadIdx.x][2]=B[2][threadIdx.x];
  }
  if(row==0){
    res[col]==0.0f;
  }
  
  if(i<n){
    if(row!=col){
      float d = (A[col][0]-B[0][row]*A[col][0]-B[0][row]);
      d += (A[col][1]-B[1][row]*A[col][1]-B[1][row]);
      d += (A[col][2]-B[2][row]*A[col][2]-B[2][row]);
      atomicAdd(&res[col],=1/sqrt(d));
    }
  }
}*/
int main(int argc, char* argv[]){
  char* &filename = argv[1];
  vector<const char*> lineAddrs;
  struct stat st;
  stat(filename, &st);
  size_t filesize = st.st_size;
  int fd = open(filename,O_RDONLY,0);
  void* file = mmap(NULL,  filesize, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
  const char* input = (const char*) file;
  int lines=0;
  lineAddrs.push_back(input);
  for(int i=0;i<filesize;i++){
    if(input[i]=='\n'){
      lines++;
      lineAddrs.push_back(input+i+1);
    }
  }
  float *valuesA = new float[lines*3];
  float *valuesB = new float[lines*3];
  float *results = new float[lines*lines];
  cudaMallocManaged(&res, (int) (lines*sizeof(float)));
  for(int i=0;i<lines;i++){
    const char *a,*b,*c;
    a=lineAddrs[i];
    b=strpbrk(strpbrk(a," \t"),"-0123456789");
    c=strpbrk(strpbrk(b," \t"),"-0123456789");
    valuesA[i]         = valuesB[3*i]   = atof(a);
    valuesA[lines+i]   = valuesB[3*i+1] = atof(b);
    valuesA[2*lines+i] = valuesB[3*i+2] = atof(c);
    if(!(i%1000)) cout<<i<<endl;
  }
  for(int i=0;i<3*lines;i++){
    if(isnan(valuesA[i])) cout<<"NAN A "<<i<<endl;
    if(isnan(valuesB[i])) cout<<"NAN A "<<i<<endl;
  }
  munmap(file, filesize);
  Matrix A, B, C;
  A.width=3;
  A.height=lines;
  A.stride=lines;
  A.elements=valuesA;
  B.width=lines;
  B.height=3;
  B.stride=3;
  B.elements=valuesB;
  C.width=lines;
  C.height=lines;
  C.stride=lines;
  C.elements=results;
  MatMul(A,B,C);
  /*
  const dim3 block_size(16,16);
  int block_dim=(int) sqrt( (lines + block_size -1)/ 256);
  size_t dim3 grid_size(block_dim,block_dim:wq);
  cout<<"Sending to GPU"<<endl;
  // launch the kernel
  calcGravity<<<grid_size, block_size>>>(lines);
 */
  cudaDeviceSynchronize();
  cout<<"Summing"<<endl;
  double total=0.0f;
  for(int i=0;i<lines*lines;i++){
    total+=(double)results[i];
    if(i%100000000 == 0) cout<<total<<endl;
    if(isnan(results[i])) cout<<"NAN: "<<i<<endl;
    //if(i%(lines*1000) == 0) cout<<endl;
  }
  cout<<total<<endl;
  return 0;
}
