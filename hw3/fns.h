__global__ void calcGravity(const size_t n){
  unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i<n){
    res[i]=0.0f;
    for(int j=0;j<n;j++){
      if(j!=i){
        float d = (x[i]-x[j])*(x[i]-x[j]);
        d +=      (y[i]-y[j])*(y[i]-y[j]);
        d +=      (z[i]-z[j])*(z[i]-z[j]);
        res[i]+=1/sqrt(d);
        res2[i*lines+j]=1/sqrt(d);
      }
    }
  }
}

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