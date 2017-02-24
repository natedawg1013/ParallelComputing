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

__device__ __managed__ float *x, *y, *z, gpuTotal;

__global__ void calcGravity(const size_t n){
  unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i==0) gpuTotal=0;
  if(i<n){
    float result = 0;
    float dx, dy, dz;
    for(int j=i+1;j<n;j++){
      dx = x[i]-x[j];
      dy = y[i]-y[j];
      dz = z[i]-z[j];
      result+=rsqrt(dx*dx+dy*dy+dz*dz);
    }
    atomicAdd(&gpuTotal, result);
  }
}

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
  cout<<"Reading file"<<endl;
  for(int i=0;i<filesize;i++){
    if(input[i]=='\n'){
      lines++;
      lineAddrs.push_back(input+i+1);
    }
  }
  cudaMallocManaged(&x,   (size_t) lines*sizeof(float));
  cudaMallocManaged(&y,   (size_t) lines*sizeof(float));
  cudaMallocManaged(&z,   (size_t) lines*sizeof(float));
  for(int i=0;i<lines;i++){
    const char *a,*b,*c;
    a=lineAddrs[i];
    b=strpbrk(strpbrk(a," \t"),"-0123456789");
    c=strpbrk(strpbrk(b," \t"),"-0123456789");
    x[i]=atof(a);
    y[i]=atof(b);
    z[i]=atof(c);
  }
  munmap(file, filesize);
  
  const size_t block_size = 256;
  size_t grid_size = (lines + block_size -1)/ block_size;
  cout<<"Sending to GPU"<<endl;
  // launch the kernel
  calcGravity<<<grid_size, block_size>>>(lines);
 
  cudaDeviceSynchronize();
  gpuTotal*=-1;
  cout<<gpuTotal<<endl;
  return 0;
}
