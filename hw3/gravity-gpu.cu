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

__device__ __managed__ float *x, *y, *z, *res, gpuTotal;

__global__ void calcGravity(const size_t n){
  unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i==0) gpuTotal=0;
  if(i<n){
    float result = 0;
    res[i]=0.0f;
    for(int j=0;j<n;j++){
      if(j!=i){
        float d = (x[i]-x[j])*(x[i]-x[j]);
        d +=      (y[i]-y[j])*(y[i]-y[j]);
        d +=      (z[i]-z[j])*(z[i]-z[j]);
        result+=1/sqrt(d);
      }
    }
    res[i]=result;
    atomicAdd(&gpuTotal, res[i]);
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
  for(int i=0;i<filesize;i++){
    if(input[i]=='\n'){
      lines++;
      lineAddrs.push_back(input+i+1);
    }
  }
  cudaMallocManaged(&x,   (size_t) lines*sizeof(float));
  cudaMallocManaged(&y,   (size_t) lines*sizeof(float));
  cudaMallocManaged(&z,   (size_t) lines*sizeof(float));
  cudaMallocManaged(&res, (size_t) lines*sizeof(float));
  for(int i=0;i<lines;i++){
    const char *a,*b,*c;
    a=lineAddrs[i];
    b=strpbrk(strpbrk(a," \t"),"-0123456789");
    c=strpbrk(strpbrk(b," \t"),"-0123456789");
    x[i]=atof(a);
    y[i]=atof(b);
    z[i]=atof(c);
    if(!(i%1000)) cout<<i<<endl;
  }
  munmap(file, filesize);
  
  const size_t block_size = 256;
  size_t grid_size = (lines + block_size -1)/ block_size;
  cout<<"Sending to GPU"<<endl;
  // launch the kernel
  calcGravity<<<grid_size, block_size>>>(lines);
 
  cudaDeviceSynchronize();
  cout<<"Summing"<<endl;
  double total=0.0f;
  for(int i=0;i<lines;i++){
    total+=res[i];
  }
  cout<<total<<endl;
  cout<<gpuTotal<<endl;
  return 0;
}
