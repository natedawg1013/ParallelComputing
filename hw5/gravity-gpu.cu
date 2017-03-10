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
__device__ __managed__ int **indices, *lens;
__device__ __managed__ float xStep, yStep, zStep;
__device__ __managed__ float counter;


__global__ void calcNearby(){
  int _x=10*blockIdx.x+threadIdx.x;
  int _y=10*blockIdx.y+threadIdx.y;
  int _z=10*blockIdx.z+threadIdx.z;
  int i = 10000*_x+100*_y+_z;
  float result = 0;
  float dx, dy, dz;
  for(int j=-1;j<=1;j++){
    for(int k=-1;k<=1;k++){
      for(int l=-1;l<=1;l++){
        if(_x+j<0 || _y+k<0 || _z+l<0) continue;
        if(_x+j>99 || _y+k>99 || _z+l>99) continue;
        int h=10000*(_x+j)+100*(_y+k)+(_z+l);
        for(int m=0;m<lens[i];m++){
          for(int n=0;n<lens[i];n++){
            //if(indices[h][m]==indices[i][n]) continue;
            atomicAdd(&counter,1);
            dx = x[indices[h][m]]-x[indices[i][n]];
            dx = y[indices[h][m]]-y[indices[i][n]];
            dx = z[indices[h][m]]-z[indices[i][n]];
            result+=rsqrt(dx*dx+dy*dy+dz*dz);
          }
        }
      }
    }
  }
  atomicAdd(&gpuTotal, result);
}



__global__ void calcOverall(){
  int _x=10*blockIdx.x+threadIdx.x;
  int _y=10*blockIdx.y+threadIdx.y;
  int _z=10*blockIdx.z+threadIdx.z;
  int i = 10000*_x+100*_y+_z;
  float result = 0;
  float dx, dy, dz;
  for(int j=0;j<100;j++){
    for(int k=0;k<100;k++){
      for(int l=0;l<100;l++){
        int h=10000*j+100*k+l;
        if(abs(_x-j)<2) continue;
        if(abs(_y-k)<2) continue;
        if(abs(_z-l)<2) continue;
        int weight=lens[i]*lens[h];
        dx = (_x-j)*xStep;
        dy = (_y-k)*yStep;
        dz = (_z-l)*zStep;
        result+=rsqrt(dx*dx+dy*dy+dz*dz)*weight;
      }
    }
  }
  atomicAdd(&gpuTotal, result);
}

int main(int argc, char* argv[]){
  counter=0;
  char* &filename = argv[1];
  vector<const char*> lineAddrs;
  struct stat st;
  int ndex=1;
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
  float maxX=x[0], minX=x[0], maxY=y[0], minY=y[0], maxZ=z[0], minZ=z[0];
  int chunkSize=(lines+ndex-1)/ndex;
  cout<<"Calculating grid size"<<endl;
  for(int i=0;i<ndex;i++){
    float _maxX=x[i*chunkSize], _minX=x[i*chunkSize], _maxY=y[i*chunkSize],
          _minY=y[i*chunkSize], _maxZ=z[i*chunkSize], _minZ=z[i*chunkSize];
    for(int j=i*chunkSize+1;j<(i+1)*chunkSize;j++){
      if(j>=lines) break;
      if(x[j]<_minX) _minX=x[j];
      if(x[j]>_maxX) _maxX=x[j];
      if(y[j]<_minY) _minY=y[j];
      if(y[j]>_maxY) _maxY=y[j];
      if(z[j]<_minZ) _minZ=z[j];
      if(z[j]>_maxZ) _maxZ=z[j];
    }
    {
      if(_minX<minX) minX=_minX;
      if(_maxX>maxX) maxX=_maxX;
      if(_minY<minY) minY=_minY;
      if(_maxY>maxY) maxY=_maxY;
      if(_minZ<minZ) minZ=_minZ;
      if(_maxZ>maxZ) maxZ=_maxZ;
    }
  }
  xStep=(maxX-minX)/99;
  yStep=(maxY-minY)/99;
  zStep=(maxZ-minZ)/99;
  typedef vector<int> bit;
  typedef vector<bit> skinny;
  typedef vector<skinny> flat;
  typedef vector<flat> pack;
  pack pointLists(100, flat(100, skinny(100)));
  cout<<"Assigning points"<<endl;
  for(int i=0;i<lines;i++){
    int _x=(int)((x[i]-minX)/xStep);
    int _y=(int)((y[i]-minY)/yStep);
    int _z=(int)((z[i]-minZ)/zStep);
    pointLists[_x][_y][_z].push_back(i);
  }
  cudaMallocManaged(&indices,   (size_t) 1000000*sizeof(int*));
  cudaMallocManaged(&lens,   (size_t) 1000000*sizeof(float));
  //cudaMallocManaged(&res,   (size_t) 1000000*sizeof(float*));
  for(int i=0;i<100;i++){
    for(int j=0;j<100;j++){
      for(int k=0;k<100;k++){
        int count=pointLists[i][j][k].size();
        //cout<<count;
        int index=10000*i+100*j+k;
        lens[index]=count;
        cudaMallocManaged(&(indices[index]),   (size_t) count*sizeof(int));
        cudaMemcpy(&(indices[index]), &(pointLists[i][j][k][0]),
                   count*sizeof(int), cudaMemcpyHostToDevice);
      }
    }
  }


  cout<<"Done"<<endl;
  float total=0.0f;
  dim3 dimBlock(10,10,10);
  dim3 dimGrid(10,10,10);
  cout<<"Sending to GPU"<<endl;
  // launch the kernel
  gpuTotal=0;
  calcOverall<<<dimGrid, dimBlock>>>();
  calcNearby<<<dimGrid, dimBlock>>>();
 
  cudaDeviceSynchronize();
  gpuTotal*=-1/2.0;
  cout<<gpuTotal<<endl;
  cout<<counter<<endl;
  return 0;
}
