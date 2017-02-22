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

float *x, *y, *z;



int main(int argc, char* argv[]){
  char* &filename = argv[1];
  vector<const char*> lineAddrs;
  int ndex = 0;//omp_get_max_threads();
  if(argc>=3) ndex=atoi(argv[2]);
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
  x=(float*) malloc((size_t) lines*sizeof(float));
  y=(float*) malloc((size_t) lines*sizeof(float));
  z=(float*) malloc((size_t) lines*sizeof(float));
  for(int i=0;i<lines;i++){
    const char *a,*b,*c;
    char temp[30];
    a=lineAddrs[i];
    b=strpbrk(strpbrk(a," \t"),"-0123456789");
    c=strpbrk(strpbrk(b," \t"),"-0123456789");
    x[i]=atof(a);
    y[i]=atof(b);
    z[i]=atof(c);
    if(!(i%1000)) cout<<i<<endl;
  }
  munmap(file, filesize);
  double total=0.0f;
#pragma omp parallel for reduction(+:total)
  for(int i=0;i<lines;i++){
    double subtotal=0.0f, dx, dy, dz;
    for(int j=0;j<lines;j++){
      if(i==j) continue;
      dx=x[i]-x[j];
      dx=y[i]-y[j];
      dx=z[i]-z[j];
      double d=sqrt(dx*dx+dy*dy+dz*dz);
      if(d==0.0f) continue;
      subtotal+=1/d;
    }
    total+=subtotal;
  }
  cout<<total<<endl;
  return 0;
}
