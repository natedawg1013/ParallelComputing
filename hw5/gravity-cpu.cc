#include <omp.h>
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
  int ndex = omp_get_max_threads();
  cout<<ndex<<endl;
  if(argc>=3) ndex=atoi(argv[2]);
  struct stat st;
  stat(filename, &st);
  size_t filesize = st.st_size;
  int fd = open(filename,O_RDONLY,0);
  void* file = mmap(NULL,  filesize, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
  const char* input = (const char*) file;
  int lines=0;
  cout<<"Loading points"<<endl;
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
    //if(!(i%1000)) cout<<i<<endl;
  }
  munmap(file, filesize);
  //double total=0.0f;
  float maxX=x[0], minX=x[0], maxY=y[0], minY=y[0], maxZ=z[0], minZ=z[0];
  int chunkSize=(lines+ndex-1)/ndex;
  cout<<"Calculating grid size"<<endl;
#pragma omp parallel for
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
#pragma omp critical
    {
      if(_minX<minX) minX=_minX;
      if(_maxX>maxX) maxX=_maxX;
      if(_minY<minY) minY=_minY;
      if(_maxY>maxY) maxY=_maxY;
      if(_minZ<minZ) minZ=_minZ;
      if(_maxZ>maxZ) maxZ=_maxZ;
    }
  }
  float xStep=(maxX-minX)/99;
  float yStep=(maxY-minY)/99;
  float zStep=(maxZ-minZ)/99;
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
  cout<<"Done"<<endl;
  float total=0.0f;
//#pragma omp parallel for schedule(dynamic,1) collapse(3) reduction(+:total)
  for(int i=0;i<100;i++){
    for(int j=0;j<100;j++){
      for(int k=0;k<100;k++){
        float subtotal=0.0f;
//#pragma omp parallel for schedule(dynamic,1) collapse(3) reduction(+:subtotal)
        for(int l=-1;l<=1;l++){
          for(int m=-1;m<=1;m++){
            for(int n=-1;n<=1;n++){
              int _x, _y, _z;
              _x=i+l;
              _y=j+m;
              _z=k+n;
              if(_x<0||_y<0||_z<0) continue;
              if(_x>99||_y>99||_z>99) continue;
              for(int p=0;p<pointLists[_x][_y][_z].size();p++){
                for(int o=0;o<pointLists[i][j][k].size();o++){
                  int me = pointLists[i][j][k][o];
                  int you=pointLists[_x][_y][_z][p];
                  if(me==you) continue;
                  float dx=x[me]-x[you];
                  float dy=y[me]-y[you];
                  float dz=z[me]-z[you];
                  subtotal+=1.0/sqrt(dx*dx+dy*dy+dz*dz);
                }
              }
            }
          }
        }
        total+=subtotal;
      }
    }
  }
#pragma omp parallel for schedule(dynamic,1) collapse(3) reduction(+:total)
  for(int i=0;i<100;i++){
    for(int j=0;j<100;j++){
      for(int k=0;k<100;k++){
        float subtotal=0.0f;
        if(pointLists[i][j][k].size()==0) continue;
#pragma omp parallel for schedule(dynamic,1) collapse(3) reduction(+:subtotal)
          for(int l=0;l<100;l++){
            for(int m=0;m<100;m++){
              for(int n=0;n<100;n++){
                if(abs(i-l)<2) continue;
                if(abs(j-m)<2) continue;
                if(abs(k-n)<2) continue;
                int weight=pointLists[l][m][n].size()*pointLists[i][j][k].size();
                if(weight==0) continue;
                float dx=(i-l)*xStep;
                float dy=(j-m)*yStep;
                float dz=(k-n)*zStep;
                subtotal+=((float)(weight))/sqrt(dx*dx+dy*dy+dz*dz);
              }
            }
          }
        //}
        total+=subtotal;
      }
    }
  }
  total/=2;
  cout<<total<<endl;

  return 0;
}
