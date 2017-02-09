#include <cstdio>
#include <ctime>
#include <iostream> 
using namespace std;

int main(int argc, char* argv[]){
  volatile int a[100][100];
  volatile int b[100][100];
  volatile int c[100][100];

  for(int i=0;i<10;i++){
    for(int j=0;j<100;j++){
      a[i][j]=b[i][j]=1;
    }
  }
  
  clock_t begin = clock();
  for(int i=0;i<100;i++){
    for(int j=0;j<100;j++){
      c[i][j]=a[i][j]*b[i][j];
    }
  }
  clock_t end = clock();

  cout<<"Time elapsed: "<<(double(end-begin)/CLOCKS_PER_SEC)<<" seconds"<<endl;

  return 0;
}
