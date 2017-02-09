#include <cstdio>
#include <ctime>
#include <iostream> 
#include <omp.h>
#include <cstdlib>

#define count 1000000000

using namespace std;



int main(int argc, char* argv[]){
  float a[count];
  struct timespec start, finish;
  double elapsed;

  for(int i=0;i<count;i++){
    a[i]= (float) (rand()) / RAND_MAX;
  }

  float total=0.0; 
  clock_gettime(CLOCK_MONOTONIC, &start);
#pragma omp parallel for reduction(+:total)
  for(int i=0;i<count;i++){
    total+=a[i];
  }
  clock_gettime(CLOCK_MONOTONIC, &finish);

  elapsed = (finish.tv_sec - start.tv_sec);
  elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

  cout<<"Time elapsed: "<<elapsed<<" seconds"<<endl;
  cout<<"Result: "<<total<<endl;
  
  return 0;
}
