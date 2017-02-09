// Time-stamp: </w/parallel/w/openmp/class/taskfib.cc, Sun,  2 Feb 2014, 12:06:13 EST, http://wrfranklin.org/>

// Demo OpenMP tasks for fibonacci; from the OpenMP standard doc.

#include <omp.h>
#include <iostream>
#include <math.h>

#define count 10000

using namespace std;

int fib(int n) {
}

int main() {
  struct timespec start, finish;
  double elapsed1, elapsed2;
  double total1=0.0, total2=0.0;
  int i=0;
  for(int j=0;j<count;j++){
    clock_gettime(CLOCK_MONOTONIC, &start);
#pragma omp task 
    {
      i++;
    }
#pragma omp taskwait 
    clock_gettime(CLOCK_MONOTONIC, &finish);

    elapsed1 = (finish.tv_sec - start.tv_sec);
    elapsed1 += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

    clock_gettime(CLOCK_MONOTONIC, &start);
    i++;
    clock_gettime(CLOCK_MONOTONIC, &finish);

    elapsed2 = (finish.tv_sec - start.tv_sec);
    elapsed2 += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

    total1+=elapsed1;
    total2+=elapsed2;
  }
  cout<<"Approximate task overhead: "<<(total1-total2)/(double)count<<endl;
  return 0;
}
