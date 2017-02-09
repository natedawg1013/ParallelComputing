// Time-stamp: </w/parallel/w/openmp/class/taskfib.cc, Sun,  2 Feb 2014, 12:06:13 EST, http://wrfranklin.org/>

// Demo OpenMP tasks for fibonacci; from the OpenMP standard doc.

#include <omp.h>
#include <iostream>
#include <math.h>

using namespace std;

#define PRINT(arg)  #arg "= " << (arg)     // Print an expression's name then its value, possibly 
// followed by a comma or endl.   
// Ex: cout << PRINTC(x) << PRINTN(y);
#define PRINTC(arg)  redtty << #arg << deftty << "= " << (arg) << ", "
#define PRINTN(arg)  redtty << #arg << deftty << "= " << (arg)  << endl
const string redtty("\033[1;31m");   // tell tty to switch to red
const string deftty("\033[0m");      // tell tty to switch back to default color

// Configuration settings
const int N=45;
const int MINPAR=28;

int ntasks=0;

int fib(int n) {
  int i, j;
  if (n<2)
    return n;
  else if (n<MINPAR) {
    return fib(n-1)+fib(n-2);
  }
  else {
#pragma omp task shared(i)
    {
    i=fib(n-1);
#pragma omp atomic
    ntasks++;
  }
#pragma omp task shared(j)
    {
      j=fib(n-2);
#pragma omp atomic
      ntasks++;
    }
#pragma omp taskwait
    return i+j;
  }
  }

    int main() {

    cout << PRINTC(omp_get_num_threads()) << PRINTC(omp_get_max_threads()) 
	 << PRINTN(omp_get_num_procs())<< PRINTN(omp_get_wtick());

    double start = omp_get_wtime();

#pragma omp parallel 
    {
#ifdef V1
#pragma omp critical
#endif
    {
    cout << "Starting parallel, " << PRINTN(omp_get_thread_num());
  }
#ifdef V2
#pragma omp barrier
#endif

#pragma omp single
    {
    cout << PRINTC(N) << PRINTN(fib(N));
  }
  }
    double elapsed = omp_get_wtime() - start;
    cout << PRINTC(ntasks) << PRINTN(elapsed);
  }

