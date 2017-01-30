#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char* argv[]){
  int nThreads = omp_get_max_threads();
  int linPerThd = 100000;
  FILE* data = fopen(argv[1],"r");
  perror("");
  char* line = NULL;
  char** lines = new char*[linPerThd*nThreads];
  for(int i=0;i<linPerThd*nThreads;++i) lines[i]=NULL;
  size_t count=300;
  getline(&line, &count, data);
  printf("Reading %d lines\n", linPerThd*nThreads);
  for(int i=0;i<linPerThd*nThreads;++i){
    count=300;
    getline(&(lines[i]), &count, data);
    if(i%(linPerThd*nThreads/10)==0){
      printf("|");
      fflush(stdout);
    }
  }
  printf("Done\n");
  double total = 0.0;
  #pragma omp parallel for
  for(int j=0;j<nThreads;j++){
    double subtotal=0.0;
    for(int k=0;k<linPerThd;k++){
      char *ptr, *current;
      char *start, *end;
      double length;
      current=strtok_r(lines[j*linPerThd+k], ",", &ptr);
      for(int i=0;i<5;i++){
        //if(i==1) start=strdup(current);
        //if(i==2) end=strdup(current);
        if(i==4) length=atof(current);
        current=strtok_r(NULL, ",", &ptr);
      }
      subtotal+=length;
    }
    #pragma omp atomic
    total+=subtotal;
  }
  printf("Average trip length: %f\n", total/(linPerThd*nThreads));
    
  return 0;
}
