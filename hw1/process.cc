#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace std;

typedef struct{
  int startHour;
  double length;
} entry;

int fillBuffer(char** buffer, int nlines, int linLen, FILE* file, bool skipFirst){
  size_t len = (size_t) linLen;
  int lineCount=0, ret=0;
  
  printf("Reading %d lines\n", nlines);
  
  if(skipFirst) ret=getline(&(buffer[0]), &len, file);
  
  for(int i=0;i<nlines;++i){
    len = (size_t) linLen;
    ret = getline(&(buffer[i]), &len, file);
    if(ret == -1) break;
    if(i % (nlines/100) == 0) fprintf(stderr, "|");
    lineCount++;
  }
  printf("Done\n");
  return lineCount;
}

int min(int a, int b){
  return (a<b ? a : b);
}

void processChunk(char** buffer, double& total, vector<entry> &out, int linPerThd, int lines){
  int threads = 1 + ((lines-1)/linPerThd);
  #pragma omp parallel for
  for(int j=0;j<threads;j++){
    double subtotal=0.0;
    int lineCount = min(lines-linPerThd*j, linPerThd);
    entry *subList = new entry[lineCount];
    for(int k=0;k<lineCount;k++){
      char *ptr = NULL, *current = NULL;
      current=strtok_r(buffer[j*linPerThd+k], ",", (&ptr));
      entry& toAdd = subList[k];
      for(int i=0;i<5;i++){
        if(i==1) sscanf(current, "%*d-%*d-%*d %d", &(toAdd.startHour));
        //if(i==2) sscanf(current, "%*d-%*d-%*d %d", &(toAdd.endHour));
        if(i==4) toAdd.length=atof(current);
        current=strtok_r(NULL, ",", &ptr);
      }
      subtotal+=toAdd.length;
    }
    #pragma omp atomic
    total+=subtotal;
    #pragma omp critical
    out.insert(out.end(), subList, subList+lineCount);
    delete[] subList;
  }
}

int main(int argc, char* argv[]){
  vector<entry> entries;
  int nThreads = omp_get_max_threads();
  int linPerThd = 1000000;
  int chunkSize = nThreads*linPerThd;
  FILE* data = fopen(argv[1],"r");
  char** lines = new char*[nThreads*linPerThd];
  fprintf(stderr, "Allocating memory\n");
  for(int i=0;i<chunkSize;++i) lines[i]=new char[181];
  
  double total = 0.0;
  int totalCount = 0;

  int lineCount = fillBuffer(lines, chunkSize, 180, data, true);
  processChunk(lines, total, entries, linPerThd, lineCount);
  totalCount+=lineCount;
  while(lineCount == chunkSize){
    lineCount = fillBuffer(lines, chunkSize, 180, data, false);
    processChunk(lines, total, entries, linPerThd, lineCount);
    totalCount+=lineCount;
  }

  printf("Average trip length: %f\n", total/(totalCount));
  printf("Freeing memory\n"); 
  for(int i=0;i<linPerThd*nThreads;++i) delete lines[i];
  delete[] lines;
  
  printf("Processing...\n");
  int finalCounts[24] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  int maxChunk = totalCount/nThreads+1;
  #pragma omp parallel for
  for(int i=0;i<nThreads;i++){
    int counts[24]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    int nentries = min(maxChunk, totalCount-(maxChunk*i));
    for(int j=i*maxChunk;j<i*maxChunk+nentries;j++){
      if(entries[j].length>(total/totalCount)){
        if(entries[j].startHour<0 || entries[j].startHour>23)
          fprintf(stderr, "Bad hour: %d\n", entries[j].startHour);
        else
          counts[entries[j].startHour]++;
      }
    }
    #pragma omp critical
    for(int j=0;j<24;j++){
      finalCounts[j]+=counts[j];
    }
  }

  int max = 0;
  for(int i=1;i<24;i++)
    if(finalCounts[i]>finalCounts[max]) max=i;
  printf("Hour with most rides longer than average: %0d:00\n", max);
  return 0;
}
