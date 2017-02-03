#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

const int LINES_PER_THREAD = 1000000;
const double DISTANCE_CUTOFF = 1000.0;

using namespace std;

typedef struct{
  int startHour;
  double length;
} entry;

int fillBuffer(char** buffer, int nlines, int linLen, FILE* file, bool skipFirst, bool &done){
  size_t len = (size_t) linLen;
  int lineCount=0, ret=0;
  
  fprintf(stderr, "Reading %d lines\n", nlines);
  
  if(skipFirst) ret=getline(&(buffer[0]), &len, file);

  done=false;
  for(int i=0;i<nlines;++i){
    len = (size_t) linLen;
    ret = getline(&(buffer[i]), &len, file);
    if(ret == -1){
      done=true;
      break;
    }
    if(i % (nlines/76) == 0) fprintf(stderr, "|");
    lineCount++;
  }
  fprintf(stderr, "Done\n");
  return lineCount;
}

int min(int a, int b){
  return (a<b ? a : b);
}

int processChunk(char** buffer, double& total, vector<entry> &out, int lines){
  int threads = omp_get_max_threads();
  int totalValid=0;
  int linPerThd=(lines+threads-1)/threads;
  #pragma omp parallel for
  for(int j=0;j<threads;j++){
    double subtotal=0.0;
    int lineCount = min(lines-linPerThd*j, linPerThd);
    entry *subList = new entry[lineCount];
    int valid=0;
    for(int k=0;k<lineCount;k++){
      char *ptr = NULL, *current = NULL;
      current=strtok_r(buffer[j*linPerThd+k], ",", (&ptr));
      entry& toAdd = subList[valid];
      for(int i=0;i<5;i++){
        if(i==1) sscanf(current, "%*d-%*d-%*d %d", &(toAdd.startHour));
        //if(i==2) sscanf(current, "%*d-%*d-%*d %d", &(toAdd.endHour));
        if(i==4) {
          toAdd.length=atof(current);
          if(toAdd.length<DISTANCE_CUTOFF) valid++;
        }
        current=strtok_r(NULL, ",", &ptr);
      }
      if(toAdd.length<1000)
        subtotal+=toAdd.length;
    }
    #pragma omp critical
    {
      total+=subtotal;
      totalValid+=valid;
      out.insert(out.end(), subList, subList+valid);
    }
    delete[] subList;
  }
  return totalValid;
}

int main(int argc, char* argv[]){
  vector<entry> entries;
  int nThreads = omp_get_max_threads();
  int linPerThd = LINES_PER_THREAD;
  int chunkSize = nThreads*linPerThd;
  FILE* data = fopen(argv[1],"r");
  char** lines = new char*[nThreads*linPerThd];
  fprintf(stderr, "Allocating memory\n");
  for(int i=0;i<chunkSize;++i) lines[i]=new char[181];
  
  double total = 0.0;
  int totalCount = 0;
  int validCount=0;
  bool done=false;
  int lineCount = fillBuffer(lines, chunkSize, 180, data, true, done);
  int valid = processChunk(lines, total, entries, lineCount);
  totalCount+=lineCount;
  validCount+=valid;
  while(!done){
    lineCount = fillBuffer(lines, chunkSize, 180, data, false, done);
    valid = processChunk(lines, total, entries, lineCount);
    totalCount+=lineCount;
    validCount+=valid;
  }

  printf("Average trip length: %f\n", total/(validCount));
  fprintf(stderr, "Freeing memory\n"); 
  for(int i=0;i<linPerThd*nThreads;++i) delete lines[i];
  delete[] lines;
  
  printf("Processing...\n");
  int finalCounts[24] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  int maxChunk = validCount/nThreads+1;
  #pragma omp parallel for
  for(int i=0;i<nThreads;i++){
    int counts[24]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    int nentries = min(maxChunk, validCount-(maxChunk*i));
    for(int j=i*maxChunk;j<i*maxChunk+nentries;j++){
      if(entries[j].length>(total/validCount)){
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
