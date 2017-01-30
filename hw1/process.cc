#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char* argv[]){
  FILE* data = fopen(argv[1],"r");
  char* line = NULL;
  size_t count=299;
  getline(&line, &count, data);
  count=300;
  getline(&line, &count, data);
  char *ptr, *current;
  char *start, *end, *length;
  current=strtok_r(line, ",", &ptr);
  for(int i=0;i<5;i++){
    if(i==1) start=strdup(current);
    if(i==2) end=strdup(current);
    if(i==4) length=strdup(current);
    current=strtok_r(NULL, ",", &ptr);
  }
  printf("%s/%s/%s\n", start, end, length);
  return 0;
}
