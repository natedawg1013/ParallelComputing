// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
  int width;
  int height;
  int stride; 
  float* elements;
} Matrix;

#define BLOCK_SIZE 16

float *xc, *yc, *zc, *resc;

__device__ __managed__ float *x, *y, *z, *res, *res2;

float *valuesA;
float *valuesB;
float *results;

int lines;