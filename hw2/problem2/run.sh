#!/bin/bash

for i in 1 2 4 8 16 32 64
do
  export OMP_NUM_THREADS=$i
  ./incorrect
  ./atomic
  ./critical
  ./reduction
done
