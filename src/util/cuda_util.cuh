#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <string>

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

// Iterations used in psueoinverse calculation
#ifndef ITERATIONS
#define ITERATIONS 30
#endif

#ifndef SIGN
#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
#endif

#define NUMTHREADS 64

static void checkForCudaError(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("CUDA error %s in file %s at line %d \n", cudaGetErrorString(error),
           file, line);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(error) (checkForCudaError(error, __FILE__, __LINE__))

// This will output the proper error string when calling cudaGetLastError
#define checkLastCudaError(msg) __checkLastCudaError(msg, __FILE__, __LINE__)

inline void __checkLastCudaError(const char *errorMessage, const char *file,
                                 const int line) {
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA Failure at %s(%i) : %s : (%d) %s.\n", file, line,
            errorMessage, static_cast<int>(err), cudaGetErrorString(err));
    exit(-1);
  }
}

extern __device__ double PYTHAG(double a, double b);

extern __host__ __device__ size_t binomialCoeff(int n, int k);

extern __host__ __device__ void ithCombination(int *sepset_nodes, size_t comb_number,
                                    int level, size_t neighbours);

template <int lvlSize>
__device__ void pseudoinverse(double Submat[][lvlSize],
                                double SubmatPInv[][lvlSize],
                                double v[][lvlSize], double *rv1, double *w,
                                double res[][lvlSize]);
