#include "../util/constants.hpp"
#include "corHelper.cuh"
#include "corOwn.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define PERTHREAD 8

void gpuPMCC(const double *h_mat, uint64_t n, int dim, double *cors) {
  if (VERBOSE)
    printf("Cor started with N=%lu, dim=%i\n", n, dim);
  size_t dbytes = sizeof(double);
  double *d_mat, *d_means, *d_stddevs, *d_cors_copy;
  dim3 block(NUMTHREADS), grid(n, n), gridX(n);

  cudaMallocManaged(reinterpret_cast<void **>(&d_cors_copy), n * n * dbytes);
  cudaMallocManaged(reinterpret_cast<void **>(&d_means), n * dbytes);
  cudaMallocManaged(reinterpret_cast<void **>(&d_stddevs), n * dbytes);

  cudaMallocManaged(reinterpret_cast<void **>(&d_mat), n*dim*dbytes);
  memcpy(d_mat, h_mat, n*dim*dbytes);
  cudaDeviceSynchronize();

  gpuMeans<<<gridX, block>>>(d_mat, n, dim, d_means);
  cudaDeviceSynchronize();
  if (VERBOSE)
    printf("Means successful \n");

  gpuSD<<<gridX, block>>>(d_mat, n, dim, d_means, d_stddevs);
  cudaDeviceSynchronize();
  if (VERBOSE)
    printf("SD successful \n");

  gpuPMCC<<<grid, block>>>(d_mat, n, dim, d_means, d_stddevs, d_cors_copy);
  cudaDeviceSynchronize();
  memcpy(cors, d_cors_copy, n * n * dbytes);
  if (VERBOSE)
    printf("PMCC successful \n");

  // Free allocated space
  cudaFree(d_cors_copy);
  cudaFree(d_means);
  cudaFree(d_stddevs);
  cudaFree(d_mat);
}

void gpuPMCCShared(const double *h_mat, uint64_t n, int dim, double *cors) {
  size_t dbytes = sizeof(double);
  double *d_mat, *d_means, *d_stddevs, *d_cors_copy;
  size_t gridY = ((n % PERTHREAD == 0) ? n / PERTHREAD : (n / PERTHREAD) + 1);
  dim3 block(NUMTHREADS), grid(n, gridY), gridX(n);
  cudaMallocManaged(reinterpret_cast<void **>(&d_cors_copy), n * n * dbytes);
  cudaMallocManaged(reinterpret_cast<void **>(&d_means), n * dbytes);
  cudaMallocManaged(reinterpret_cast<void **>(&d_stddevs), n * dbytes);

  cudaMallocManaged(reinterpret_cast<void **>(&d_mat), n*dim*dbytes);
  memcpy(d_mat, h_mat, n*dim*dbytes);
  cudaDeviceSynchronize();

  gpuMeans<<<gridX, block>>>(d_mat, n, dim, d_means);
  cudaDeviceSynchronize();

  gpuSD<<<gridX, block>>>(d_mat, n, dim, d_means, d_stddevs);
  cudaDeviceSynchronize();

  gpuPMCCShared<<<grid, block>>>(d_mat, n, dim, d_means, d_stddevs, d_cors_copy);
  cudaDeviceSynchronize();
  memcpy(cors, d_cors_copy, n * n * dbytes);

  // Free allocated space
  cudaFree(d_cors_copy);
  cudaFree(d_means);
  cudaFree(d_stddevs);
  cudaFree(d_mat);
}
