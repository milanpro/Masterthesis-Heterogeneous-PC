#include "state.cuh"
#include <algorithm>

MMState::MMState(uint64_t p, int observations, double alpha, int maxLevel)
    : p(p), observations(observations), alpha(alpha), maxLevel(maxLevel) {
  maxCondSize = std::max(maxLevel, 1);
  checkCudaErrors(cudaMallocManaged(&adj, (uint64_t)sizeof(int) * p * p));
  checkCudaErrors(cudaMallocManaged(&cor, (uint64_t)sizeof(double) * p * p));
  checkCudaErrors(cudaMallocManaged(&pMax, (uint64_t)sizeof(double) * p * p));
  checkCudaErrors(
  cudaMallocManaged(&sepSets, (uint64_t)sizeof(int) * p * p * maxCondSize));
  checkCudaErrors(
  cudaMallocManaged(&adj_compact, (uint64_t)sizeof(int) * p * p));
  checkCudaErrors(cudaMallocManaged(&max_adj, (uint64_t)sizeof(int)));
  checkCudaErrors(cudaMallocManaged(&lock, (uint64_t)sizeof(int) * p * p));
  std::fill_n(adj, p * p, 1);
  std::fill_n(adj_compact, p * p, 1);
  for (int i = 0; i < p; ++i) {
    adj[i * p + i] = 0;
    adj_compact[i * p + i] = 0;
  }
  std::fill_n(pMax, p * p, 0.0);
  std::fill_n(sepSets, p * p * maxCondSize, -1);

  memset(lock, 0, (uint64_t)sizeof(int) * p * p);
  max_adj[0] = p;
}

void MMState::destroy() {
  checkCudaErrors(cudaFree(adj));
  checkCudaErrors(cudaFree(cor));
  checkCudaErrors(cudaFree(pMax));
  checkCudaErrors(cudaFree(sepSets));
  checkCudaErrors(cudaFree(max_adj));
  checkCudaErrors(cudaFree(adj_compact));
  checkCudaErrors(cudaFree(lock));
}