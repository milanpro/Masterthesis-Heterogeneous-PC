#include "State.cuh"
#include <algorithm>

State::State(uint64_t p, int observations, double alpha, int maxCondSize)
    : p(p), observations(observations), alpha(alpha), maxCondSize(maxCondSize) {
  cor = reinterpret_cast<double *>(malloc(p * p * sizeof(double)));
  adj = reinterpret_cast<int *>(malloc(p * p * sizeof(int)));
  std::fill_n(adj, p * p, 1);
  for (int i = 0; i < p; ++i) {
    adj[i * p + i] = 0;
  }
  pMax = reinterpret_cast<double *>(malloc(p * p * sizeof(double)));
  std::fill_n(pMax, p * p, 0.0);
  sepSets = reinterpret_cast<int *>(malloc(maxCondSize * p * p * sizeof(int)));
  std::fill_n(sepSets, p * p * maxCondSize, -1);
  cudaHostRegister(pMax, (uint64_t)sizeof(double) * p * p, 0);
  cudaHostRegister(cor, (uint64_t)sizeof(double) * p * p, 0);
  cudaHostRegister(adj, (uint64_t)sizeof(int) * p * p, 0);
  cudaHostRegister(sepSets, (uint64_t)sizeof(int) * p * p * maxCondSize, 0);
}

State::State(const State &state)
    : p(state.p), observations(state.observations), alpha(state.alpha),
      maxCondSize(state.maxCondSize) {
  adj = new int[p * p];
  memcpy(adj, state.adj, p * p * sizeof(int));
  pMax = new double[p * p];
  memcpy(pMax, state.pMax, p * p * sizeof(double));
  cor = new double[p * p];
  memcpy(cor, state.cor, p * p * sizeof(double));
  sepSets = new int[p * p * maxCondSize];
  memcpy(sepSets, state.sepSets, p * p * maxCondSize * sizeof(int));
  cudaHostRegister(pMax, (uint64_t)sizeof(double) * p * p, 0);
  cudaHostRegister(cor, (uint64_t)sizeof(double) * p * p, 0);
  cudaHostRegister(adj, (uint64_t)sizeof(int) * p * p, 0);
  cudaHostRegister(sepSets, (uint64_t)sizeof(int) * p * p * maxCondSize, 0);
}

State::~State() {
  cudaHostUnregister(pMax);
  cudaHostUnregister(cor);
  cudaHostUnregister(adj);
  cudaHostUnregister(sepSets);
  delete[] pMax;
  delete[] sepSets;
  delete[] cor;
  delete[] adj;
}

GPUState::GPUState(uint64_t p, int observations, double alpha, int maxCondSize)
    : p(p), observations(observations), alpha(alpha), maxCondSize(maxCondSize) {
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&adj),
                             (uint64_t)sizeof(int) * p * p));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&cor),
                             (uint64_t)sizeof(double) * p * p));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&pMax),
                             (uint64_t)sizeof(double) * p * p));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&sepSets),
                             (uint64_t)sizeof(int) * p * p * maxCondSize));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&lock),
                             (uint64_t)sizeof(int) * p * p));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&rowMapping),
                             (uint64_t)sizeof(int) * p));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&colMapping),
                             (uint64_t)sizeof(int) * p));
  checkCudaErrors(cudaMemset(lock, 0, (uint64_t)sizeof(int) * p * p));
}

void GPUState::destroy() {
  checkCudaErrors(cudaFree(adj));
  checkCudaErrors(cudaFree(cor));
  checkCudaErrors(cudaFree(pMax));
  checkCudaErrors(cudaFree(sepSets));
  checkCudaErrors(cudaFree(lock));
  checkCudaErrors(cudaFree(rowMapping));
  checkCudaErrors(cudaFree(colMapping));
}

MMGPUState::MMGPUState(uint64_t p, int observations, double alpha,
                       int maxCondSize)
    : p(p), observations(observations), alpha(alpha), maxCondSize(maxCondSize) {
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

void MMGPUState::destroy() {
  checkCudaErrors(cudaFree(adj));
  checkCudaErrors(cudaFree(cor));
  checkCudaErrors(cudaFree(pMax));
  checkCudaErrors(cudaFree(sepSets));
  checkCudaErrors(cudaFree(max_adj));
  checkCudaErrors(cudaFree(adj_compact));
  checkCudaErrors(cudaFree(lock));
}
