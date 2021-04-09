#include "state.cuh"
#include <algorithm>
#include <iostream>

MMState::MMState(uint64_t p, int observations, double alpha, int maxLevel, int mainDeviceId)
    : p(p), observations(observations), alpha(alpha), maxLevel(maxLevel)
{
  checkCudaErrors(cudaSetDevice(mainDeviceId));
  maxCondSize = std::max(maxLevel, 1);
  checkCudaErrors(cudaMallocManaged(&adj, (uint64_t)sizeof(int) * p * p));
  checkCudaErrors(cudaMallocManaged(&cor, (uint64_t)sizeof(double) * p * p));
  checkCudaErrors(cudaMallocManaged(&pMax, (uint64_t)sizeof(double) * p * p));
  checkCudaErrors(cudaMallocManaged(&node_status, (uint64_t)sizeof(bool) * p * p));
  checkCudaErrors(
      cudaMallocManaged(&sepSets, (uint64_t)sizeof(int) * p * p * maxCondSize));
  checkCudaErrors(
      cudaMallocManaged(&adj_compact, (uint64_t)sizeof(int) * p * p));
  checkCudaErrors(cudaMallocManaged(&max_adj, (uint64_t)sizeof(int)));
  checkCudaErrors(cudaMallocManaged(&lock, (uint64_t)sizeof(int) * p * p));
  std::fill_n(adj, p * p, 1);
  std::fill_n(adj_compact, p * p, 1);
  std::fill_n(node_status, p * p, false);
  for (int i = 0; i < p; ++i)
  {
    adj[i * p + i] = 0;
    adj_compact[i * p + i] = 0;
  }
  std::fill_n(pMax, p * p, 0.0);
  std::fill_n(sepSets, p * p * maxCondSize, -1);
  memset(lock, 0, (uint64_t)sizeof(int) * p * p);
  max_adj[0] = (int)p;
  gpu_done = false;
}

void MMState::adviceReadonlyCor(std::vector<int> gpuList)
{
  checkCudaErrors(cudaMemAdvise(cor, (uint64_t)sizeof(double) * p * p, cudaMemAdviseSetReadMostly, 0));
  for (auto deviceId : gpuList)
  {
    checkCudaErrors(cudaMemPrefetchAsync(cor,
                                         (uint64_t)sizeof(double) * p * p, deviceId));
  }
}

void MMState::prefetchRows(int startRow, int rowCount, int deviceId)
{
  checkCudaErrors(cudaMemPrefetchAsync(adj + startRow * p,
                                       (uint64_t)sizeof(int) * rowCount,
                                       deviceId));
  checkCudaErrors(cudaMemPrefetchAsync(adj_compact + startRow * p,
                                       (uint64_t)sizeof(int) * rowCount,
                                       deviceId));
  checkCudaErrors(cudaMemPrefetchAsync(pMax + startRow * p,
                                       (uint64_t)sizeof(double) * rowCount,
                                       deviceId));
  checkCudaErrors(cudaMemPrefetchAsync(sepSets + startRow * p * maxCondSize,
                                       (uint64_t)sizeof(int) * rowCount * maxCondSize,
                                       deviceId));
}

void MMState::memAdvise(std::vector<int> gpuList)
{
  checkCudaErrors(cudaMemAdvise(adj_compact,
                                (uint64_t)sizeof(int) * p * p,
                                cudaMemAdviseSetReadMostly, 0));

  int numberOfGPUs = gpuList.size();
  for (int i = 0; i < gpuList.size(); i++)
  {
    int deviceId = gpuList[i];

    checkCudaErrors(cudaMemAdvise(lock + ((p * p / numberOfGPUs) * i),
                                  (uint64_t)sizeof(int) * p * p / numberOfGPUs,
                                  cudaMemAdviseSetPreferredLocation, deviceId));

    checkCudaErrors(cudaMemAdvise(lock,
                                  (uint64_t)sizeof(int) * p * p,
                                  cudaMemAdviseSetAccessedBy, deviceId));
  }
}

void MMState::destroy()
{
  checkCudaErrors(cudaFree(adj));
  checkCudaErrors(cudaFree(cor));
  checkCudaErrors(cudaFree(pMax));
  checkCudaErrors(cudaFree(sepSets));
  checkCudaErrors(cudaFree(adj_compact));
  checkCudaErrors(cudaFree(lock));
  checkCudaErrors(cudaFree(node_status));
}