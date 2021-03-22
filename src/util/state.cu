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
  max_adj[0] = (int)p;
}

void MMState::adviceReadonlyCor(int numberOfGPUs) {
  checkCudaErrors(cudaMemAdvise(cor, (uint64_t)sizeof(double) * p * p, cudaMemAdviseSetReadMostly, 0));
  for (int deviceId = 0; deviceId < numberOfGPUs; deviceId++) {
    checkCudaErrors(cudaMemPrefetchAsync(cor,
      (uint64_t)sizeof(double) * p * p, deviceId));
  }
}

void MMState::prefetchRows(int startRow, int rowCount, int deviceId) {
      checkCudaErrors(cudaMemPrefetchAsync(adj + startRow * p,
        (uint64_t)sizeof(int) * rowCount,
        deviceId, 0));
      checkCudaErrors(cudaMemPrefetchAsync(adj_compact + startRow * p,
        (uint64_t)sizeof(int) * rowCount,
        deviceId, 0));
      checkCudaErrors(cudaMemPrefetchAsync(pMax + startRow * p,
        (uint64_t)sizeof(double) * rowCount,
        deviceId, 0));
      checkCudaErrors(cudaMemPrefetchAsync(sepSets + startRow * p * maxCondSize,
        (uint64_t)sizeof(int) * rowCount * maxCondSize,
        deviceId, 0));
    }

void MMState::memAdvise(int numberOfGPUs) {
  checkCudaErrors(cudaMemAdvise(adj_compact,
    (uint64_t)sizeof(int) * p * p,
    cudaMemAdviseSetReadMostly, 0));

  for (int deviceId = 0; deviceId < numberOfGPUs; deviceId++) {
//     checkCudaErrors(cudaMemAdvise(adj + p * p / numberOfGPUs * deviceId,
//       (uint64_t)sizeof(int) * p * p / numberOfGPUs,
//       cudaMemAdviseSetPreferredLocation, deviceId));

// checkCudaErrors(cudaMemAdvise(pMax + p * p / numberOfGPUs * deviceId,
//       (uint64_t)sizeof(double) * p * p / numberOfGPUs,
//       cudaMemAdviseSetPreferredLocation, deviceId));

// checkCudaErrors(cudaMemAdvise(sepSets + p * p * maxCondSize / numberOfGPUs * deviceId,
//       (uint64_t)sizeof(int) * p * p * maxCondSize / numberOfGPUs,
//       cudaMemAdviseSetPreferredLocation, deviceId));

checkCudaErrors(cudaMemAdvise(lock + p * p / numberOfGPUs * deviceId,
      (uint64_t)sizeof(int) * p * p / numberOfGPUs,
      cudaMemAdviseSetPreferredLocation, deviceId));

// setting accessed by
// checkCudaErrors(cudaMemAdvise(adj,
//       (uint64_t)sizeof(int) * p * p,
//       cudaMemAdviseSetAccessedBy, deviceId));
// checkCudaErrors(cudaMemAdvise(adj_compact,
//       (uint64_t)sizeof(int) * p * p,
//       cudaMemAdviseSetAccessedBy, deviceId));
// checkCudaErrors(cudaMemAdvise(pMax,
//       (uint64_t)sizeof(double) * p * p,
//       cudaMemAdviseSetAccessedBy, deviceId));
// checkCudaErrors(cudaMemAdvise(sepSets,
//       (uint64_t)sizeof(int) * p * p * maxCondSize,
//       cudaMemAdviseSetAccessedBy, deviceId));
checkCudaErrors(cudaMemAdvise(lock,
      (uint64_t)sizeof(int) * p * p,
      cudaMemAdviseSetAccessedBy, deviceId));

  }
}

void MMState::destroy() {
  checkCudaErrors(cudaFree(adj));
  checkCudaErrors(cudaFree(cor));
  checkCudaErrors(cudaFree(pMax));
  checkCudaErrors(cudaFree(sepSets));
  checkCudaErrors(cudaFree(adj_compact));
  checkCudaErrors(cudaFree(lock));
}