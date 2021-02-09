#include "../util/indepUtil.hpp"
#include "../util/State.cuh"
#include "mm_indepTests.cuh"
#include <unordered_map>
#include <iostream>
#include <chrono>

TestResult gpuIndTestL0(MMGPUState *state, SplitTaskQueue *gpuQueue, int gpusUsed) {
  auto start = std::chrono::system_clock::now();
  int edgesPerGPU =
      (state->p * (state->p - 1L) / 2 + (gpusUsed - 1)) / gpusUsed;
  int numthreads = min(edgesPerGPU, NUMTHREADS);
  dim3 block(numthreads), grid((edgesPerGPU + numthreads - 1) / numthreads);

  #pragma omp parallel for
  for(int i = 0; i < gpusUsed; i++){
    cudaSetDevice(i);
    MMtestL0Triangle<<<grid, block>>>(*state, edgesPerGPU, i);
    cudaDeviceSynchronize();
    getLastCudaError("L0 Kernel execution failed");
  }
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::system_clock::now() - start)
                      .count();

  std::unordered_map<std::string, uint64_t> subTimes(
      {{"Copy", 0}, {"Test", 0}});
  return {static_cast<uint64_t>(duration), (state->p * (state->p - 1L)) / 2,
          subTimes};
}