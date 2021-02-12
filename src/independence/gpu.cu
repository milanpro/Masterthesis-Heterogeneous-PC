#include "../util/indepUtil.hpp"
#include "../util/State.cuh"
#include "rowIndTest.cuh"
#include <unordered_map>
#include <iostream>
#include <chrono>

__global__ void testRow(int level, MMGPUState state, int row_node) {
  int col_node = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (col_node < state.p) {
    testRowTriangluar(level, state, row_node, col_node);
  }
}

TestResult gpuIndTest(int level, MMGPUState *state, SplitTaskQueue *gpuQueue) {
  auto start = std::chrono::system_clock::now();
  cudaSetDevice(0);
  int numthreads = min((int)state->p, NUMTHREADS);
  dim3 block(numthreads), grid(state->p / numthreads);

  int row_count = gpuQueue->size_approx();
  
  #pragma omp parallel for
  for(int i = 0; i < row_count; i++){
    SplitTask curTask;
    if(gpuQueue->try_dequeue(curTask)) {
      testRow<<<grid, block>>>(level, *state, curTask.row);
      getLastCudaError("L0 Kernel execution failed");
    }
  }
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::system_clock::now() - start)
                      .count();

  std::unordered_map<std::string, uint64_t> subTimes(
      {{"Copy", 0}, {"Test", 0}});
  return {static_cast<uint64_t>(duration), (state->p * (state->p - 1L)) / 2,
          subTimes};
}