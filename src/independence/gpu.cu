#include "../util/indepUtil.hpp"
#include "../util/State.cuh"
#include "rowIndTest.cuh"
#include <iostream>
#include <chrono>

__global__ void testRowL0(MMGPUState state, int row_node) {
  int col_node = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (col_node < state.p) {
    testRowL0TriangluarGPU(state, row_node, col_node);
  }
}

__global__ void testRowL1(MMGPUState state, int row_node) {
  int col_node = blockIdx.x;
  if (col_node < state.p) {
    testRowL1TriangluarGPU(state, row_node, col_node);
  }
}

TestResult gpuIndTest(int level, MMGPUState *state, SplitTaskQueue *gpuQueue, int maxEdgeCount) {
  auto start = std::chrono::system_clock::now();
  int numthreads = min((int)state->p, NUMTHREADS);
  dim3 block(numthreads), grid((state->p + numthreads - 1) / numthreads);
  if (level == 1) {
    numthreads = min((int)state->p, NUMTHREADS);
    grid = dim3(state->p);
  }

  int row_count = gpuQueue->size_approx();
  
  #pragma omp parallel for
  for(int i = 0; i < row_count; i++){
    SplitTask curTask;
    if(gpuQueue->try_dequeue(curTask)) {
      switch (level) {
        case 0:
          testRowL0<<<grid, block>>>(*state, curTask.row);
          break;
        case 1:
          testRowL0<<<grid, block>>>(*state, curTask.row);
          break;
      }
      getLastCudaError("L0 Kernel execution failed");
    }
  }
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::system_clock::now() - start)
                      .count();

  return {static_cast<uint64_t>(duration), (state->p * (state->p - 1L)) / 2};
}