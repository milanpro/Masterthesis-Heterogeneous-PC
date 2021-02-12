#include "../util/indepUtil.hpp"
#include "../util/State.cuh"
#include "mm_indepTests.cuh"
#include <unordered_map>
#include <iostream>
#include <chrono>

__global__ void testRowL0(MMGPUState state, int row_node) {
    int col_node = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (col_node < state.p) {
        if (state.adj[state.p * row_node + col_node]) {
        double pVal = mm_calcPValue(state.cor[state.p * row_node + col_node],
          state.observations);
        state.pMax[state.p * col_node + row_node] = pVal;
        if (state.pMax[state.p * col_node + row_node] >= state.alpha) {
        state.adj[state.p * row_node + col_node] = 0;
        state.adj[state.p * col_node + row_node] = 0;
        state.adj_compact[state.p * row_node + col_node] = 0;
        state.adj_compact[state.p * col_node + row_node] = 0;
        state.sepSets[(col_node * state.maxCondSize * state.p) +
        (row_node * state.maxCondSize)] = -2;
      }
    }
  }
}

TestResult gpuIndTestL0(MMGPUState *state, SplitTaskQueue *gpuQueue) {
  auto start = std::chrono::system_clock::now();
  cudaSetDevice(0);
  int numthreads = min((int)state->p, NUMTHREADS);
  dim3 block(numthreads), grid(state->p / numthreads);

  int row_count = gpuQueue->size_approx();
  
  #pragma omp parallel for
  for(int i = 0; i < row_count; i++){
    SplitTask curTask;
    if(gpuQueue->try_dequeue(curTask)) {
      testRowL0<<<grid, block>>>(*state, curTask.row);
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