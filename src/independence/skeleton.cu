#include "../util/cudaUtil.cuh"
#include "../util/matrixPrint.cuh"
#include "../util/constants.hpp"
#include "cpu.cuh"
#include "gpu.cuh"
#include "skeleton.cuh"
#include "mm_test.cuh"
#include <iostream>
#include <string>
#include <unordered_set>
#include <memory>

void calcSkeleton(MMGPUState *state, int gpusUsed, int maxMem,
                    int startLevel) {

  int devID = setCudaDevice();
  int maxEdgeCount = state->p * (state->p - 1L) / 2;
  if (VERBOSE)
    std::cout << "maxCondSize: " << state->maxCondSize
              << "  observations: " << state->observations
              << "  p: " << state->p << " gpusUsed: " << gpusUsed << std::endl;

  TestResult res, resCPU, resGPU;
  if (startLevel <= 0) {
    auto cpuQueue = std::unique_ptr<SplitTaskQueue>(new SplitTaskQueue());
    auto gpuQueue = std::unique_ptr<SplitTaskQueue>(new SplitTaskQueue());
    for (int row = 0; row < state->p; row++)
    {
      if (row % 2 == 0) {
        cpuQueue->enqueue(SplitTask{row});
      } else {
        gpuQueue->enqueue(SplitTask{row});
      }
    }
    
    resCPU = cpuIndTest(0, state, cpuQueue.get());
    resGPU = gpuIndTest(0, state, gpuQueue.get(), maxEdgeCount);
    if (VERBOSE) {
      std::cout << "Order 0 finished with " << resCPU.tests + resGPU.tests << " tests in "
                << max(resCPU.duration,resGPU.duration) << " µs." << std::endl;
      std::cout << "\t CPU time: " << resCPU.duration  << " µs GPU time: "
                << resGPU.duration << " µs." << std::endl;
    }
  }

  if (state->maxCondSize < 1) {
    return;
  }
  if (startLevel <= 1) {
    auto cpuQueue = std::unique_ptr<SplitTaskQueue>(new SplitTaskQueue());
    auto gpuQueue = std::unique_ptr<SplitTaskQueue>(new SplitTaskQueue());
    for (int row = 0; row < state->p; row++)
    {
      if (row % 2 == 0) {
        cpuQueue->enqueue(SplitTask{row});
      } else {
        gpuQueue->enqueue(SplitTask{row});
      }
    }
    
    resCPU = cpuIndTest(1, state, cpuQueue.get());
    resGPU = gpuIndTest(1, state, gpuQueue.get(), maxEdgeCount);
    if (VERBOSE) {
      std::cout << "Order 1 finished with " << resCPU.tests + resGPU.tests << " tests in "
                << max(resCPU.duration,resGPU.duration) << " µs." << std::endl;
      std::cout << "\t CPU time: " << resCPU.duration  << " µs GPU time: "
                << resGPU.duration << " µs." << std::endl;
    }
  }


  int lvl = (startLevel < 2) ? 2 : startLevel;
  while(lvl <= state->maxCondSize){
    res = MMtestLN(state, maxMem, gpusUsed, lvl);
    if (VERBOSE) {
      std::cout << "Order " << lvl << " finished with " << res.tests
                << " tests in " << res.duration << " microseconds."
                << std::endl;
    }
    ++lvl;
  }

  if (VERBOSE) {
    printMMSepsets(state);
  }
}

void printMMSepsets(MMGPUState *state) {
  int nrEdges = 0;
  for (int i = 0; i < state->p; i++) {
    for (int j = i + 1; j < state->p; j++) {
      if (!state->adj[i * state->p + j]) {
        std::string sepset_string = "";
        for (int k = 0; k < state->maxCondSize; k++) {
          int current_sepset_node =
              state->sepSets[(i * state->maxCondSize * state->p) +
                             (j * state->maxCondSize) + k];
          if (current_sepset_node == -2) {
            std::cout << "Separation from " << i << " to " << j << std::endl;
            break;
          } else if (current_sepset_node == -1) {
            break;
          } else {
            sepset_string.append(std::to_string(current_sepset_node));
            sepset_string.append(" ");
          }
        }
        if (sepset_string != "") {
          std::cout << "Separation from " << i << " to " << j << " via "
                    << sepset_string << std::endl;
        }
      } else {
        nrEdges++;
      }
    }
  }
  std::cout << "Total number of edges: " << nrEdges << std::endl;
}
