#include "../util/cuda_util.cuh"
#include "../util/matrix_print.cuh"
#include "../util/constants.hpp"
#include "cpu.cuh"
#include "gpu.cuh"
#include "skeleton.cuh"
#include "compact.cuh"
#include <iostream>
#include <string>
#include <unordered_set>
#include <memory>
#include <thread>
#include <future>

void calcLevel(GPUState *state, int maxMem, int maxEdgeCount, int level) {
  if (level >= 2) {
    std::cout << "Start Compacting" << std::endl;
    callCompact(state, 0, 1, state->p);
    std::cout << "Finished Compacting" << std::endl;
  }

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
  auto resCPUFuture = std::async(cpuIndTest, level, state, cpuQueue.get());
  auto resGPUFuture = std::async(gpuIndTest, level, state, gpuQueue.get(), maxEdgeCount);

  TestResult resCPU = resCPUFuture.get();
  TestResult resGPU = resGPUFuture.get();

  if (VERBOSE) {
    std::cout << "Order " << level << " finished with " << resCPU.tests + resGPU.tests << " tests in "
              << max(resCPU.duration,resGPU.duration) << " µs." << std::endl;
    std::cout << "\t CPU time: " << resCPU.duration  << " µs GPU time: "
              << resGPU.duration << " µs." << std::endl;
  }
}

void calcSkeleton(GPUState *state, int gpusUsed, int maxMem,
                    int startLevel) {
  int maxEdgeCount = state->p * (state->p - 1L) / 2;
  if (VERBOSE)
    std::cout << "maxCondSize: " << state->maxCondSize
              << "  observations: " << state->observations
              << "  p: " << state->p << " gpusUsed: " << gpusUsed << std::endl;

  for (int lvl = startLevel; lvl <= state->maxCondSize; lvl++) {
    calcLevel(state, maxMem, maxEdgeCount, lvl);
  }

  if (VERBOSE) {
    printSepsets(state);
  }
}

void printSepsets(GPUState *state) {
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
