#include "../util/cudaUtil.cuh"
#include "../util/matrixPrint.cuh"
#include "../util/constants.h"
#include "skeleton.cuh"
#include "mm_test.cuh"
#include <iostream>
#include <string>
#include <unordered_set>

void calcSkeleton(MMGPUState *state, int gpusUsed, int maxMem,
                    std::unordered_map<std::string, uint64_t> *subSteps,
                    int startLevel) {

  int devID = setCudaDevice();

  if (VERBOSE)
    std::cout << "maxCondSize: " << state->maxCondSize
              << "  observations: " << state->observations
              << "  p: " << state->p << " gpusUsed: " << gpusUsed << std::endl;

  TestResult res;
  if (startLevel <= 0) {
    res = MMtestL0(state, maxMem, gpusUsed);
    if (VERBOSE) {
      std::cout << "Order 0 finished with " << res.tests << " tests in "
                << res.duration << " microseconds." << std::endl;
    }
  }

  if (state->maxCondSize < 1) {
    return;
  }
  if (startLevel <= 1) {
    res = MMtestL1(state, maxMem, gpusUsed);
  }
  cudaDeviceSynchronize();
  if (VERBOSE) {
    std::cout << "Order 1 finished with " << res.tests << " tests in "
              << res.duration << " microseconds." << std::endl;
  }
  if (!res.subSteps.empty() && subSteps != nullptr) {
    *subSteps = res.subSteps;
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
