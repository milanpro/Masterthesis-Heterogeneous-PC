#include "../util/indepUtil.hpp"
#include "../util/State.cuh"
#include "mm_indepTests.cuh"
#include <unordered_map>
#include <iostream>
#include <chrono>
#include "omp.h"

const int MULTITHREAD_THRESHHOLD = 100;

TestResult cpuIndTestL0(MMGPUState *state, SplitTaskQueue *cpuQueue) {
  auto start = std::chrono::system_clock::now();

  auto p = state->p;
  auto row_count = cpuQueue->size_approx();
  #pragma omp parallel for if(row_count > MULTITHREAD_THRESHHOLD)
  for(int _j = 0; _j < row_count; _j++)
  {
    SplitTask curTask;
    if(cpuQueue->try_dequeue(curTask)) {
      auto row = curTask.row;
      //std::cout << "ThreadID: " << omp_get_thread_num() << std::endl;
      for (int i = 0; i < p ; i++) { // triangle of matrix and ignore diagonal for perf boost
        int adjIndex = row * p + i;
        if (state->adj[adjIndex]) {
          double pVal = mm_calcPValue(state->cor[adjIndex], state->observations);
          state->pMax[adjIndex] = pVal;
          if (state->pMax[adjIndex] >= state->alpha) {
            state->adj[adjIndex] = 0.f;
            state->sepSets[(row * state->maxCondSize) +
                      (i * state->maxCondSize)] = -2;
          }
        }
      }
    }
  }
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::system_clock::now() - start)
    .count();
  std::unordered_map<std::string, uint64_t> subTimes(
      {{"SubMat", 0}, {"Test", 0}, {"Copy", 0}, {"Merge", 0}});
  return {static_cast<uint64_t>(duration), 0, subTimes};
};