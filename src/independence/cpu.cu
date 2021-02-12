#include "../util/indepUtil.hpp"
#include "../util/State.cuh"
#include "rowIndTest.cuh"
#include <unordered_map>
#include <iostream>
#include <chrono>
#include "omp.h"

const int MULTITHREAD_THRESHHOLD = 100;

TestResult cpuIndTest(int level, MMGPUState *state, SplitTaskQueue *cpuQueue) {
  auto start = std::chrono::system_clock::now();

  auto row_count = cpuQueue->size_approx();
  #pragma omp parallel for if(row_count > MULTITHREAD_THRESHHOLD)
  for(int _j = 0; _j < row_count; _j++)
  {
    SplitTask curTask;
    if(cpuQueue->try_dequeue(curTask)) {
      auto row_node = curTask.row;
      for (int col_node = 0; col_node < row_node ; col_node++) {
        testRowTriangluar(level, *state, row_node, col_node);
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