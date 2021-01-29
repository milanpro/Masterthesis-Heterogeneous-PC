#pragma once
#include "./GPUData.cuh"
#include "./State.cuh"
#include "./concurrentqueue.h"
#include "./cudaUtil.cuh"
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#define NUMTHREADS 64

typedef std::unordered_map<int, std::unordered_set<int>> SepSets;

template <typename Iterator>
bool next_combination(const Iterator first, Iterator k, const Iterator last) {
  // Credits: Mark Nelson http://marknelson.us
  if ((first == last) || (first == k) || (last == k))
    return false;
  Iterator i1 = first;
  Iterator i2 = last;
  ++i1;
  if (last == i1)
    return false;
  i1 = last;
  --i1;
  i1 = k;
  --i2;
  while (first != i1) {
    if (*--i1 < *i2) {
      Iterator j = k;
      while (!(*i1 < *j))
        ++j;
      std::iter_swap(i1, j);
      ++i1;
      ++j;
      i2 = k;
      std::rotate(i1, j, last);
      while (last != j) {
        ++j;
        ++i2;
      }
      std::rotate(k, i2, last);
      return true;
    }
  }
  std::rotate(first, k, last);
  return false;
}

struct SplitTask {
  int blockRow;
  int blockCol;
  int blockSize;
};

struct IndepTask {
  int i;
  int j;
  int lvl;
  std::vector<int> sep;
  State *state;
};

struct TestResult {
  uint64_t duration;
  uint64_t tests;
  std::unordered_map<std::string, uint64_t> subSteps;
};

using TaskQueue = moodycamel::ConcurrentQueue<IndepTask>;
using SplitTaskQueue = moodycamel::ConcurrentQueue<SplitTask>;
