#pragma once
#include "./concurrentqueue.h"
#define NUMTHREADS 64

struct SplitTask
{
  int row;
  int rowCount;
};

struct TestResult
{
  uint64_t duration;
  uint64_t tests;
};

using SplitTaskQueue = moodycamel::ConcurrentQueue<SplitTask>;
