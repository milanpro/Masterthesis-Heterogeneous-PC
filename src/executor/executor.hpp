#pragma once
#include "../util/indep_util.hpp"
#include "../util/state.cuh"
#include <vector>

struct Executor
{
  std::vector<SplitTask> tasks;

public:
  void enqueueSplitTask(SplitTask task)
  {
    tasks.push_back(task);
  }
};
