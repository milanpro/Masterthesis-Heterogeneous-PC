#pragma once
#include "../util/indep_util.hpp"
#include "../util/state.cuh"
#include <vector>

struct Executor
{
  /**
   * Task list for that executor. Only used in pre-balanced approach.
   */
  std::vector<SplitTask> tasks;

public:
  void enqueueSplitTask(SplitTask task)
  {
    tasks.push_back(task);
  }

  void cleanupSplitTasks() {
    tasks.clear();
  }
};
