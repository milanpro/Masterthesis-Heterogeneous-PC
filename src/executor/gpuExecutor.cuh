#include "executor.hpp"

struct GPUExecutor : Executor
{
  int maxEdgeCount;
  int numberOfGPUs;
  MMState *state;
  TestResult executeLevel(int level, bool verbose = false);

  GPUExecutor(MMState *state, int maxEdgeCount, int numberOfGPUs)
      : state(state), maxEdgeCount(maxEdgeCount), numberOfGPUs(numberOfGPUs)
  {
  }
};
