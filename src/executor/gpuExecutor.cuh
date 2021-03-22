#include "executor.hpp"

struct GPUExecutor : Executor
{
  int maxEdgeCount;
  std::vector<int> gpuList;
  MMState *state;
  TestResult executeLevel(int level, bool verbose = false);

  GPUExecutor(MMState *state, int maxEdgeCount, std::vector<int> gpuList)
      : state(state), maxEdgeCount(maxEdgeCount), gpuList(gpuList)
  {
  }
};
