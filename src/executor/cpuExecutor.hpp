#include "executor.hpp"

struct CPUExecutor : Executor
{
  MMState *state;
  TestResult executeLevel(int level, bool verbose = false);
  CPUExecutor(MMState *state) : state(state)
  {
  }
};
