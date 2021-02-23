#include "executor.hpp"

struct CPUExecutor : Executor
{
  MMState *state;
  TestResult executeLevel(int level);
  CPUExecutor(MMState *state) : state(state)
  {
  }
};
