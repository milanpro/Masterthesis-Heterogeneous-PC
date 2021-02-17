#include "../util/indep_util.hpp"
#include "../util/state.cuh"

namespace CPU
{
  TestResult executeLevel(int level, MMState *state, SplitTaskQueue *cpuQueue);
}