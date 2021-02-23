#include "../util/indep_util.hpp"
#include "../util/state.cuh"
#include <vector>

namespace CPU
{
  TestResult executeLevel(int level, MMState *state, std::vector<SplitTask> &CPURows);
}