#include "../util/indep_util.hpp"
#include "../util/state.cuh"
#include <vector>

namespace GPU {
  TestResult executeLevel(int level, MMState *state, std::vector<SplitTask> &GPURows, int maxEdgeCount, int numberOfGPUs);
}
