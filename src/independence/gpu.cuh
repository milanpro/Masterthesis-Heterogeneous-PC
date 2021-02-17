#include "../util/indep_util.hpp"
#include "../util/state.cuh"

namespace GPU {
  TestResult executeLevel(int level, MMState *state, SplitTaskQueue *gpuQueue, int maxEdgeCount, int numberOfGPUs);
}
