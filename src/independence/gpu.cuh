#include "../util/indep_util.hpp"
#include "../util/state.cuh"

TestResult gpuIndTest(int level, GPUState *state, SplitTaskQueue *gpuQueue, int maxEdgeCount, int numberOfGPUs);