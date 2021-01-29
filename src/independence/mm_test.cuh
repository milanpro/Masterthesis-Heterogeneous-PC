#ifndef R_PACKAGE_SRC_MANAGEDMEMORY_INDEPENDENCE_MM_TEST_CUH_
#define R_PACKAGE_SRC_MANAGEDMEMORY_INDEPENDENCE_MM_TEST_CUH_
#include "../util/indepUtil.h"

TestResult MMtestL0(MMGPUState *state, int blockSize = -1, int gpusUsed = 1);

TestResult MMtestL1(MMGPUState *state, int blockSize, int gpusUsed);

TestResult MMtestLN(MMGPUState *state, int blockSize, int gpusUsed, int lvl);
#endif // R_PACKAGE_SRC_MANAGEDMEMORY_INDEPENDENCE_MM_TEST_CUH_
