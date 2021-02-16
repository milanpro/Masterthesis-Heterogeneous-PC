#include "../util/indep_util.hpp"
#include <string>
#include <unordered_map>

void calcSkeleton(
    GPUState *state, int gpusUsed, int maxMem = -1,
    int startLevel = 0);

void printSepsets(GPUState *state);
