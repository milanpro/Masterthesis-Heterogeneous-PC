#include "../util/indepUtil.hpp"
#include <string>
#include <unordered_map>

void calcSkeleton(
    MMGPUState *state, int gpusUsed, int maxMem = -1,
    int startLevel = 0);

void printMMSepsets(MMGPUState *state);
