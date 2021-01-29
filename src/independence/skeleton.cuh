#include "../util/indepUtil.h"
#include <string>
#include <unordered_map>

void calcSkeleton(
    MMGPUState *state, int gpusUsed, int maxMem = -1,
    std::unordered_map<std::string, uint64_t> *subSteps = nullptr,
    int startLevel = 0);

void printMMSepsets(MMGPUState *state);
