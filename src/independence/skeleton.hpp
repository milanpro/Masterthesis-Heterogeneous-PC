#include "../util/indep_util.hpp"
#include <string>
#include <unordered_map>

void calcSkeleton(
    MMState *state, std::vector<int> gpuList, bool verbose, int heterogeneity = 0, int maxMem = -1,
    int startLevel = 0);

void printSepsets(MMState *state);
