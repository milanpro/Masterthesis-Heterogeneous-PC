#include "../util/indep_util.hpp"
#include <string>
#include <unordered_map>

void calcSkeleton(
    MMState *state, int gpusUsed, bool verbose, int maxMem = -1,
    int startLevel = 0);

void printSepsets(MMState *state);
