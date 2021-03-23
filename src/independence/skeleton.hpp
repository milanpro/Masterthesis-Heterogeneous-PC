#include "../util/indep_util.hpp"
#include <string>
#include <unordered_map>

void calcSkeleton(
    MMState *state, std::vector<int> gpuList, bool verbose, std::string csvExportFile, int heterogeneity, bool showSepsets);

int printSepsets(MMState *state, bool verbose);
