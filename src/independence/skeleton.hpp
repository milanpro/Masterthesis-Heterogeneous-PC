#include "../util/indep_util.hpp"
#include "../loadbalance/balancer.hpp"
#include <string>
#include <unordered_map>

void calcSkeleton(
    MMState *state, std::vector<int> gpuList, bool verbose, bool workstealing, std::string csvExportFile, Balancer balancer, bool showSepsets);

int printSepsets(MMState *state, bool verbose);
