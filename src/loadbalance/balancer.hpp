#include "../util/state.cuh"
#include "../executor/gpuExecutor.cuh"
#include "../executor/cpuExecutor.hpp"
#include <vector>
#include <memory>

enum Heterogeneity {
  All = 0,
  GPUOnly = 1,
  CPUOnly = 2
};

struct Balancer
{
  bool verbose;
  Heterogeneity heterogeneity;
  std::vector<int> gpuList;
  MMState *state;
  std::vector<int> gpuToSMCountMap;
  int ompThreadCount;
  std::shared_ptr<CPUExecutor> cpuExecutor;
  std::shared_ptr<GPUExecutor> gpuExecutor;

  void balance(int level);
  unsigned long long execute(int level);
  Balancer(std::vector<int> gpuList, MMState *state, Heterogeneity heterogeneity = Heterogeneity::All, bool verbose = false);
};
