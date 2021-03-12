#include "../util/state.cuh"
#include "../executor/gpuExecutor.cuh"
#include "../executor/cpuExecutor.hpp"
#include <vector>
#include <memory>

struct Balancer
{
  bool verbose;
  int numberOfGPUs;
  MMState *state;
  std::vector<int> gpuToSMCountMap;
  int ompThreadCount;
  std::shared_ptr<CPUExecutor> cpuExecutor;
  std::shared_ptr<GPUExecutor> gpuExecutor;

  void balance(int level);
  unsigned long long execute(int level);
  Balancer(int numberOfGPUs, MMState *state, bool verbose = false);
};
