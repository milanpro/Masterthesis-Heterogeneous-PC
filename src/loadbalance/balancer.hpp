#include "../util/state.cuh"
#include "../executor/gpuExecutor.cuh"
#include "../executor/cpuExecutor.hpp"
#include <vector>
#include <memory>
#include <tuple>

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

  int64_t balance(int level);
  std::tuple<TestResult, TestResult> execute(int level);
  Balancer(std::vector<int> gpuList, MMState *state, Heterogeneity heterogeneity = Heterogeneity::All, bool verbose = false);
};
