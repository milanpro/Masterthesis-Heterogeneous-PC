#pragma once
#include "../util/state.cuh"
#include "../executor/gpuExecutor.cuh"
#include "../executor/cpuExecutor.hpp"
#include <vector>
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
  float rows_multiplier;
  float rows_multiplier_l2;
  float rows_multiplier_l3;
  std::shared_ptr<CPUExecutor> cpuExecutor;
  std::shared_ptr<GPUExecutor> gpuExecutor;

  /**
   * Balance tasks onto the executors. (Only necessary for pre-balanced execution)
   */
  int64_t balance(int level);

  /**
   * Start executors for both CPU and GPU. Uses the pre-balanced approach.
   * Tasks have to be balanced beforehand
   */
  std::tuple<TestResult, TestResult> execute(int level, int numThreads);

  /**
   * Start both CPU and GPU executors. Uses the workstealing approach.
   */
  std::tuple<TestResult, TestResult> executeWorkstealing(int level, int numThreads);

  Balancer(){}
  Balancer(std::vector<int> gpuList, MMState *state, std::tuple<float, float, float> row_multipliers, Heterogeneity heterogeneity = Heterogeneity::All, bool verbose = false);
};
