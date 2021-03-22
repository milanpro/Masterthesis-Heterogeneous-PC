#include "balancer.hpp"
#include "../util/cuda_util.cuh"
#include <iostream>
#include <future>
#include <cmath>
#include <algorithm>
#include <vector>
#include <omp.h>
#include <chrono>

Balancer::Balancer(int numberOfGPUs, MMState *state, Heterogeneity heterogeneity, bool verbose) : numberOfGPUs(numberOfGPUs), state(state), verbose(verbose), heterogeneity(heterogeneity)
{
  int maxGPUCount = getDeviceCount();

  if (numberOfGPUs > maxGPUCount)
  {
    std::cout << "Only " << maxGPUCount << " GPUs available for kernel execution" << std::endl;
    exit(-1);
  }

  // for (int i = 0; i < numberOfGPUs; i++)
  // {
  //   gpuToSMCountMap.push_back(getDeviceSMCount(i));
  // }

  ompThreadCount = omp_get_max_threads();

  int maxEdgeCount = (int)(state->p * (state->p - 1L) / 2);
  gpuExecutor = std::make_shared<GPUExecutor>(state, maxEdgeCount, numberOfGPUs);
  cpuExecutor = std::make_shared<CPUExecutor>(state);
}

void Balancer::balance(int level)
{
  if (verbose)
  {
    std::cout << "Start balancing rows..." << std::endl;
  }
  auto start = std::chrono::system_clock::now();

  cpuExecutor->cleanupSplitTasks();
  gpuExecutor->cleanupSplitTasks();

  int variableCount = state->p;
  int balancedRows = 0;

  if (
      heterogeneity == Heterogeneity::CPUOnly)
  {
    for (int row = 0; row < variableCount; row++)
    {
      cpuExecutor->enqueueSplitTask(SplitTask{row, 1});
    }
  }
  else if (level == 0 || heterogeneity == Heterogeneity::GPUOnly)
  {
    gpuExecutor->enqueueSplitTask(SplitTask{0, variableCount});
    state->prefetchRows(0, variableCount, 0);
  }
  else
  {
    /**
     * GPU execution time should scale with row_length / num_threads in level 1
     * in higher levels more permutations are possible though the GPU exec time should scale with
     * (row_length! / (row_length - level + 2)!) / num_threads
     * or binomialCoeff(row_neighbours, level) / num_threads
     **/

    size_t max_row_test_count = level > 1 ? binomialCoeff(variableCount - 2, level) : variableCount - 2;

    int max_test_iterations_gpu = std::ceil((float)max_row_test_count / (float)NUMTHREADS);

    std::vector<int> test_iterations_gpu(variableCount, 0);

    for (int row = 0; row < variableCount; row++)
    {
      int row_length = state->adj_compact[row * variableCount + variableCount - 1];
      if (row_length >= level)
      {
        size_t row_test_count = level > 1 ? binomialCoeff(row_length - 1, level) : row_length - 1;
        test_iterations_gpu[row] = std::ceil((float)row_test_count / (float)NUMTHREADS);
      }
    }

    std::vector<int> sorted_test_iterations_gpu = test_iterations_gpu;

    std::sort(sorted_test_iterations_gpu.begin(), sorted_test_iterations_gpu.end());

    int max_iterations_threshold = ompThreadCount * level < variableCount ? sorted_test_iterations_gpu[variableCount - (ompThreadCount * level)] : sorted_test_iterations_gpu[std::ceil(variableCount * 0.75)];

    int min_iterations_threshold = ompThreadCount * 0.1 < variableCount ? sorted_test_iterations_gpu[std::floor(ompThreadCount * 0.1)] : sorted_test_iterations_gpu[std::floor(variableCount * 0.1)];

    if (verbose)
    {
      std::cout << "Maximum possible iterations: " << max_test_iterations_gpu << "\nMax GPU iteration threshold: " << max_iterations_threshold << "\nMin GPU iteration threshold: " << min_iterations_threshold << std::endl;
    }

    for (int row = 0; row < variableCount; row++)
    {
      if (test_iterations_gpu[row] == 0)
      {
        if (balancedRows + 1 == row)
        {
          balancedRows = row;
        }
        continue;
      }

      int cpu_row_count = cpuExecutor->tasks.size();

      if (test_iterations_gpu[row] >= max_iterations_threshold || test_iterations_gpu[row] <= min_iterations_threshold || (variableCount - row) <= level * (ompThreadCount - cpu_row_count))
      {
        if (balancedRows < row)
        {
          gpuExecutor->enqueueSplitTask(SplitTask{balancedRows, row - balancedRows});
          state->prefetchRows(balancedRows, row - balancedRows, 0);
        }
        cpuExecutor->enqueueSplitTask(SplitTask{row, 1});
        balancedRows = row;
      }
    }
    if (balancedRows < variableCount)
    {
      gpuExecutor->enqueueSplitTask(SplitTask{balancedRows, variableCount - balancedRows});
      state->prefetchRows(balancedRows, variableCount - balancedRows, 0);
    }
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::system_clock::now() - start)
                        .count();
    if (verbose)
    {
      std::cout << "Balanced " << cpuExecutor->tasks.size() << " rows on the CPU and " << variableCount - cpuExecutor->tasks.size() << " rows on the GPU in " << duration << " \u03BCs." << std::endl;
    }
  }
}

unsigned long long Balancer::execute(int level)
{
  auto verbose = this->verbose;
  if (verbose)
  {
    std::cout << "Start execution..." << std::endl;
  }
  auto cpuExecutor = this->cpuExecutor;
  auto gpuExecutor = this->gpuExecutor;
  auto resGPUFuture = std::async([gpuExecutor, level, verbose] {
    return gpuExecutor->executeLevel(level, verbose);
  });
  auto resCPUFuture = std::async([cpuExecutor, level, verbose] {
    return cpuExecutor->executeLevel(level, verbose);
  });

  TestResult resGPU = resGPUFuture.get();
  TestResult resCPU = resCPUFuture.get();

if (cpuExecutor->tasks.size() != 0) {
  cpuExecutor->migrateEdges(level, verbose);
  state->prefetchRows(0, state->p, 0);
}

  unsigned long long duration = std::max(resCPU.duration, resGPU.duration);
  if (verbose)
  {
    std::cout << "Order " << level << " finished with " << resCPU.tests + resGPU.tests << " tests in "
              << duration << " \u03BCs.\n"
              << std::endl;
  }
  return duration;
}