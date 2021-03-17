#include "balancer.hpp"
#include "../util/cuda_util.cuh"
#include <iostream>
#include <future>
#include <cmath>
#include <algorithm>
#include <vector>
#include <omp.h>

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
  cpuExecutor->cleanupSplitTasks();
  gpuExecutor->cleanupSplitTasks();

  int variableCount = state->p;
  int balancedRows = 0;

  if (level == 0 || heterogeneity == Heterogeneity::GPUOnly)
  {
    gpuExecutor->enqueueSplitTask(SplitTask{0, variableCount});
  }
  else if (
      heterogeneity == Heterogeneity::CPUOnly)
  {
    for (int row = 0; row < variableCount; row++)
    {
      cpuExecutor->enqueueSplitTask(SplitTask{row, 1});
    }
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

    int iterations_threshhold = ompThreadCount * level < variableCount ? sorted_test_iterations_gpu[variableCount - (ompThreadCount * level)] : sorted_test_iterations_gpu[std::ceil(variableCount * 0.75)];

    //std::cout << "max_test_iterations_gpu: " << max_test_iterations_gpu << " iteration threshhold: " << iterations_threshhold << std::endl;

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

      if (test_iterations_gpu[row] <= 10)
      {
        continue;
      }

      int cpu_row_count = cpuExecutor->tasks.size();

      if (test_iterations_gpu[row] >= iterations_threshhold || (variableCount - row) <= level * (ompThreadCount - cpu_row_count))
      {
        //std::cout << "Balanced on CPU with: " << test_iterations_gpu[row] << " iterations." << std::endl;

        if (balancedRows < row)
        {
          gpuExecutor->enqueueSplitTask(SplitTask{balancedRows, row - balancedRows});
        }
        cpuExecutor->enqueueSplitTask(SplitTask{row, 1});
        balancedRows = row;
      }
    }
    if (balancedRows < variableCount)
    {
      gpuExecutor->enqueueSplitTask(SplitTask{balancedRows, variableCount - balancedRows});
    }
  }
}

unsigned long long Balancer::execute(int level)
{
  auto verbose = this->verbose;
  auto cpuExecutor = this->cpuExecutor;
  auto gpuExecutor = this->gpuExecutor;
  auto resCPUFuture = std::async([cpuExecutor, level, verbose] {
    return cpuExecutor->executeLevel(level, verbose);
  });
  auto resGPUFuture = std::async([gpuExecutor, level, verbose] {
    return gpuExecutor->executeLevel(level, verbose);
  });

  TestResult resCPU = resCPUFuture.get();
  TestResult resGPU = resGPUFuture.get();

  unsigned long long duration = std::max(resCPU.duration, resGPU.duration);
  if (verbose)
  {
    std::cout << "Order " << level << " finished with " << resCPU.tests + resGPU.tests << " tests in "
              << duration << " \u03BCs." << std::endl;
  }
  return duration;
}