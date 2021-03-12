#include "balancer.hpp"
#include "../util/cuda_util.cuh"
#include <iostream>
#include <future>
#include <cmath>
#include <omp.h>

Balancer::Balancer(int numberOfGPUs, MMState *state, bool verbose) : numberOfGPUs(numberOfGPUs), state(state), verbose(verbose)
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

  if (level == 0)
  {
    gpuExecutor->enqueueSplitTask(SplitTask{0, variableCount});
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

    float max_test_iterations_gpu = std::ceil((float)max_row_test_count / (float)NUMTHREADS);

    for (int row = 0; row < variableCount; row++)
    {
      int row_length = state->adj_compact[row * variableCount + variableCount - 1];
      if (row_length < level)
      {
        if (balancedRows + 1 == row)
        {
          balancedRows = row;
        }
        continue;
      }

      size_t row_test_count = level > 1 ? binomialCoeff(row_length - 1, level) : row_length - 1;

      float test_iterations_gpu = std::ceil((float)row_test_count / (float)NUMTHREADS);
      float cpu_row_count = cpuExecutor->tasks.size();
      std::cout << "max_test_iterations_gpu: " << max_test_iterations_gpu << " test_iterations_gpu: " << test_iterations_gpu << std::endl;

      if (test_iterations_gpu == max_test_iterations_gpu || (variableCount - row) <= level * (ompThreadCount - cpu_row_count))
      {
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