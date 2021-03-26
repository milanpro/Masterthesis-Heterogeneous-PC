#include "balancer.hpp"
#include "../util/cuda_util.cuh"
#include <iostream>
#include <future>
#include <cmath>
#include <algorithm>
#include <vector>
#include <omp.h>
#include <chrono>

Balancer::Balancer(std::vector<int> gpuList, MMState *state, Heterogeneity heterogeneity, bool verbose) : gpuList(gpuList), state(state), verbose(verbose), heterogeneity(heterogeneity)
{
  int maxGPUCount = getDeviceCount();

  if (gpuList.size() > maxGPUCount)
  {
    std::cout << "Only " << maxGPUCount << " GPUs available for kernel execution" << std::endl;
    exit(-1);
  }

  ompThreadCount = omp_get_max_threads();

  int maxEdgeCount = (int)(state->p * (state->p - 1L) / 2);
  gpuExecutor = std::make_shared<GPUExecutor>(state, maxEdgeCount, gpuList);
  cpuExecutor = std::make_shared<CPUExecutor>(state);
}

int64_t Balancer::balance(int level)
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
  int rowsPerGPU = (int)std::ceil((float)variableCount / (float)gpuList.size());
  if (
      heterogeneity == Heterogeneity::CPUOnly)
  {
    // CPU only execution
    for (int row = 0; row < variableCount; row++)
    {
      cpuExecutor->enqueueSplitTask(SplitTask{row, 1});
    }
  }
  else if (level == 0 || heterogeneity == Heterogeneity::GPUOnly)
  {
    // GPU only execution
    gpuExecutor->enqueueSplitTask(SplitTask{0, variableCount});

    for (int i = 0; i < gpuList.size(); i++) {
      state->prefetchRows(i * rowsPerGPU, rowsPerGPU, gpuList[i]);
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

    int max_rows_on_cpu_multiplier = 0.25;

    if (level == 2) {
      max_rows_on_cpu_multiplier = 0.55;
    } else if (level == 3) {
      max_rows_on_cpu_multiplier = 0.05;
    }

    int max_rows_on_cpu = max_rows_on_cpu_multiplier * ompThreadCount;

    // Calculate maximum iterations the GPU could possibly need in this level
    size_t max_row_test_count = level > 1 ? binomialCoeff(variableCount - 2, level) : variableCount - 2;

    int max_test_iterations_gpu = std::ceil((float)max_row_test_count / (float)NUMTHREADS);

    // Calculate iterations thresholdwith which is balanced on the CPU
    cpuExecutor->calculateRowLengthMap(level);

    int max_iterations_idx = max_rows_on_cpu < variableCount ? max_rows_on_cpu : std::ceil(variableCount * max_rows_on_cpu_multiplier);

    auto row_length = std::get<1>(cpuExecutor->rowLengthMap[max_iterations_idx]);

    size_t row_test_count = level > 1 ? binomialCoeff(row_length - 1, level) : row_length - 1;
    int max_iterations_threshold = std::ceil((float)row_test_count / (float)NUMTHREADS);

    // Calculate iterations needed per row on the GPU
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

    if (verbose)
    {
      std::cout << "Maximum possible iterations: " << max_test_iterations_gpu << "\nMax GPU iteration threshold: " << max_iterations_threshold <<  std::endl;
    }

    // Start balancing
    for (int row = 0; row < variableCount; row++)
    {
      // Skip rows which are empty
      if (test_iterations_gpu[row] == 0)
      {
        if (balancedRows + 1 == row)
        {
          balancedRows = row;
        }
        continue;
      }

      // Rows already scheduled on CPU
      int cpu_row_count = cpuExecutor->tasks.size();

      if (test_iterations_gpu[row] >= max_iterations_threshold && cpu_row_count < max_rows_on_cpu)
      {
        if (balancedRows < row)
        {
          gpuExecutor->enqueueSplitTask(SplitTask{balancedRows, row - balancedRows});
          int deviceId = gpuList[balancedRows / rowsPerGPU];
          state->prefetchRows(balancedRows, row - balancedRows, deviceId);
        }
        // Balance row on CPU
        cpuExecutor->enqueueSplitTask(SplitTask{row, 1});
        balancedRows = row;
      }
    }
    if (balancedRows < variableCount)
    {
      // place left rows on GPU
      gpuExecutor->enqueueSplitTask(SplitTask{balancedRows, variableCount - balancedRows});
      int deviceId = gpuList[balancedRows / rowsPerGPU];
      state->prefetchRows(balancedRows, variableCount - balancedRows, deviceId);
    }
  }
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::system_clock::now() - start)
                      .count();
  if (verbose)
  {
    std::cout << "Balanced " << cpuExecutor->tasks.size() << " rows on the CPU and " << variableCount - cpuExecutor->tasks.size() << " rows on the GPU in " << duration << " \u03BCs." << std::endl;
  }
  return duration;
}

std::tuple<TestResult, TestResult> Balancer::execute(int level)
{
  auto verbose = this->verbose;
  if (verbose)
  {
    std::cout << "Start execution..." << std::endl;
  }
  auto cpuExecutor = this->cpuExecutor;
  auto gpuExecutor = this->gpuExecutor;
  auto maxRowLength = level == 0 ? state->p : std::get<1>(cpuExecutor->rowLengthMap[0]);
  auto resGPUFuture = std::async([gpuExecutor, level, maxRowLength, verbose] {
    return gpuExecutor->executeLevel(level, false, maxRowLength, verbose);
  });
  auto resCPUFuture = std::async([cpuExecutor, level, verbose] {
    return cpuExecutor->executeLevel(level, verbose);
  });

  TestResult resGPU = resGPUFuture.get();
  TestResult resCPU = resCPUFuture.get();

if (level != 0) {
  cpuExecutor->migrateEdges(level, verbose);
  int rowsPerGPU = (int)std::ceil((float)state->p / (float)gpuList.size());
  for (int i = 0; i < gpuList.size(); i++) {
    state->prefetchRows(i * rowsPerGPU, rowsPerGPU, gpuList[i]);
  }
}

  unsigned long long duration = std::max(resCPU.duration, resGPU.duration);
  if (verbose)
  {
    std::cout << "Order " << level << " finished with " << resCPU.tests + resGPU.tests << " tests in "
              << duration << " \u03BCs.\n"
              << std::endl;
  }
  return {resCPU, resGPU};
}


std::tuple<TestResult, TestResult> Balancer::executeWorkstealing(int level)
{
  auto verbose = this->verbose;
  if (verbose)
  {
    std::cout << "Start workstealing execution..." << std::endl;
  }

  auto cpuExecutor = this->cpuExecutor;
  auto gpuExecutor = this->gpuExecutor;
  cpuExecutor->calculateRowLengthMap(level);

  auto maxRowLength = std::get<1>(cpuExecutor->rowLengthMap[0]);

  auto resGPUFuture = std::async([gpuExecutor, level, maxRowLength, verbose] {
    return gpuExecutor->executeLevel(level, true, maxRowLength, verbose);
  });
  auto resCPUFuture = std::async([cpuExecutor, level, verbose] {
    return cpuExecutor->workstealingExecuteLevel(level, verbose);
  });

  TestResult resGPU = resGPUFuture.get();
  TestResult resCPU = resCPUFuture.get();

if (level != 0) {
  cpuExecutor->migrateEdges(level, verbose);
  int rowsPerGPU = (int)std::ceil((float)state->p / (float)gpuList.size());
  for (int i = 0; i < gpuList.size(); i++) {
    state->prefetchRows(i * rowsPerGPU, rowsPerGPU, gpuList[i]);
  }
}

  unsigned long long duration = std::max(resCPU.duration, resGPU.duration);
  if (verbose)
  {
    std::cout << "Order " << level << " finished with " << resCPU.tests + resGPU.tests << " tests in "
              << duration << " \u03BCs.\n"
              << std::endl;
  }
  return {resCPU, resGPU};
}
