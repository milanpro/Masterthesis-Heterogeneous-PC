#include "balancer.hpp"
#include "../util/cuda_util.cuh"
#include "../util/assertions.hpp"
#include <iostream>
#include <future>
#include <cmath>
#include <omp.h>
#include <chrono>

Balancer::Balancer(std::vector<int> gpuList, MMState *state, std::tuple<float, float, float> row_multipliers, Heterogeneity heterogeneity, bool verbose) : gpuList(gpuList), state(state), verbose(verbose), heterogeneity(heterogeneity)
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

  rows_multiplier = std::get<0>(row_multipliers);
  rows_multiplier_l2 = std::get<1>(row_multipliers);
  rows_multiplier_l3 = std::get<2>(row_multipliers);
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
  int balancedRows = -1;
  int balancedOnGPU = 0;
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
    balancedOnGPU = variableCount;
    for (int i = 0; i < gpuList.size(); i++)
    {
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

    float max_rows_on_cpu_multiplier = rows_multiplier;

    if (level == 2)
    {
      max_rows_on_cpu_multiplier = rows_multiplier_l2;
    }
    else if (level == 3)
    {
      max_rows_on_cpu_multiplier = rows_multiplier_l3;
    }

    int max_rows_on_cpu = (float)max_rows_on_cpu_multiplier * ompThreadCount;

    // Calculate row length threshold used for balancing
    cpuExecutor->calculateRowLengthMap(level);
    int max_iterations_idx = max_rows_on_cpu < variableCount ? max_rows_on_cpu : std::ceil(variableCount * max_rows_on_cpu_multiplier);
    auto max_row_length = std::get<1>(cpuExecutor->rowLengthMap[max_iterations_idx]);

    // Start balancing
    for (int row = 0; row < variableCount; row++)
    {
      // Skip rows which are too short
      if (state->adj_compact[row * variableCount + variableCount - 1] < level)
      {
        if (balancedRows + 1 == row)
        {
          balancedRows = row;
        }
        continue;
      }

      if (state->adj_compact[row * variableCount + variableCount - 1] > max_row_length)
      {
        if (balancedRows < (row - 1))
        {
          int missing_rows = row - balancedRows - 1;
          gpuExecutor->enqueueSplitTask(SplitTask{balancedRows + 1, missing_rows});
          balancedOnGPU += missing_rows;
          int deviceId = gpuList[balancedRows / rowsPerGPU];
        }
        // Balance row on CPU
        cpuExecutor->enqueueSplitTask(SplitTask{row, 1});
        balancedRows = row;
      }
    }
    if (balancedRows < variableCount - 1)
    {
      // place left rows on GPU
      int missing_rows = variableCount - balancedRows - 1;
      gpuExecutor->enqueueSplitTask(SplitTask{balancedRows + 1, missing_rows});
      balancedOnGPU += missing_rows;
      int deviceId = gpuList[balancedRows / rowsPerGPU];
    }
  }
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::system_clock::now() - start)
                      .count();
  if (verbose)
  {
    std::cout << "Balanced " << cpuExecutor->tasks.size() << " rows on the CPU and " << balancedOnGPU << " rows on the GPU in " << duration << " ms." << std::endl;
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

  auto resGPUFuture = std::async([gpuExecutor, level, verbose] {
    return gpuExecutor->executeLevel(level, false, verbose);
  });
  auto resCPUFuture = std::async([cpuExecutor, level, verbose] {
    return cpuExecutor->executeLevel(level, verbose);
  });

  TestResult resGPU = resGPUFuture.get();
  TestResult resCPU = resCPUFuture.get();

  if (level != 0)
  {
#if MIGRATE_EDGES
    cpuExecutor->migrateEdges(level, verbose);
#endif
    int rowsPerGPU = (int)std::ceil((float)state->p / (float)gpuList.size());
    for (int i = 0; i < gpuList.size(); i++)
    {
      state->prefetchRows(i * rowsPerGPU, rowsPerGPU, gpuList[i]);
    }
  }
#if MIGRATE_EDGES
  else if (heterogeneity == Heterogeneity::CPUOnly)
  {
    cpuExecutor->migrateEdges(level, verbose);
  }
#endif

  unsigned long long duration = std::max(resCPU.duration, resGPU.duration);
  if (verbose)
  {
    std::cout << "Order " << level << " finished with " << resCPU.tests + resGPU.tests << " tests in "
              << duration << " ms.\n"
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

#ifndef NDEBUG
  assertNodeStatus(state, level);
#endif

  std::fill_n(state->node_status, state->p * state->p, false);

  auto cpuExecutor = this->cpuExecutor;
  auto gpuExecutor = this->gpuExecutor;
  cpuExecutor->calculateRowLengthMap(level);

  auto resGPUFuture = std::async([gpuExecutor, level, verbose] {
    return gpuExecutor->executeLevel(level, true, verbose);
  });
  auto resCPUFuture = std::async([cpuExecutor, level, verbose] {
    return cpuExecutor->workstealingExecuteLevel(level, verbose);
  });

  TestResult resGPU = resGPUFuture.get();
  TestResult resCPU = resCPUFuture.get();

  if (level != 0)
  {
#if MIGRATE_EDGES
    cpuExecutor->migrateEdges(level, verbose);
#endif
    int rowsPerGPU = (int)std::ceil((float)state->p / (float)gpuList.size());
    for (int i = 0; i < gpuList.size(); i++)
    {
      state->prefetchRows(i * rowsPerGPU, rowsPerGPU, gpuList[i]);
    }
  }

  unsigned long long duration = std::max(resCPU.duration, resGPU.duration);
  if (verbose)
  {
    std::cout << "Order " << level << " finished with " << resCPU.tests + resGPU.tests << " tests in "
              << duration << " ms.\n"
              << std::endl;
  }
  return {resCPU, resGPU};
}
