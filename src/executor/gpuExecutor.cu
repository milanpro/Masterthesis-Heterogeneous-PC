#include "gpuExecutor.cuh"
#include "../util/cuda_util.cuh"
#include "./testing/gpuRowTests.cuh"
#include "./testing/gpuWorkstealingTests.cuh"
#include <cmath>
#include <chrono>
#include <iostream>

TestResult GPUExecutor::executeLevel(int level, bool workstealing, int maxRowLength, bool verbose)
{
  if (tasks.size() == 0)
  {
    return {0, 0};
  }

  int *rows;
  int row_count = 0;
  checkCudaErrors(cudaMallocManaged(&rows, (uint64_t)sizeof(int) * state->p));

  for (auto task : tasks)
  {
    for (auto i = task.row; i < task.row + task.rowCount; i++)
    {
      if (state->adj_compact[i * state->p + state->p - 1] >= level) {
        rows[row_count] = i;
        row_count++;
      }
    }
  }

  if (row_count == 0) {
    return {0, 0};
  }

  auto start = std::chrono::system_clock::now();
  int numthreads = NUMTHREADS;
  dim3 block = dim3(numthreads);
  dim3 grid;

  int numberOfGPUs = gpuList.size();
  int rowsPerGPU = (int)std::ceil((float)row_count / (float)numberOfGPUs);

  if (level == 0)
  {
    int max_rows = (int)std::ceil((float)rowsPerGPU * (float)state->p / (float)numthreads);
    grid = dim3(max_rows);
  }
  else
  {
    std::cout << "rowsPerGPU " << rowsPerGPU << " maxRowLength " << maxRowLength << std::endl;
    grid = dim3(rowsPerGPU, maxRowLength);
  }

#pragma omp parallel for num_threads(numberOfGPUs) if (numberOfGPUs > 1)
  for (int i = 0; i < numberOfGPUs; i++)
  {
    int deviceId = gpuList[i];
    int startRow = i * rowsPerGPU;
    checkCudaErrors(cudaSetDevice(deviceId));
    if (workstealing)
    {
      switch (level)
      {
      case 0:
        testRowL0<<<grid, block>>>(*state, startRow, rowsPerGPU);
        break;
      case 1:
        testRowWorkstealingL1<<<grid, block, sizeof(double) * numthreads>>>(*state, rows, startRow, row_count);
        break;
      case 2:
        testRowWorkstealingLN<4, 2><<<grid, block>>>(*state, rows, startRow, row_count);
        break;
      case 3:
        testRowWorkstealingLN<5, 3><<<grid, block>>>(*state, rows, startRow, row_count);
        break;
      }
    }
    else
    {
      switch (level)
      {
      case 0:
        testRowL0<<<grid, block>>>(*state, startRow, rowsPerGPU);
        break;
      case 1:
        testRowL1<<<grid, block, sizeof(double) * numthreads>>>(*state, rows, startRow, row_count);
        break;
      case 2:
        testRowLN<4, 2><<<grid, block>>>(*state, rows, startRow, row_count);
        break;
      case 3:
        testRowLN<5, 3><<<grid, block>>>(*state, rows, startRow, row_count);
        break;
      }
    }
    std::string msg = "L " + std::to_string(level) + " Kernel execution failed";
    cudaDeviceSynchronize();
    checkLastCudaError(msg.c_str());
  }

  if (workstealing) {
    state->gpu_done = level % 2 == 1;
  }
  
  auto duration = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                            std::chrono::system_clock::now() - start)
                                            .count());
  if (verbose)
  {
    std::cout << "\tGPU is done. Time: " << (int)duration << " \u03BCs." << std::endl;
  }
  return TestResult{duration, (int)(state->p * (state->p - 1L)) / 2};
}
