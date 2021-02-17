#include "../util/cuda_util.cuh"
#include "../util/matrix_print.cuh"
#include "../util/constants.hpp"
#include "cpu.hpp"
#include "gpu.cuh"
#include <armadillo>
#include "skeleton.hpp"
#include "compact.cuh"
#include <iostream>
#include <string>
#include <unordered_set>
#include <memory>
#include <thread>
#include <future>

void calcLevel(MMState *state, int maxMem, int maxEdgeCount, int level, int numberOfGPUs)
{
  if (level >= 2)
  {
    int device_row_count = state->p / numberOfGPUs;
    int max_additional_row_index = state->p % numberOfGPUs;
    #pragma omp parallel for
    for (int i = 0; i < numberOfGPUs; i++)
    {
      int actual_device_row_count =
        device_row_count + (i < max_additional_row_index);
      callCompact(state, i, numberOfGPUs, actual_device_row_count);
    }
  }

  auto cpuQueue = std::unique_ptr<SplitTaskQueue>(new SplitTaskQueue());
  auto gpuQueue = std::unique_ptr<SplitTaskQueue>(new SplitTaskQueue());
  for (int row = 0; row < state->p; row++)
  {
    if (row % 3 == 0)
    {
      cpuQueue->enqueue(SplitTask{row, 1});
    }
    else if (row % 3 == 1)
    {
      gpuQueue->enqueue(SplitTask{row, 2});
    }
  }
  auto resCPUFuture = std::async(CPU::executeLevel, level, state, cpuQueue.get());
  auto resGPUFuture = std::async(GPU::executeLevel, level, state, gpuQueue.get(), maxEdgeCount, numberOfGPUs);

  TestResult resCPU = resCPUFuture.get();
  TestResult resGPU = resGPUFuture.get();

  if (VERBOSE)
  {
    std::cout << "Order " << level << " finished with " << resCPU.tests + resGPU.tests << " tests in "
              << std::max(resCPU.duration, resGPU.duration) << " µs." << std::endl;
    std::cout << "\t CPU time: " << resCPU.duration << " µs GPU time: "
              << resGPU.duration << " µs." << std::endl;
  }
}

void calcSkeleton(MMState *state, int numberOfGPUs, int maxMem,
                  int startLevel)
{
  int maxEdgeCount = state->p * (state->p - 1L) / 2;
  if (VERBOSE)
    std::cout << "maxCondSize: " << state->maxCondSize
              << "  observations: " << state->observations
              << "  p: " << state->p << " number of GPUS: " << numberOfGPUs << std::endl;

  for (int lvl = startLevel; lvl <= state->maxLevel; lvl++)
  {
    calcLevel(state, maxMem, maxEdgeCount, lvl, numberOfGPUs);
  }

  if (VERBOSE)
  {
    printSepsets(state);
  }
}

void printSepsets(MMState *state)
{
  int nrEdges = 0;
  for (int i = 0; i < state->p; i++)
  {
    for (int j = i + 1; j < state->p; j++)
    {
      if (!state->adj[i * state->p + j])
      {
        std::string sepset_string = "";
        for (int k = 0; k < state->maxCondSize; k++)
        {
          int current_sepset_node =
              state->sepSets[(i * state->maxCondSize * state->p) +
                             (j * state->maxCondSize) + k];
          if (current_sepset_node == -2)
          {
            std::cout << "Separation from " << i << " to " << j << std::endl;
            break;
          }
          else if (current_sepset_node == -1)
          {
            break;
          }
          else
          {
            sepset_string.append(std::to_string(current_sepset_node));
            sepset_string.append(" ");
          }
        }
        if (sepset_string != "")
        {
          std::cout << "Separation from " << i << " to " << j << " via "
                    << sepset_string << std::endl;
        }
      }
      else
      {
        //std::cout << "Edge from " << i << " to " << j << std::endl;
        nrEdges++;
      }
    }
  }
  std::cout << "Total number of edges: " << nrEdges << std::endl;
}
