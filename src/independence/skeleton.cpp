#include "../util/cuda_util.cuh"
#include "../util/matrix_print.cuh"
#include "../util/constants.hpp"
#include "cpu.hpp"
#include "gpu.cuh"
#include "skeleton.hpp"
#include "compact.cuh"
#include <iostream>
#include <string>
#include <future>
#include <vector>

void calcLevel(MMState *state, int maxMem, int maxEdgeCount, int level, int numberOfGPUs, bool verbose)
{
  if (level >= 2)
  {
    int device_row_count = ((int) state->p) / numberOfGPUs;
    int max_additional_row_index = state->p % numberOfGPUs;
    #pragma omp parallel for
    for (int i = 0; i < numberOfGPUs; i++)
    {
      int actual_device_row_count =
        device_row_count + (i < max_additional_row_index);
      callCompact(state, i, numberOfGPUs, actual_device_row_count);
    }
  }

  std::vector<SplitTask> CPURows;
  std::vector<SplitTask> GPURows;
  for (int row = 0; row < state->p; row++)
  {
    if (row % 2 == 0)
    {
      CPURows.push_back(SplitTask{row, 1});
    }
    else
    {
      GPURows.push_back(SplitTask{row, 1});
    }
  }
  auto resCPUFuture = std::async(CPU::executeLevel, level, state, CPURows);
  auto resGPUFuture = std::async(GPU::executeLevel, level, state, GPURows, maxEdgeCount, numberOfGPUs);

  TestResult resCPU = resCPUFuture.get();
  TestResult resGPU = resGPUFuture.get();

  if (verbose)
  {
    std::cout << "Order " << level << " finished with " << resCPU.tests + resGPU.tests << " tests in "
              << std::max(resCPU.duration, resGPU.duration) << " \u03BCs." << std::endl;
    std::cout << "\t CPU time: " << resCPU.duration << " \u03BCs GPU time: "
              << resGPU.duration << " \u03BCs." << std::endl;
  }
}

void calcSkeleton(MMState *state, int numberOfGPUs, bool verbose, int maxMem,
                  int startLevel)
{
  int maxEdgeCount = (int) (state->p * (state->p - 1L) / 2);
  if (verbose)
    std::cout << "maxCondSize: " << state->maxCondSize
              << "  observations: " << state->observations
              << "  p: " << state->p << " number of GPUS: " << numberOfGPUs << std::endl;

  for (int lvl = startLevel; lvl <= state->maxLevel; lvl++)
  {
    calcLevel(state, maxMem, maxEdgeCount, lvl, numberOfGPUs, verbose);
  }

  if (verbose)
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
