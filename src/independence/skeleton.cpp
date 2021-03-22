#include "../util/cuda_util.cuh"
#include "../loadbalance/balancer.hpp"
#include "skeleton.hpp"
#include "compact.cuh"
#include <iostream>
#include <string>
#include <future>
#include <vector>

unsigned long long calcLevel(MMState *state, int maxMem, std::vector<int> gpuList, int level, bool verbose, Balancer *balancer)
{
  int numberOfGPUs = gpuList.size();
  if (level >= 1)
  {
    int device_row_count = (state->p + numberOfGPUs - 1) / numberOfGPUs;
#pragma omp parallel for
    for (int i = 0; i < numberOfGPUs; i++)
    {
      callCompact(state, gpuList[i], i, numberOfGPUs, device_row_count);
    }
  }

  balancer->balance(level);

  return balancer->execute(level);
}

void calcSkeleton(MMState *state, std::vector<int> gpuList, bool verbose, int heterogeneity, int maxMem,
                  int startLevel)
{
  
  if (verbose)
    std::cout << "maxCondSize: " << state->maxCondSize
              << "  observations: " << state->observations
              << "  p: " << state->p << " number of GPUS: " << gpuList.size() << std::endl;

  state->adviceReadonlyCor(gpuList);
  state->memAdvise(gpuList);

  auto balancer = Balancer(gpuList, state, static_cast<Heterogeneity>(heterogeneity), verbose);

  unsigned long long duratonSum = 0;
  for (int lvl = startLevel; lvl <= state->maxLevel; lvl++)
  {
    duratonSum += calcLevel(state, maxMem, gpuList, lvl, verbose, &balancer);
  }

  if (verbose)
  {
    std::cout << "Summed execution duration: " << duratonSum << " \u03BCs." << std::endl;

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
