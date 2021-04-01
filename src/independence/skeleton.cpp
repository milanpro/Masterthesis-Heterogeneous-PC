#include "../util/cuda_util.cuh"
#include "../loadbalance/balancer.hpp"
#include "skeleton.hpp"
#include "compact.cuh"
#include <iostream>
#include <fstream>
#include <string>
#include <future>
#include <tuple>
#include <vector>
#include <cmath>
#include "skeleton.hpp"
#include <omp.h>

typedef std::tuple<int64_t, int64_t, std::tuple<TestResult, TestResult>> LevelMetrics;

LevelMetrics calcLevel(MMState *state, std::vector<int> gpuList, int level, bool verbose, bool workstealing, Balancer *balancer)
{
  auto start = std::chrono::system_clock::now();
  int numberOfGPUs = gpuList.size();
  int device_row_count = (state->p + numberOfGPUs - 1) / numberOfGPUs;
  if (level >= 1)
  {
#pragma omp parallel for
    for (int i = 0; i < numberOfGPUs; i++)
    {
      callCompact(state, gpuList[i], i, numberOfGPUs, device_row_count);
    }
  }

  if (level != 0)
  {
    std::ofstream csvFile;
    csvFile.open("" + level + "test_iteration_map.csv", std::ios::app | std::ios::out);

    for (int row = 0; row < state->p; row++)
    {
      int row_length = state->adj_compact[row * state->p + state->p - 1];
      if (row_length >= level)
      {
        size_t row_test_count = level > 1 ? binomialCoeff(row_length - 1, level) : row_length - 1;
        int test_iterations_gpu = std::ceil((float)row_test_count / (float)NUMTHREADS);
        csvFile << row_test_count << "," << test_iterations_gpu << std::endl;
      }
    }
    csvFile.close();
  }

  std::tuple<TestResult, TestResult> execRes;
  int64_t balanceDur = 0;
  if (workstealing)
  {
    if (level == 0)
    {
      balancer->gpuExecutor->enqueueSplitTask(SplitTask{0, (int)state->p});
    }
    for (int i = 0; i < numberOfGPUs; i++)
    {
      state->prefetchRows(i * device_row_count, device_row_count, gpuList[i]);
    }
    execRes = balancer->executeWorkstealing(level);
  }
  else
  {
    balanceDur = balancer->balance(level);
    execRes = balancer->execute(level);
  }

  auto levelDur = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::system_clock::now() - start)
                      .count();

  return {levelDur, balanceDur, execRes};
}

void calcSkeleton(MMState *state, std::vector<int> gpuList, bool verbose, bool workstealing, std::string csvExportFile, int heterogeneity, bool showSepsets)
{

  if (verbose)
    std::cout << "maxCondSize: " << state->maxCondSize
              << "  observations: " << state->observations
              << "  p: " << state->p << " number of GPUS: " << gpuList.size() << std::endl;

  auto start = std::chrono::system_clock::now();
  state->adviceReadonlyCor(gpuList);
  state->memAdvise(gpuList);

  auto balancer = Balancer(gpuList, state, static_cast<Heterogeneity>(heterogeneity), verbose);

  std::vector<LevelMetrics> levelMetrics;
  for (int lvl = 0; lvl <= state->maxLevel; lvl++)
  {
    auto metric = calcLevel(state, gpuList, lvl, verbose, workstealing, &balancer);
    levelMetrics.push_back(metric);
  }

  auto executionDuration = std::chrono::duration_cast<std::chrono::microseconds>(
                               std::chrono::system_clock::now() - start)
                               .count();

  if (verbose)
  {
    std::cout << "Execution duration: " << executionDuration << " \u03BCs." << std::endl;
  }

  int nrEdges = printSepsets(state, showSepsets);

  if (csvExportFile != "")
  {
    std::ofstream csvFile;
    csvFile.open(csvExportFile, std::ios::app | std::ios::out);

    int numGPUs = balancer.heterogeneity == Heterogeneity::CPUOnly ? 0 : gpuList.size();
    csvFile << numGPUs << ",";

    int numOMPThreads = omp_get_max_threads();
    csvFile << numOMPThreads << ",";

    csvFile << nrEdges << ",";

    for (auto metric : levelMetrics)
    {
      auto [levelDur, balanceDur, execRes] = metric;
      auto [cpuRes, gpuRes] = execRes;
      csvFile << levelDur << "," << balanceDur << "," << cpuRes.duration << "," << gpuRes.duration << ",";
    }

    csvFile << executionDuration << std::endl;

    csvFile.close();
  }
}

int printSepsets(MMState *state, bool verbose)
{
  int nrEdges = 0;
  for (int i = 0; i < state->p; i++)
  {
    for (int j = i + 1; j < state->p; j++)
    {
      if (!state->adj[i * state->p + j])
      {
        if (verbose)
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
      }
      else
      {
        //std::cout << "Edge from " << i << " to " << j << std::endl;
        nrEdges++;
      }
    }
  }
  if (verbose)
  {
    std::cout << "Total number of edges: " << nrEdges << std::endl;
  }
  return nrEdges;
}
