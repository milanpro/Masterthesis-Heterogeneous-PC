#include "../util/cuda_util.cuh"
#include "../util/csv_parser.hpp"
#include "../util/sepset_util.hpp"
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

SkeletonCalculator::SkeletonCalculator(int maxLevel, double alpha, bool workstealing, std::string csvExportFile, bool verbose) : maxLevel(maxLevel), alpha(alpha), workstealing(workstealing), csvExportFile(csvExportFile), verbose(verbose)
{
  gpuList = {0};
  heterogeneity = Heterogeneity::All;
}

SkeletonCalculator::set_heterogeneity(Heterogeneity heterogeneity) : heterogeneity(heterogeneity)
{
  return this;
}

SkeletonCalculator::set_gpu_list(std::vector<int> gpuList) : gpuList(gpuList)
{
  return this;
}

SkeletonCalculator::read_csv_file(std::string input_file)
{
  string _match(inputFile);
  shared_ptr<arma::mat> array_data;
  vector<string> column_names(0);

  if (_match.find(".csv") != string::npos)
  {
    array_data = CSVParser::read_csv_to_mat(inputFile, column_names);
  }
  else
  {
    cout << "Cannot process file '" << inputFile << "\'." << endl;
    cout << "Has to be .csv format." << endl;
    exit(-1);
  }
  return array_data;
}

SkeletonCalculator::add_observations(std::string input_file, bool use_p9_ats)
{
  auto array_data = this->read_csv_file(input_file);
  state = MMState(array_data.get()->n_cols, (int)array_data.get()->n_rows, alpha, maxLevel, gpuList[0], use_p9_ats);
  gpuPMCC(array_data.get()->begin(), state.p, state.observations, state.cor, gpuList[0], verbose);
  return this;
}

SkeletonCalculator::add_correlation_matrix(std::string input_file, int observation_count, bool use_p9_ats)
{
  auto array_data = this->read_csv_file(input_file);
  state = MMState(array_data.get()->n_cols, observation_count, alpha, maxLevel, gpuList[0], use_p9_ats);
  memcpy(state.cor, array_data.get()->begin(), state.p * state.p * sizeof(double));
  return this;
}

SkeletonCalculator::initialize_balancer(std::tuple<float, float, float> balancer_thresholds)
{
  balancer = Balancer(gpuList, &state, balancer_thresholds, heterogeneity, verbose);
  return this;
}

SkeletonCalculator::export_metrics(int nrEdges)
{
  if (csvExportFile != "")
  {
    std::ofstream csvFile;
    csvFile.open(csvExportFile, std::ios::app | std::ios::out);

    int numGPUs = heterogeneity == Heterogeneity::CPUOnly ? 0 : gpuList.size();
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

SkeletonCalculator::calcLevel(int level)
{
  auto start = std::chrono::system_clock::now();
  int numberOfGPUs = gpuList.size();
  int device_row_count = (state.p + numberOfGPUs - 1) / numberOfGPUs;
  if (level >= 1)
  {
#pragma omp parallel for
    for (int i = 0; i < numberOfGPUs; i++)
    {
      callCompact(&state, gpuList[i], i, numberOfGPUs, device_row_count);
    }

#ifndef NDEBUG
    assertAdjCompactIsAdj(state);
#endif
  }

  if (verbose)
    std::cout << "Max row length: " << state.max_adj[0] << std::endl;

  std::tuple<TestResult, TestResult> execRes;
  int64_t balanceDur = 0;
  if (workstealing)
  {
    if (level == 0)
    {
      balancer.gpuExecutor->enqueueSplitTask(SplitTask{0, (int)state.p});
    }
    for (int i = 0; i < numberOfGPUs; i++)
    {
      state->prefetchRows(i * device_row_count, device_row_count, gpuList[i]);
    }
    execRes = balancer.executeWorkstealing(level);
  }
  else
  {
    balanceDur = balancer.balance(level);
    execRes = balancer.execute(level);
  }

  auto levelDur = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::system_clock::now() - start)
                      .count();

  return {levelDur, balanceDur, execRes};
}

SkeletonCalculator::run(bool print_sepsets)
{

  if (verbose)
    std::cout << "maxCondSize: " << state.maxCondSize
              << "  observations: " << state.observations
              << "  p: " << state.p << " number of GPUS: " << gpuList.size() << std::endl;

  auto start = std::chrono::system_clock::now();
  state.adviceReadonlyCor(gpuList);
  state.memAdvise(gpuList);

  std::vector<LevelMetrics> levelMetrics;
  for (int lvl = 0; lvl <= state.maxLevel; lvl++)
  {
    auto metric = this->calcLevel(lvl);
    levelMetrics.push_back(metric);
  }

  auto executionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::system_clock::now() - start)
                               .count();

  if (verbose)
  {
    std::cout << "Execution duration: " << executionDuration << " ms." << std::endl;
  }

  int nrEdges = printSepsets(&state, print_sepsets, verbose);

  this->export_metrics(nrEdges);
}

void assertAdjCompactIsAdj(MMState *state)
{
  int active_edges = 0;
  for (int i = 0; i < state->p; i++)
  {
    int row_length = 0;
    for (int j = 0; j < state->p; j++)
    {
      int deletedAdj = state->adj[i * state->p + j];
      int deletedAdjComp = 0;
      for (int k = 0; k < state->adj_compact[i * state->p + state->p - 1]; k++)
      {
        if (state->adj_compact[i * state->p + k] == j)
        {
          deletedAdjComp = 1;
          row_length++;
          break;
        }
      }
      assert(deletedAdj == deletedAdjComp);
    }
    assert(state->adj_compact[i * state->p + state->p - 1] == row_length);
    active_edges += row_length;
  }
  std::cout << "Active edges: " << active_edges << std::endl;
}
