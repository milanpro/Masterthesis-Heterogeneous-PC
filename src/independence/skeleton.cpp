#include <iostream>
#include <fstream>
#include "./skeleton.hpp"
#include "./compact.cuh"
#include "../util/cuda_util.cuh"
#include "../util/csv_parser.hpp"
#include "../util/sepset_util.hpp"
#include "../util/assertions.hpp"
#include "../correlation/corOwn.cuh"

using namespace std;

SkeletonCalculator::SkeletonCalculator(int maxLevel, double alpha, bool workstealing, string csvExportFile, int numThreads, bool verbose) : maxLevel(maxLevel), alpha(alpha), workstealing(workstealing), csvExportFile(csvExportFile), numThreads(numThreads), verbose(verbose)
{
  gpuList = {0};
  heterogeneity = Heterogeneity::All;
}

void SkeletonCalculator::set_heterogeneity(Heterogeneity heterogeneity)
{
  this->heterogeneity = heterogeneity;
}

void SkeletonCalculator::set_gpu_list(vector<int> gpuList)
{
  this->gpuList = gpuList;
}

shared_ptr<arma::mat> SkeletonCalculator::read_csv_file(string input_file)
{
  if (verbose)
  {
    cout << "Reading file: " << input_file << endl;
  }

  string _match(input_file);
  shared_ptr<arma::mat> array_data;
  vector<string> column_names(0);

  if (_match.find(".csv") != string::npos)
  {
    array_data = CSVParser::read_csv_to_mat(input_file, column_names);
  }
  else
  {
    cout << "Cannot process file '" << input_file << "\'." << endl;
    cout << "Has to be .csv format." << endl;
    exit(-1);
  }
  return array_data;
}

void SkeletonCalculator::add_observations(string input_file, bool use_p9_ats)
{
  auto array_data = this->read_csv_file(input_file);
  state = MMState(array_data.get()->n_cols, (int)array_data.get()->n_rows, alpha, maxLevel, gpuList[0], use_p9_ats);
  gpuPMCC(array_data.get()->begin(), state.p, state.observations, state.cor, gpuList[0], verbose);
}

void SkeletonCalculator::add_correlation_matrix(string input_file, int observation_count, bool use_p9_ats)
{
  auto array_data = this->read_csv_file(input_file);
  state = MMState(array_data.get()->n_cols, observation_count, alpha, maxLevel, gpuList[0], use_p9_ats);
  memcpy(state.cor, array_data.get()->begin(), state.p * state.p * sizeof(double));
}

void SkeletonCalculator::initialize_balancer(tuple<float, float, float> balancer_thresholds)
{
  balancer = Balancer(gpuList, &state, balancer_thresholds, heterogeneity, verbose);
}

void SkeletonCalculator::export_metrics(vector<LevelMetrics> levelMetrics, int64_t executionDuration, int nrEdges)
{
  if (csvExportFile != "")
  {
    if (verbose)
    {
      cout << "Export metrics to CSV file: " << csvExportFile << endl;
    }
    ofstream csvFile;
    csvFile.open(csvExportFile, ios::app | ios::out);

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

    csvFile << executionDuration << endl;

    csvFile.close();
  }
}

LevelMetrics SkeletonCalculator::calcLevel(int level)
{
  auto start = chrono::system_clock::now();
  int numberOfGPUs = gpuList.size();
  int device_row_count = (state.p + numberOfGPUs - 1) / numberOfGPUs;
  if (level >= 1)
  {
#pragma omp parallel for num_threads(numberOfGPUs) 
    for (int i = 0; i < numberOfGPUs; i++)
    {
      callCompact(&state, gpuList[i], i, numberOfGPUs, device_row_count);
    }

#ifndef NDEBUG
    assertAdjCompactIsAdj(&state);
#endif
  }

  if (verbose)
    cout << "Max row length: " << state.max_adj[0] << endl;

  tuple<TestResult, TestResult> execRes;
  int64_t balanceDur = 0;
  if (workstealing)
  {
    if (level == 0)
    {
      balancer.gpuExecutor->enqueueSplitTask(SplitTask{0, (int)state.p});
    }
    for (int i = 0; i < numberOfGPUs; i++)
    {
      state.prefetchRows(i * device_row_count, device_row_count, gpuList[i]);
    }
    execRes = balancer.executeWorkstealing(level, numThreads);
  }
  else
  {
    balanceDur = balancer.balance(level);
    execRes = balancer.execute(level, numThreads);
  }

  auto levelDur = chrono::duration_cast<chrono::milliseconds>(
                      chrono::system_clock::now() - start)
                      .count();

  return {levelDur, balanceDur, execRes};
}

void SkeletonCalculator::run(bool print_sepsets)
{

  if (verbose)
  {
    cout << "Using " << omp_get_max_threads() << " OpenMP thread(s) in pool" << endl;
    cout << "Using following GPUs:" << endl;
    for (auto deviceId : gpuList)
    {
      cout << "\t" << deviceId << endl;
    }
    cout << "maxCondSize: " << state.maxCondSize
         << "  observations: " << state.observations
         << "  p: " << state.p << " number of GPUS: " << gpuList.size() << endl;
  }

  auto start = chrono::system_clock::now();
  state.adviceReadonlyCor(gpuList);
  state.memAdvise(gpuList);

  vector<LevelMetrics> levelMetrics;
  for (int lvl = 0; lvl <= state.maxLevel; lvl++)
  {
    auto metric = this->calcLevel(lvl);
    levelMetrics.push_back(metric);
  }

  auto executionDuration = chrono::duration_cast<chrono::milliseconds>(
                               chrono::system_clock::now() - start)
                               .count();

  if (verbose)
  {
    cout << "Execution duration: " << executionDuration << " ms." << endl;
  }

  int nrEdges = printSepsets(&state, print_sepsets, verbose);

  this->export_metrics(levelMetrics, executionDuration, nrEdges);
}
