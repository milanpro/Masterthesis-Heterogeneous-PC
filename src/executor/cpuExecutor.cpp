#include "cpuExecutor.hpp"
#include "./testing/cpuWorkstealingTests.hpp"
#include "./testing/cpuRowTests.hpp"
#include "./testing/cpuUtil.hpp"
#include <chrono>
#include <iostream>
#include <vector>
#include <tuple>
#include <omp.h>
#include <atomic>
#include <cmath>

bool compTuple(std::tuple<int, int> i, std::tuple<int, int> j) { return (std::get<1>(i) > std::get<1>(j)); }

TestResult CPUExecutor::workstealingExecuteLevel(int level, bool verbose)
{
  if (level == 0)
  {
    return TestResult{0, 0};
  }
  auto start = std::chrono::system_clock::now();
  std::atomic<int> edges_done = 0;
  bool gpu_done = level % 2 == 1;
  int p = (int)state->p;
  int max_row_length = std::get<1>(rowLengthMap[0]);
  int row_count = rowLengthMap.size();
  int edge_count = row_count * max_row_length;
#pragma omp parallel
  {
    int id = omp_get_thread_num();
    int offset = omp_get_num_threads();
    int idx = edge_count - id;
    int col = idx % max_row_length;
    int row = row_count - ((idx - col) / max_row_length);
    while (state->gpu_done != gpu_done)
    {
      auto [row_node, row_length] = rowLengthMap[row];
      if (row_length > col && row_length > level)
      {
        auto col_node = state->adj_compact[row_node * p + col];
        bool expected = false;
        if (col_node != row_node)
        {
#if WITH_CUDA_ATOMICS
          bool active = state->node_status[row_node * p + col_node].compare_exchange_strong(expected, true);
#else
          bool active = !state->node_status[row_node * p + col_node];
#endif
          if (active)
          {
#if WITH_CUDA_ATOMICS
            state->node_status[row_node * p + col_node] = true;
#endif
            testEdgeWorkstealing(state, row_node, col, col_node, deletedEdges, row_length, edges_done, level);
          }
        }
      }

      idx -= offset;
      if (idx < 0)
      {
        break;
      }
      col = idx % max_row_length;
      row = row_count - ((idx - col) / max_row_length);
    }
  }

  auto duration = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                            std::chrono::system_clock::now() - start)
                                            .count());
  if (verbose)
  {
    std::cout
        << "\tCPU is done. Time: " << (int)duration << " ms. Edges stolen: " << edges_done << std::endl;
  }
  return TestResult{duration, 0};
}

TestResult CPUExecutor::executeLevel(int level, bool verbose)
{
  if (tasks.size() == 0)
  {
    return {0, 0};
  }
  auto start = std::chrono::system_clock::now();

  std::vector<std::tuple<int, int>> sortedRows;
  for (auto task : tasks)
  {
    for (auto i = task.row; i < task.row + task.rowCount; i++)
    {
      int row_length = state->adj_compact[i * state->p + state->p - 1];
      if (row_length >= level)
      {
        sortedRows.push_back({i, row_length});
      }
    }
  }
  std::sort(sortedRows.begin(), sortedRows.end(), compTuple);
  int p = (int)state->p;
#pragma omp parallel for shared(state, level, sortedRows, p) default(none) collapse(2) schedule(dynamic, 10)
  for (auto i = 0; i < sortedRows.size(); i++)
  {
    for (int col_node = 0; col_node < p; col_node++)
    {
      testEdge(level, state, std::get<0>(sortedRows[i]), col_node, deletedEdges);
    }
  }

  auto duration = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                            std::chrono::system_clock::now() - start)
                                            .count());
  if (verbose)
  {
    std::cout << "\tCPU is done. Time: " << duration << " ms." << std::endl;
  }
  return TestResult{duration, 0};
}

void CPUExecutor::migrateEdges(int level, bool verbose)
{
  if (verbose)
  {
    std::cout << "Migrating " << deletedEdges->size_approx() << " CPU edges..." << std::endl;
  }

  DeletedEdge delEdge;
  while (deletedEdges->try_dequeue(delEdge))
  {
    if (level == 0)
    {
      deleteEdgeLevel0(state, delEdge.col, delEdge.row, delEdge.pMax);
    }
    else
    {
      deleteEdge(state, level, delEdge.col, delEdge.row, delEdge.pMax, delEdge.sepSet);
    }
  }
}

void CPUExecutor::calculateRowLengthMap(int level)
{
  rowLengthMap.clear();

  for (int row = 0; row < state->p; row++)
  {
    int row_length = state->adj_compact[row * state->p + state->p - 1];
    if (row_length >= level)
    {
      rowLengthMap.push_back({row, row_length});
    }
  }

  std::sort(rowLengthMap.begin(), rowLengthMap.end(), compTuple);
}
