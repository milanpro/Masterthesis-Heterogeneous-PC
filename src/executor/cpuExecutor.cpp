#include "cpuExecutor.hpp"
#include "./testing/cpuWorkstealingTests.hpp"
#include "./testing/cpuRowTests.hpp"
#include <chrono>
#include <iostream>
#include <vector>
#include <tuple>
#include <omp.h>
#include <atomic>

TestResult CPUExecutor::workstealingExecuteLevel(int level, bool verbose)
{
  if (level == 0)
  {
    return TestResult{0, 0};
  }
  auto start = std::chrono::system_clock::now();
  std::atomic<int> edges_done = 0;
  bool edge_done = level % 2 == 1;
#pragma omp parallel
  {
    int id = omp_get_thread_num();
    int offset = omp_get_num_threads();
    int row = id;
    while (row < rowLengthMap.size() && state->gpu_done != edge_done)
    {
      auto [row_node, row_length] = rowLengthMap[row];
      for (int i = row_length - 1; i >= 0; i--)
      {
        auto col_node = state->adj_compact[row_node * state->p + i];
        if (state->node_status[row_node * state->p + col_node] != edge_done)
        {
          if (level == 1)
          {
            testEdgeWorkstealingL1(state, row_node, i, col_node, deletedEdges, row_length, edges_done);
          }
          else
          {
            testEdgeWorkstealingLN(state, row_node, i, col_node, deletedEdges, row_length, edges_done, edge_done, level);
          }
        }
      }
      row += offset;
    }
  }

  state->gpu_done = edge_done;

  auto duration = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                            std::chrono::system_clock::now() - start)
                                            .count());
  if (verbose)
  {
    std::cout
        << "\tCPU is done. Time: " << (int)duration << " \u03BCs. Edges stolen: " << edges_done << std::endl;
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

#pragma omp parallel for shared(state, level, tasks) default(none) collapse(2) schedule(guided)
  for (auto i = 0; i < tasks.size(); i++)
  {
    for (int col_node = 0; col_node < state->p; col_node++)
    {
      testEdge(level, state, tasks[i].row, col_node, deletedEdges);
    }
  }

  auto duration = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                            std::chrono::system_clock::now() - start)
                                            .count());
  if (verbose)
  {
    std::cout << "\tCPU is done. Time: " << (int)duration << " \u03BCs." << std::endl;
  }
  return TestResult{duration, 0};
}

void CPUExecutor::migrateEdges(int level, bool verbose)
{
  if (verbose)
  {
    std::cout << "Migrating CPU edges..." << std::endl;
  }

  DeletedEdge delEdge;
  while (deletedEdges->try_dequeue(delEdge))
  {
    state->adj[state->p * delEdge.row + delEdge.col] = 0;
    state->adj[state->p * delEdge.col + delEdge.row] = 0;

    if (delEdge.row < delEdge.col)
    {
      state->pMax[state->p * delEdge.row + delEdge.col] = delEdge.pMax;
      if (level == 0)
      {
        state->sepSets[delEdge.row * state->p * state->maxCondSize +
                       delEdge.col * state->maxCondSize] = -2;
      }
      else
      {
        for (int j = 0; j < level; ++j)
        {
          state->sepSets[delEdge.row * state->p * state->maxCondSize +
                         delEdge.col * state->maxCondSize + j] = delEdge.sepSet[j];
        }
      }
    }
    else
    {
      state->pMax[state->p * delEdge.col + delEdge.row] = delEdge.pMax;
      if (level == 0)
      {
        state->sepSets[delEdge.col * state->p * state->maxCondSize +
                       delEdge.row * state->maxCondSize] = -2;
      }
      else
      {
        for (int j = 0; j < level; ++j)
        {
          state->sepSets[delEdge.col * state->p * state->maxCondSize +
                         delEdge.row * state->maxCondSize + j] = delEdge.sepSet[j];
        }
      }
    }
  }
}

bool compTuple(std::tuple<int, int> i, std::tuple<int, int> j) { return (std::get<1>(i) > std::get<1>(j)); }

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
