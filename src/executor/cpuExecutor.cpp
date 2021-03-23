#include "cpuExecutor.hpp"
#include "armadillo"
#include "boost/math/distributions/normal.hpp"
#include "boost/math/special_functions/log1p.hpp"
#include <chrono>
#include <iostream>
#include <vector>
#include <tuple>
#include <omp.h>

namespace CPU
{
#define CUT_THR 0.9999999
  double calcPValue(double r_in, int sampleSize)
  {
    double r = boost::math::isnan(r_in) ? 0.0 : std::min(CUT_THR, std::abs(r_in));
    double absz = sqrt(sampleSize - 3.0) * 0.5 * boost::math::log1p(2 * r / (1 - r));
    boost::math::normal distN;
    return (2 * boost::math::cdf(boost::math::complement(distN, absz)));
  }

  double pValL1(double x1, double x2, double x3, int sampleSize)
  {
    // with edge i, j given k values are:
    // x1: edge i, j
    // x2: edge i, k
    // x3: edge j, k
    double r = (x1 - x2 * x3) / sqrt((1.0 - x3 * x3) * (1.0 - x2 * x2));
    return calcPValue(r, sampleSize);
  }

  double pValLN(arma::dmat Submat, int observations)
  {
    arma::mat SubmatPInv = arma::pinv(Submat);
    double r = -SubmatPInv(0, 1) / sqrt(SubmatPInv(0, 0) * SubmatPInv(1, 1));
    return calcPValue(r, observations);
  }

  void testRowL0Triangluar(MMState *state, int row_node, int col_node, std::shared_ptr<EdgeQueue> eQueue)
  {
    auto idx = state->p * row_node + col_node;
    if (col_node < row_node && state->adj[idx])
    {
      auto inv_idx = state->p * col_node + row_node;
      double pVal = calcPValue(state->cor[idx], state->observations);
      if (pVal >= state->alpha)
      {
        DeletedEdge result;
        result.col = col_node;
        result.row = row_node;
        result.pMax = pVal;
        result.sepSet = {};
        eQueue->enqueue(result);
      }
    }
  }

  void testRowL1(MMState *state, int row_node, int col_node, std::shared_ptr<EdgeQueue> eQueue)
  {
    int p = (int)state->p;
    int row_count = state->adj_compact[row_node * state->p + state->p - 1];
    if (col_node < p &&
        row_node < p &&
        row_count > col_node && // col_node not available
        row_count >= 1)
    {
      auto actual_col_node = state->adj_compact[row_node * state->p + col_node];
      int subIndex;
      for (int next = 0; next < row_count; next++)
      {
        if (next != col_node)
        {
          subIndex = state->adj_compact[row_node * p + next];
          double pVal = pValL1(
              state->cor[row_node * p + actual_col_node],
              state->cor[row_node * p + subIndex],
              state->cor[actual_col_node * p + subIndex], state->observations);
          // Check pVal in regards to alpha and delete edge + save sepset + save pMax (Needs compare and swap to lock against other threads)
          if (pVal >= state->alpha)
          {
            DeletedEdge result;
            result.col = actual_col_node;
            result.row = row_node;
            result.pMax = pVal;
            result.sepSet = {state->adj_compact[actual_col_node * p + subIndex]};
            eQueue->enqueue(result);
            break;
          }
        }
      }
    }
  }

  template <int lvlSize, int kLvlSizeSmall>
  void testRowLN(MMState *state, int row_node, int col_node, std::shared_ptr<EdgeQueue> eQueue)
  {
    int row_count = state->adj_compact[row_node * state->p + state->p - 1];

    if (col_node < state->p &&
        row_node < state->p &&
        row_count > col_node && // col_node not available
        row_count >= kLvlSizeSmall)
    {

      auto actual_col_node = state->adj_compact[row_node * state->p + col_node]; // get actual id
      int row_neighbours = row_count - 1;                                        // get number of neighbors && exclude col_node
      size_t row_test_count = binomialCoeff(row_neighbours, kLvlSizeSmall);
      int sepset_nodes[kLvlSizeSmall];

      for (size_t test_index = 0; test_index < row_test_count;
           test_index++)
      {
        ithCombination(sepset_nodes, test_index, kLvlSizeSmall,
                       row_neighbours);

        // Fill sepset_nodes array with actual ids
        for (int ind = 0; ind < kLvlSizeSmall; ++ind)
        {
          if (sepset_nodes[ind] - 1 >= col_node)
          {
            sepset_nodes[ind] =
                state->adj_compact[row_node * state->p + sepset_nodes[ind]];
          }
          else
          {
            sepset_nodes[ind] =
                state->adj_compact[row_node * state->p + sepset_nodes[ind] - 1];
          }
        }
        arma::dmat Submat(lvlSize, lvlSize, arma::fill::eye);

        Submat(0, 1) = Submat(1, 0) = state->cor[row_node * state->p + actual_col_node];

        for (int j = 2; j < lvlSize; ++j)
        {
          // set correlations of X
          Submat(0, j) = Submat(j, 0) =
              state->cor[row_node * state->p + sepset_nodes[j - 2]];
          // set correlations of Y
          Submat(1, j) = Submat(j, 1) =
              state->cor[actual_col_node * state->p + sepset_nodes[j - 2]];
        }
        for (int i = 2; i < lvlSize; ++i)
        {
          for (int j = i + 1; j < lvlSize; ++j)
          {
            Submat(i, j) = Submat(j, i) =
                state->cor[sepset_nodes[i - 2] * state->p + sepset_nodes[j - 2]];
          }
        }

        double pVal = pValLN(Submat, state->observations);

        // Check pVal in regards to alpha and delete edge + save sepset + save pMax (Needs compare and swap to lock against other threads)
        if (pVal >= state->alpha)
        {
          DeletedEdge result;
          result.col = actual_col_node;
          result.row = row_node;
          result.pMax = pVal;
          result.sepSet = {sepset_nodes[0]};
          for (int j = 1; j < kLvlSizeSmall; ++j)
          {
            result.sepSet.push_back(sepset_nodes[j]);
          }
          eQueue->enqueue(result);
          break;
        }
      }
    }
  }

  void testEdge(int level, MMState *state, int row_node, int col_node, std::shared_ptr<EdgeQueue> eQueue)
  {
    switch (level)
    {
    case 0:
      testRowL0Triangluar(state, row_node, col_node, eQueue);
      break;
    case 1:
      testRowL1(state, row_node, col_node, eQueue);
      break;
    case 2:
      testRowLN<4, 2>(state, row_node, col_node, eQueue);
      break;
    case 3:
      testRowLN<5, 3>(state, row_node, col_node, eQueue);
      break;
    }
  }

  void testEdgeWorkstealingL1(MMState *state, int row_node, int col_node, int actual_col_node, std::shared_ptr<EdgeQueue> eQueue, int row_count)
  {
    int p = (int)state->p;

    int subIndex;
    for (int next = 0; next < row_count; next++)
    {
      if (next != col_node)
      {
        subIndex = state->adj_compact[row_node * p + next];
        double pVal = pValL1(
            state->cor[row_node * p + actual_col_node],
            state->cor[row_node * p + subIndex],
            state->cor[actual_col_node * p + subIndex], state->observations);
        if (state->node_status[row_node * state->p + actual_col_node])
        {
          return;
        }
        // Check pVal in regards to alpha and delete edge + save sepset + save pMax (Needs compare and swap to lock against other threads)
        if (pVal >= state->alpha)
        {
          state->node_status[row_node * state->p + actual_col_node] = true;
          DeletedEdge result;
          result.col = actual_col_node;
          result.row = row_node;
          result.pMax = pVal;
          result.sepSet = {state->adj_compact[actual_col_node * p + subIndex]};
          eQueue->enqueue(result);
          return;
        }
      }
    }
    state->node_status[row_node * state->p + actual_col_node] = true;
  }

  void testEdgeWorkstealingLN(MMState *state, int row_node, int col_node, int actual_col_node, std::shared_ptr<EdgeQueue> eQueue, int row_count, bool edge_done, int level)
  {
    int row_neighbours = row_count - 1; // get number of neighbours && exclude col_node
    size_t row_test_count = binomialCoeff(row_neighbours, level);
    int sepset_nodes[level];
    int lvlSize = level + 2;
    for (size_t test_index = 0; test_index < row_test_count;
         test_index++)
    {
      if (state->node_status[row_node * state->p + actual_col_node] == edge_done)
      {
        return;
      }

      ithCombination(sepset_nodes, test_index, level,
                     row_neighbours);

      // Fill sepset_nodes array with actual ids
      for (int ind = 0; ind < level; ++ind)
      {
        if (sepset_nodes[ind] - 1 >= col_node)
        {
          sepset_nodes[ind] =
              state->adj_compact[row_node * state->p + sepset_nodes[ind]];
        }
        else
        {
          sepset_nodes[ind] =
              state->adj_compact[row_node * state->p + sepset_nodes[ind] - 1];
        }
      }

      arma::dmat Submat(lvlSize, lvlSize, arma::fill::eye);

      Submat(0, 1) = Submat(1, 0) = state->cor[row_node * state->p + actual_col_node];

      for (int j = 2; j < lvlSize; ++j)
      {
        // set correlations of X
        Submat(0, j) = Submat(j, 0) =
            state->cor[row_node * state->p + sepset_nodes[j - 2]];
        // set correlations of Y
        Submat(1, j) = Submat(j, 1) =
            state->cor[actual_col_node * state->p + sepset_nodes[j - 2]];
      }
      for (int i = 2; i < lvlSize; ++i)
      {
        for (int j = i + 1; j < lvlSize; ++j)
        {
          Submat(i, j) = Submat(j, i) =
              state->cor[sepset_nodes[i - 2] * state->p + sepset_nodes[j - 2]];
        }
      }

      double pVal = pValLN(Submat, state->observations);

      if (state->node_status[row_node * state->p + actual_col_node] == edge_done)
      {
        return;
      }
      // Check pVal in regards to alpha and delete edge + save sepset + save pMax (Needs compare and swap to lock against other threads)
      if (pVal >= state->alpha)
      {
        state->node_status[row_node * state->p + actual_col_node] = edge_done;
        DeletedEdge result;
        result.col = actual_col_node;
        result.row = row_node;
        result.pMax = pVal;
        result.sepSet = {sepset_nodes[0]};
        for (int j = 1; j < level; ++j)
        {
          result.sepSet.push_back(sepset_nodes[j]);
        }
        eQueue->enqueue(result);
        return;
      }
    }
    state->node_status[row_node * state->p + actual_col_node] = edge_done;
  }

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

TestResult CPUExecutor::workstealingExecuteLevel(int level, bool verbose)
{
  if (level == 0) {
      return TestResult{0, 0};
  }
  auto start = std::chrono::system_clock::now();

  std::vector<std::tuple<int, int>> row_length_map;

  for (int row = 0; row < state->p; row++)
  {
    int row_length = state->adj_compact[row * state->p + state->p - 1];
    if (row_length >= level)
    {
      row_length_map.push_back({row, row_length});
    }
  }

  std::sort(row_length_map.begin(), row_length_map.end(), compTuple);
  bool edge_done = level % 2 == 1;
#pragma omp parallel
  {
    int id = omp_get_thread_num();
    int offset = omp_get_num_threads();
    int row = id;
    while (row < row_length_map.size() && state->gpu_done != edge_done)
    {
      auto [row_node, row_length] = row_length_map[id];
      for (int i = 0; i < row_length; i++)
      {
        auto col_node = state->adj_compact[row_node * state->p + i];
        if (state->node_status[row_node * state->p + col_node] != edge_done)
        {
          if (level == 1)
          {
            CPU::testEdgeWorkstealingL1(state, row_node, i, col_node, deletedEdges, row_length);
          }
          else
          {
            CPU::testEdgeWorkstealingLN(state, row_node, i, col_node, deletedEdges, row_length, edge_done, level);
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
    std::cout << "\tCPU is done. Time: " << (int)duration << " \u03BCs." << std::endl;
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
      CPU::testEdge(level, state, tasks[i].row, col_node, deletedEdges);
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