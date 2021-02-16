#include "../util/indep_util.hpp"
#include "../util/state.cuh"
#include "armadillo"
#include "boost/dynamic_bitset.hpp"
#include "boost/math/distributions/normal.hpp"
#include "boost/math/special_functions/fpclassify.hpp"
#include "boost/math/special_functions/log1p.hpp"
#include <unordered_map>
#include <iostream>
#include <chrono>
#include "omp.h"

const int MULTITHREAD_THRESHHOLD = 100;

void testRowL0TriangluarCPU(GPUState *state, int row_node, int col_node)
{
  int idx = state->p * row_node + col_node;
  if (col_node < row_node && state->adj[idx])
  {
    int inv_idx = state->p * col_node + row_node;
    double pVal = calcPValue(state->cor[idx], state->observations);
    state->pMax[inv_idx] = pVal;
    if (state->pMax[inv_idx] >= state->alpha)
    {
      state->adj[idx] = 0;
      state->adj[inv_idx] = 0;
      state->adj_compact[idx] = 0;
      state->adj_compact[inv_idx] = 0;
      state->sepSets[(col_node * state->maxCondSize * state->p) +
                     (row_node * state->maxCondSize)] = -2;
    }
  }
}

void testRowL1TriangluarCPU(GPUState *state, int row_node, int col_node)
{
  int p = state->p;
  int idx = p * row_node + col_node;
  if (col_node < row_node && state->adj[idx])
  {
    for (int next = 0; next < p; next++)
    {
      if (row_node != next && col_node != next)
      {
        if (state->adj[row_node * p + next] != 0 || state->adj[col_node * p + next] != 0)
        {
          int pVal = pValL1(
              state->cor[idx],
              state->cor[row_node * p + next],
              state->cor[col_node * p + next], state->observations);
          if (pVal > state->pMax[idx])
          {
            state->pMax[idx] = pVal;
            if (pVal >= state->alpha)
            {
              state->sepSets[col_node * p * state->maxCondSize +
                             row_node * state->maxCondSize] = next;
              state->adj[idx] =
                  state->adj[col_node * p + row_node] = 0;
              break;
            }
          }
        }
      }
    }
  }
}

void testRowLNTriangluarCPU(GPUState *state, int row_node, int col_node, std::shared_ptr<arma::mat> correlation, int level)
{
  int row_count = state->adj_compact[row_node * state->p + state->p - 1];
  int levelSize = level + 2;
  if (col_node < row_node &&
      row_node < state->p &&
      row_count > col_node && // col_node not available
      row_count >= level)
  {

    auto actual_col_node = state->adj_compact[row_node * state->p + col_node]; // get actual id
    size_t row_neighbours = state->adj_compact[row_node * state->p + state->p - 1] -
                            1; // get number of neighbors && exclude col_node
    size_t row_test_count = binomialCoeff(row_neighbours, level);
    int sepset_nodes[level];

    for (size_t test_index = 0; test_index < row_test_count;
         test_index++)
    {
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
      arma::dmat Submat(levelSize, levelSize, arma::fill::eye);
      //for (int i = 0; i < levelSize; ++i) {
      // set diagonal
      //  Submat(i,i) = 1;
      //}
      Submat(0, 1) = Submat(1, 0) = state->cor[row_node * state->p + actual_col_node];

      for (int j = 2; j < levelSize; ++j)
      {
        // set correlations of X
        Submat(0, j) = Submat(j, 0) =
            state->cor[row_node * state->p + sepset_nodes[j - 2]];
        // set correlations of Y
        Submat(1, j) = Submat(j, 1) =
            state->cor[actual_col_node * state->p + sepset_nodes[j - 2]];
      }
      for (int i = 2; i < levelSize; ++i)
      {
        for (int j = i + 1; j < levelSize; ++j)
        {
          Submat(i, j) = Submat(j, i) =
              state->cor[sepset_nodes[i - 2] * state->p + sepset_nodes[j - 2]];
        }
      }

      arma::mat SubmatPInv = arma::pinv(Submat);
      double r = -SubmatPInv(0, 1) / sqrt(SubmatPInv(0, 0) * SubmatPInv(1, 1));

      double pVal = calcPValue(r, state->observations);

      // Check pVal in regards to alpha and delete edge + save sepset + save pMax (Needs compare and swap to lock against other threads)
      if (pVal >= state->alpha)
      {
        if (row_node < actual_col_node)
        {
          state->adj[state->p * row_node + actual_col_node] = 0.f;
          state->adj[state->p * actual_col_node + row_node] = 0.f;
          state->pMax[state->p * row_node + actual_col_node] = pVal;
          for (int j = 0; j < level; ++j)
          {
            state->sepSets[row_node * state->p * state->maxCondSize +
                           actual_col_node * state->maxCondSize + j] = sepset_nodes[j];
          }
        }
        else
        {

          state->adj[state->p * row_node + actual_col_node] = 0.f;
          state->adj[state->p * actual_col_node + row_node] = 0.f;
          state->pMax[state->p * actual_col_node + row_node] = pVal;
          for (int j = 0; j < level; ++j)
          {
            state->sepSets[actual_col_node * state->p * state->maxCondSize +
                           row_node * state->maxCondSize + j] = sepset_nodes[j];
          }
        }
      }
    }
  }
}

TestResult cpuIndTest(int level, GPUState *state, SplitTaskQueue *cpuQueue)
{
  auto start = std::chrono::system_clock::now();

  auto row_count = cpuQueue->size_approx();

  auto corrMat = std::make_shared<arma::mat>(state->cor, state->p, state->p, false, true);
#pragma omp parallel for if (row_count > MULTITHREAD_THRESHHOLD)
  for (int _j = 0; _j < row_count; _j++)
  {
    SplitTask curTask;
    if (cpuQueue->try_dequeue(curTask))
    {
      auto row_node = curTask.row;
      for (int col_node = 0; col_node < row_node; col_node++)
      {
        switch (level)
        {
        case 0:
          testRowL0TriangluarCPU(state, row_node, col_node);
          break;
        case 1:
          testRowL1TriangluarCPU(state, row_node, col_node);
          break;
        default:
          testRowLNTriangluarCPU(state, row_node, col_node, corrMat, level);
          break;
        }
      }
    }
  }
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::system_clock::now() - start)
                      .count();
  return {static_cast<uint64_t>(duration), 0};
};