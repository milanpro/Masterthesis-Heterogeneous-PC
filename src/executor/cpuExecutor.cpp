#include "cpuExecutor.hpp"
#include "armadillo"
#include "boost/math/distributions/normal.hpp"
#include "boost/math/special_functions/log1p.hpp"
#include <chrono>
#include <iostream>

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

  void testRowL0Triangluar(MMState *state, int row_node, int col_node)
  {
    auto idx = state->p * row_node + col_node;
    if (col_node < row_node && state->adj[idx])
    {
      auto inv_idx = state->p * col_node + row_node;
      double pVal = calcPValue(state->cor[idx], state->observations);
      state->pMax[inv_idx] = pVal;
      if (state->pMax[inv_idx] >= state->alpha)
      {
        state->adj[idx] = 0;
        state->adj[inv_idx] = 0;
        state->sepSets[(col_node * state->maxCondSize * state->p) +
                       (row_node * state->maxCondSize)] = -2;
      }
    }
  }

  void testRowL1(MMState *state, int row_node, int col_node)
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
            state->adj[p * row_node + actual_col_node] = 0;
            state->adj[p * actual_col_node + row_node] = 0;
            if (row_node < actual_col_node)
            {
              state->pMax[p * row_node + actual_col_node] = pVal;
              state->sepSets[row_node * p * state->maxCondSize +
                             actual_col_node * state->maxCondSize] =
                  state->adj_compact[row_node * p + subIndex];
            }
            else
            {
              state->pMax[p * actual_col_node + row_node] = pVal;
              state->sepSets[actual_col_node * p * state->maxCondSize +
                             row_node * state->maxCondSize] =
                  state->adj_compact[actual_col_node * p + subIndex];
            }
            break;
          }
        }
      }
    }
  }

  template <int lvlSize, int kLvlSizeSmall>
  void testRowLN(MMState *state, int row_node, int col_node)
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
          if (row_node < actual_col_node)
          {
            state->adj[state->p * row_node + actual_col_node] = 0;
            state->adj[state->p * actual_col_node + row_node] = 0;
            state->pMax[state->p * row_node + actual_col_node] = pVal;
            for (int j = 0; j < kLvlSizeSmall; ++j)
            {
              state->sepSets[row_node * state->p * state->maxCondSize +
                             actual_col_node * state->maxCondSize + j] = sepset_nodes[j];
            }
          }
          else
          {

            state->adj[state->p * row_node + actual_col_node] = 0;
            state->adj[state->p * actual_col_node + row_node] = 0;
            state->pMax[state->p * actual_col_node + row_node] = pVal;
            for (int j = 0; j < kLvlSizeSmall; ++j)
            {
              state->sepSets[actual_col_node * state->p * state->maxCondSize +
                             row_node * state->maxCondSize + j] = sepset_nodes[j];
            }
          }
          break;
        }
      }
    }
  }

  void testEdge(int level, MMState *state, int row_node, int col_node)
  {
    switch (level)
    {
    case 0:
      testRowL0Triangluar(state, row_node, col_node);
      break;
    case 1:
      testRowL1(state, row_node, col_node);
      break;
    case 2:
      testRowLN<4, 2>(state, row_node, col_node);
      break;
    case 3:
      testRowLN<5, 3>(state, row_node, col_node);
      break;
    }
  }
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
      CPU::testEdge(level, state, tasks[i].row, col_node);
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