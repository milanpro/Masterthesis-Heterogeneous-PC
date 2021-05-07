#include "cpuRowTests.hpp"
#include "../independence/cpuInd.hpp"
#include "cpuUtil.hpp"

void testRowL0Triangluar(MMState *state, int row_node, int col_node, std::shared_ptr<EdgeQueue> eQueue)
{
  auto idx = state->p * row_node + col_node;
  if (col_node < row_node && state->adj[idx])
  {
    auto inv_idx = state->p * col_node + row_node;
    double pVal = CPU::calcPValue(state->cor[idx], state->observations);
    if (pVal >= state->alpha)
    {
#if MIGRATE_EDGES
      enqueueEdgeDeletion(eQueue, col_node, row_node, pVal, {});
#else
      deleteEdgeLevel0(state, col_node, row_node, pVal);
#endif
    }
  }
}

void testRowL1(MMState *state, int row_node, int col_node, std::shared_ptr<EdgeQueue> eQueue)
{
  int p = (int)state->p;
  int row_count = state->adj_compact[row_node * p + p - 1];
  if (row_count > col_node && // col_node not available
      row_count >= 1)
  {
    auto actual_col_node = state->adj_compact[row_node * p + col_node];
    int subIndex;
    for (int next = 0; next < row_count; next++)
    {
      if (next != col_node)
      {
        subIndex = state->adj_compact[row_node * p + next];
        double pVal = CPU::pValL1(
            state->cor[row_node * p + actual_col_node],
            state->cor[row_node * p + subIndex],
            state->cor[actual_col_node * p + subIndex], state->observations);
        // Check pVal in regards to alpha and delete edge + save sepset + save pMax (Needs compare and swap to lock against other threads)
        if (pVal >= state->alpha)
        {
#if MIGRATE_EDGES
          enqueueEdgeDeletion(eQueue, actual_col_node, row_node, pVal, {state->adj_compact[actual_col_node * p + subIndex]});
#else
          deleteEdge(state, 1, actual_col_node, row_node, pVal, {state->adj_compact[actual_col_node * p + subIndex]});
#endif
          return;
        }
        if (state->adj[row_node * state->p + actual_col_node] == 0)
          return;
      }
    }
  }
}

template <int lvlSize, int kLvlSizeSmall>
void testRowLN(MMState *state, int row_node, int col_node, std::shared_ptr<EdgeQueue> eQueue)
{
  int p = (int)state->p;
  int row_count = state->adj_compact[row_node * p + p - 1];

  if (row_count > col_node && // col_node not available
      row_count >= kLvlSizeSmall)
  {

    auto actual_col_node = state->adj_compact[row_node * p + col_node]; // get actual id
    int row_neighbours = row_count - 1;                                 // get number of neighbors && exclude col_node
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
              state->adj_compact[row_node * p + sepset_nodes[ind]];
        }
        else
        {
          sepset_nodes[ind] =
              state->adj_compact[row_node * p + sepset_nodes[ind] - 1];
        }
      }
      arma::dmat Submat(lvlSize, lvlSize, arma::fill::eye);

      Submat(0, 1) = Submat(1, 0) = state->cor[row_node * p + actual_col_node];

      for (int j = 2; j < lvlSize; ++j)
      {
        // set correlations of X
        Submat(0, j) = Submat(j, 0) =
            state->cor[row_node * p + sepset_nodes[j - 2]];
        // set correlations of Y
        Submat(1, j) = Submat(j, 1) =
            state->cor[actual_col_node * p + sepset_nodes[j - 2]];
      }
      for (int i = 2; i < lvlSize; ++i)
      {
        for (int j = i + 1; j < lvlSize; ++j)
        {
          Submat(i, j) = Submat(j, i) =
              state->cor[sepset_nodes[i - 2] * p + sepset_nodes[j - 2]];
        }
      }

      double pVal = CPU::pValLN(Submat, state->observations);

      // Check pVal in regards to alpha and delete edge + save sepset + save pMax (Needs compare and swap to lock against other threads)
      if (pVal >= state->alpha)
      {
        std::vector<int> sepSet = {sepset_nodes[0]};
        for (int j = 1; j < kLvlSizeSmall; ++j)
        {
          sepSet.push_back(sepset_nodes[j]);
        }
#if MIGRATE_EDGES
        enqueueEdgeDeletion(eQueue, actual_col_node, row_node, pVal, sepSet);
#else
        deleteEdge(state, kLvlSizeSmall, actual_col_node, row_node, pVal, sepSet);
#endif
        return;
      }
      if (state->adj[row_node * state->p + actual_col_node] == 0)
        return;
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
