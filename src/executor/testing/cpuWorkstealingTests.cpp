#include "cpuWorkstealingTests.hpp"
#include "../independence/cpuInd.hpp"

void testEdgeWorkstealingL1(MMState *state, int row_node, int col_node, int actual_col_node, std::shared_ptr<EdgeQueue> eQueue, int row_count, std::atomic<int> &edges_done)
{
  int p = (int)state->p;

  int subIndex;
  for (int next = row_count - 1; next >= 0; next--)
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
        edges_done++;
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
  state->node_status[row_node * p + actual_col_node] = true;
  edges_done++;
}

void testEdgeWorkstealingLN(MMState *state, int row_node, int col_node, int actual_col_node, std::shared_ptr<EdgeQueue> eQueue, int row_count, std::atomic<int> &edges_done, int level)
{
  int p = (int)state->p;
  int row_neighbours = row_count - 1; // get number of neighbours && exclude col_node
  size_t row_test_count = binomialCoeff(row_neighbours, level);
  int sepset_nodes[level];
  int lvlSize = level + 2;
  for (int test_index = row_test_count - 1; test_index >= 0; test_index--)
  {

    ithCombination(sepset_nodes, test_index, level,
                   row_neighbours);

    // Fill sepset_nodes array with actual ids
    for (int ind = 0; ind < level; ++ind)
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
      edges_done++;
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
  edges_done++;
}