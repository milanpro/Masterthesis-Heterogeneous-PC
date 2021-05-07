#include "cpuUtil.hpp"

void deleteEdgeLevel0(MMState *state, int col_node, int row_node, double pMax)
{
  state->adj[state->p * row_node + col_node] = 0;
  state->adj[state->p * col_node + row_node] = 0;
  if (row_node < col_node)
  {
    state->pMax[state->p * row_node + col_node] = pMax;

    state->sepSets[row_node * state->p * state->maxCondSize +
                   col_node * state->maxCondSize] = -2;
  }
  else
  {
    state->pMax[state->p * col_node + row_node] = pMax;

    state->sepSets[col_node * state->p * state->maxCondSize +
                   row_node * state->maxCondSize] = -2;
  }
}

void deleteEdge(MMState *state, int level, int col_node, int row_node, double pMax, std::vector<int> sepSet)
{
  state->adj[state->p * row_node + col_node] = 0;
  state->adj[state->p * col_node + row_node] = 0;

  if (row_node < col_node)
  {
    state->pMax[state->p * row_node + col_node] = pMax;

    for (int j = 0; j < level; ++j)
    {
      state->sepSets[row_node * state->p * state->maxCondSize +
                     col_node * state->maxCondSize + j] = sepSet[j];
    }
  }
  else
  {
    state->pMax[state->p * col_node + row_node] = pMax;

    for (int j = 0; j < level; ++j)
    {
      state->sepSets[col_node * state->p * state->maxCondSize +
                     row_node * state->maxCondSize + j] = sepSet[j];
    }
  }
}

void enqueueEdgeDeletion(std::shared_ptr<EdgeQueue> deletedEdges, int col_node, int row_node, double pMax, std::vector<int> sepSet)
{
  DeletedEdge result;
  result.col = col_node;
  result.row = row_node;
  result.pMax = pMax;
  result.sepSet = sepSet;
  deletedEdges->enqueue(result);
}
