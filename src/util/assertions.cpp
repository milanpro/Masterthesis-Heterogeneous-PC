#include "assertions.hpp"
#include <iostream>
#include <cassert>

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

void assertNodeStatus(MMState *state, int level)
{
  if (level > 0)
  {
    bool edge_not_done = level % 2 == 0;
    for (int i = 0; i < state->p; i++)
    {
      for (int j = 0; j < state->p; j++)
      {
        bool status = state->node_status[i * state->p + j];
        if (state->adj[i * state->p + j] == 1 && state->adj[i * state->p + state->p - 1] >= level - 1 && i != j)
        {
          if (edge_not_done != status)
          {
            std::cout << "row " << i << " col " << j << std::endl;
          }
          assert(edge_not_done == status);
        }
      }
    }
  }
}
