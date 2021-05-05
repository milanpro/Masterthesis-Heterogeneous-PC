#include "./state.cuh"
#include <string>
#include <iostream>

int printSepsets(MMState *state, bool showSepsets, bool verbose)
{
  int nrEdges = 0;
  for (int i = 0; i < state->p; i++)
  {
    for (int j = i + 1; j < state->p; j++)
    {
      if (!state->adj[i * state->p + j])
      {
        if (showSepsets)
        {
          std::string sepset_string = "";
          for (int k = 0; k < state->maxCondSize; k++)
          {
            int current_sepset_node =
                state->sepSets[(i * state->maxCondSize * state->p) +
                               (j * state->maxCondSize) + k];
            if (current_sepset_node == -2)
            {
              std::cout << "Separation from " << i << " to " << j << "\n";
              break;
            }
            else if (current_sepset_node == -1)
            {
              break;
            }
            else
            {
              sepset_string.append(std::to_string(current_sepset_node));
              sepset_string.append(" ");
            }
          }
          if (sepset_string != "")
          {
            std::cout << "Separation from " << i << " to " << j << " via "
                      << sepset_string << "\n";
          }
        }
      }
      else
      {
        //std::cout << "Edge from " << i << " to " << j << std::endl;
        nrEdges++;
      }
    }
  }
  if (verbose)
  {
    std::cout << "Total number of edges: " << nrEdges << std::endl;
  }
  return nrEdges;
}