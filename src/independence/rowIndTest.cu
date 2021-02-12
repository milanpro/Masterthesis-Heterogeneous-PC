#include "../util/State.cuh"
#include "mm_indepTests.cuh"

__host__ __device__ void testRowL0Triangluar(MMGPUState state, int row_node, int col_node) {
  int idx = state.p * row_node + col_node;
  if (col_node < row_node && state.adj[idx]) {
    int inv_idx = state.p * col_node + row_node;
    double pVal = mm_calcPValue(state.cor[idx], state.observations);
    state.pMax[inv_idx] = pVal;
    if (state.pMax[inv_idx] >= state.alpha) {
      state.adj[idx] = 0;
      state.adj[inv_idx] = 0;
      state.adj_compact[idx] = 0;
      state.adj_compact[inv_idx] = 0;
      state.sepSets[(col_node * state.maxCondSize * state.p) +
        (row_node * state.maxCondSize)] = -2;
    }
  }
}

__host__ __device__ void testRowTriangluar(int level, MMGPUState state, int row_node, int col_node) {
  switch (level) {
    case 0:
      testRowL0Triangluar(state, row_node, col_node);
  }
}
