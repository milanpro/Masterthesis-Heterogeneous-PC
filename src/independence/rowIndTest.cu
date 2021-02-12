#include "../util/State.cuh"
#include "mm_indepTests.cuh"

__device__ void testRowL0TriangluarGPU(MMGPUState state, int row_node, int col_node) {
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

void testRowL0TriangluarCPU(MMGPUState *state, int row_node, int col_node) {
  int idx = state->p * row_node + col_node;
  if (col_node < row_node && state->adj[idx]) {
    int inv_idx = state->p * col_node + row_node;
    double pVal = mm_calcPValue(state->cor[idx], state->observations);
    state->pMax[inv_idx] = pVal;
    if (state->pMax[inv_idx] >= state->alpha) {
      state->adj[idx] = 0;
      state->adj[inv_idx] = 0;
      state->adj_compact[idx] = 0;
      state->adj_compact[inv_idx] = 0;
      state->sepSets[(col_node * state->maxCondSize * state->p) +
        (row_node * state->maxCondSize)] = -2;
    }
  }
}

__device__ void testRowL1TriangluarGPU(MMGPUState state, int row_node, int col_node) {
  int idx = state.p * row_node + col_node;
    if (state.adj_compact[idx] != 0) {
      extern __shared__ double pVals[];
      for (size_t offset = threadIdx.x; offset < state.p; offset += blockDim.x) {
        pVals[threadIdx.x] = -1;
        if (row_node != offset && col_node != offset) {
          if (state.adj_compact[row_node * state.p + offset] != 0 ||
              state.adj_compact[col_node * state.p + offset] != 0) {
            pVals[threadIdx.x] = mm_pValL1(
                state.cor[idx],
                state.cor[row_node * state.p + offset],
                state.cor[col_node * state.p + offset], state.observations);
          }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
          for (size_t i = 0; i < blockDim.x; ++i) {
            if (pVals[i] > state.pMax[col_node * state.p + row_node]) {
              state.pMax[col_node * state.p + row_node] = pVals[i];
              if (pVals[i] >= state.alpha) {
                state.sepSets[col_node * state.p * state.maxCondSize +
                              row_node * state.maxCondSize] = offset + i; // safe sepset
                state.adj[idx] =
                    state.adj[col_node * state.p + row_node] = 0; // delete edge
                break; // get out of loop
              }
            }
          }
        }
        __syncthreads();
        if (state.adj[row_node * state.p + col_node] == 0)
          break;
      }
    }
}

void testRowL1TriangluarCPU(MMGPUState *state, int row_node, int col_node) {
  int p = state->p;
  int idx = p * row_node + col_node;
  if (col_node < row_node && state->adj[idx]) {
    for (int next = 0; next < p; next++) {
      if (row_node != next && col_node != next) {
        if (state->adj[row_node * p + next] != 0 || state->adj[col_node * p + next] != 0) {
          int pVal = mm_pValL1(
                            state->cor[idx],
                state->cor[row_node * p + next],
                state->cor[col_node * p + next], state->observations);
          if (pVal > state->pMax[idx]) {
            state->pMax[idx] = pVal;
            if (pVal >= state->alpha) {
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
