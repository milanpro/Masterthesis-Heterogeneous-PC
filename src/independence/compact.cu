#include "compact.cuh"
#include "../util/cuda_util.cuh"

__global__ void compactCUTiled(MMState state, int idx, int numberOfGPUs) {
  uint64_t row_id = idx * gridDim.x + blockIdx.x;
  if (row_id >= state.p) {
    return;
  }
  uint64_t compactSize = min(state.p, (uint64_t)COMPACTSHARED);
  int tiles = (state.p + (compactSize -1)) / compactSize;
  const int chunk = (compactSize + blockDim.x - 1) / blockDim.x;
  int thid = 0;
  int tmp = 0;
  int stepSize = 0;
  int old_row_size = 0;
  extern __shared__ int matrix_row[];
  for (int t = 0; t < tiles; ++t) {
    for (int cnt = 0; cnt < chunk; cnt++) {
      thid = threadIdx.x + blockDim.x * cnt;
      matrix_row[thid] = 0;
    }
    __syncthreads();
    // copy a matrix row into shared memory
    for (int cnt = 0; cnt < chunk; cnt++) {
      thid = threadIdx.x + blockDim.x * cnt;
      if (thid + t * compactSize < state.p) {
        matrix_row[thid] = state.adj[row_id * state.p + t * compactSize + thid];
      }
    }

    __syncthreads();
    for (int s = 0; s < chunk; ++s) {
      thid = threadIdx.x + blockDim.x * s;
      stepSize = ((compactSize - s * blockDim.x) / blockDim.x) > 0
                     ? blockDim.x
                     : (compactSize - s * blockDim.x);
      for (int step = 1; step < stepSize; step = step * 2) {
        if (thid < compactSize) {
          if (threadIdx.x < step) {
            tmp = matrix_row[thid];
          } else if (threadIdx.x >= step) {
            tmp = matrix_row[thid] + matrix_row[thid - step];
          }
        }
        __syncthreads();
        if (thid < compactSize) {
          matrix_row[thid] = tmp;
        }
        __syncthreads();
      }
      if (thid == (blockDim.x * (s + 1) - 1) && s != (chunk - 1)) {
        matrix_row[thid + 1] = matrix_row[thid + 1] + matrix_row[thid];
      }
      __syncthreads();
    }

    // Compacting Step
    const int row_size = matrix_row[compactSize - 1];
    if (threadIdx.x == 0) {
      if (t > 0 && matrix_row[thid] == 1) {
        state.adj_compact[row_id * state.p + old_row_size] = t * compactSize;
      }
    }
    __syncthreads();
    for (int s = 0; s < chunk; ++s) {
      thid = threadIdx.x + blockDim.x * s;
      if (thid + t * compactSize < state.p && thid > 0) {
        if (matrix_row[thid] != matrix_row[thid - 1]) {
          state.adj_compact[row_id * state.p + old_row_size +
            matrix_row[thid] - 1] = (t * compactSize) + thid;
        }
      }
    }
    old_row_size += row_size;
  }
  __syncthreads();
  if (threadIdx.x == 0 && state.adj[row_id * state.p] == 1) {
    state.adj_compact[row_id * state.p] = 0;
  }
  if (threadIdx.x == 0) {
    atomicMax_system(state.max_adj, old_row_size);
    state.adj_compact[row_id * state.p + state.p - 1] = old_row_size;
  }
  for (int j = old_row_size + threadIdx.x; j < state.p -1; j+= blockDim.x) {
    if (j < state.p - 1) {
      state.adj_compact[row_id * state.p + j] = 0;
    }
  }
}

void callCompact(MMState *state, int deviceId, int idx, int numberOfGPUs,
                 int d_rowCount) {
  cudaSetDevice(deviceId);
  state->max_adj[0] = 0;
  compactCUTiled<<<d_rowCount, min(COMPACTSHARED, 1024), COMPACTSHARED * sizeof(int)>>>(
                    *state, idx, numberOfGPUs);
  cudaDeviceSynchronize();
  checkLastCudaError("Compacting failed");
}
