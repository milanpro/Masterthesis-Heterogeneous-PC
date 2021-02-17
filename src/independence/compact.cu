#include "compact.cuh"
#include "../util/cuda_util.cuh"

__global__ void compactCU(MMState state, int deviceId, int numberOfGPUs) {
  int row_id = (blockIdx.x * numberOfGPUs) + deviceId;
  if (row_id > state.p){
    return;
  }
  const int chunk = (state.p + blockDim.x - 1) / blockDim.x;
  int thid = 0;
  int tmp = 0;
  int stepSize = 0;
  extern __shared__ int matrix_row[];
  // copy a matrix row into shared memory
  for (int cnt = 0; cnt < chunk; cnt++) {
    thid = threadIdx.x + blockDim.x * cnt;
    if (thid < state.p) {
      matrix_row[thid] = state.adj[row_id * state.p + thid];
    }
  }

  __syncthreads();
  for (int s = 0; s < chunk; ++s) {
    thid = threadIdx.x + blockDim.x * s;
    stepSize = ((state.p - s * blockDim.x) / blockDim.x) > 0
                   ? blockDim.x
                   : (state.p - s * blockDim.x);
    for (int step = 1; step < stepSize; step = step * 2) {
      if (thid < state.p) {
        if (threadIdx.x < step) {
          tmp = matrix_row[thid];
        } else if (threadIdx.x >= step) {
          tmp = matrix_row[thid] + matrix_row[thid - step];
        }
      }
      __syncthreads();
      if (thid < state.p) {
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
  const int row_size = matrix_row[state.p - 1];

  for (int s = 0; s < chunk; ++s) {
    thid = threadIdx.x + blockDim.x * s;
    if (thid < state.p && thid > 0) {
      if (matrix_row[thid] != matrix_row[thid - 1]) {
        state.adj_compact[row_id * state.p + matrix_row[thid] - 1] = thid;
      }
      if (thid >= row_size && thid != state.p - 1) {
        state.adj_compact[row_id * state.p + thid] = 0;
      }
      if (thid == state.p - 1) {
        atomicMax(state.max_adj, matrix_row[state.p - 1]);
        state.adj_compact[row_id * state.p + state.p - 1] =
            matrix_row[state.p - 1];
      }
    }
  }

  if (threadIdx.x == 0 && state.adj[row_id * state.p] == 1) {
    state.adj_compact[row_id * state.p] = 0;
  }
}

void callCompact(MMState *state, int deviceId, int numberOfGPUs,
                 int d_rowCount) {
  cudaSetDevice(deviceId);
  compactCU<<<d_rowCount, 1024, state->p * sizeof(int)>>>(*state, deviceId,
                                                          numberOfGPUs);
  cudaDeviceSynchronize();
  checkLastCudaError("Compacting failed");
}
