#include "mm_indepTests.cuh"
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <math.h>

__host__ __device__ double mm_calcPValue(double r, int sampleSize) {
  r = isnan(r) ? 0.0 : fmin(0.9999999, fabs(r));
  double absz = sqrt(sampleSize - 3.0) * 0.5 * log1p(2.0 * r / (1.0 - r));
  return 2.0 * (1.0 - normcdf(absz));
}

__host__ __device__ double mm_pValL1(double x1, double x2, double x3,
                                     int sampleSize) {
  // with edge i, j given k values are:
  // x1: edge i, j
  // x2: edge i, k
  // x3: edge j, k
  double r = (x1 - x2 * x3) / sqrt((1.0 - x3 * x3) * (1.0 - x2 * x2));
  return mm_calcPValue(r, sampleSize);
}

__global__ void MMtestL0Triangle(MMGPUState state, int edgesPerGPU,
                                 int deviceId) {
  int id = deviceId * edgesPerGPU + ((blockIdx.x * blockDim.x) + threadIdx.x);
  if (id < (state.p * (state.p - 1) / 2) &&
      id < ((deviceId + 1) * edgesPerGPU)) {
    // build contingency table
    int row_node = static_cast<int>(sqrt(0.25 + 2 * id) + 0.5);
    int col_node = id - (row_node * (row_node - 1) / 2);

    if (state.adj[state.p * row_node + col_node]) {
      double pVal = mm_calcPValue(state.cor[state.p * row_node + col_node],
                                  state.observations);
      state.pMax[state.p * col_node + row_node] = pVal;
      if (state.pMax[state.p * col_node + row_node] >= state.alpha) {
        state.adj[state.p * row_node + col_node] = 0;
        state.adj[state.p * col_node + row_node] = 0;
        state.adj_compact[state.p * row_node + col_node] = 0;
        state.adj_compact[state.p * col_node + row_node] = 0;
        state.sepSets[(col_node * state.maxCondSize * state.p) +
                      (row_node * state.maxCondSize)] = -2;
      }
    }
  }
}

__global__ void MMtestL1Triangle(MMGPUState state, int edgesPerGPU,
                                 int deviceId) {
  // We keep input and output adjacencies separate to keep order correct
  int id = deviceId * edgesPerGPU + blockIdx.x;
  if (id < (state.p * (state.p - 1) / 2) &&
      id < ((deviceId + 1) * edgesPerGPU)) {
    int row_node = static_cast<int>(sqrt(0.25 + 2 * id) + 0.5);
    int col_node = id - (row_node * (row_node - 1) / 2);
    if (state.adj_compact[row_node * state.p + col_node] != 0) {
      extern __shared__ double pVals[];
      for (int offset = threadIdx.x; offset < state.p; offset += blockDim.x) {
        pVals[threadIdx.x] = -1;
        if (row_node != offset && col_node != offset) {
          // check if edge still exists and if edge
          // is available to separation set
          if (state.adj_compact[row_node * state.p + offset] != 0 ||
              state.adj_compact[col_node * state.p + offset] != 0) {
            pVals[threadIdx.x] = mm_pValL1(
                state.cor[row_node * state.p + col_node],
                state.cor[row_node * state.p + offset],
                state.cor[col_node * state.p + offset], state.observations);
          }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
          for (int i = 0; i < blockDim.x; ++i) {
            if (pVals[i] > state.pMax[col_node * state.p + row_node]) {
              state.pMax[col_node * state.p + row_node] = pVals[i];
              if (pVals[i] >= state.alpha) {
                // CAREFUL CURRENTLY LIMIT sepsets to Size 1
                // as we only go through level 0 and 1
                state.sepSets[col_node * state.p * state.maxCondSize +
                              row_node * state.maxCondSize] = offset + i;
                state.adj[row_node * state.p + col_node] =
                    state.adj[col_node * state.p + row_node] = 0;
                break;
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
}

__global__ void compact(MMGPUState state, int deviceId, int gpusUsed) {
  int row_id = (blockIdx.x * gpusUsed) + deviceId;
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
