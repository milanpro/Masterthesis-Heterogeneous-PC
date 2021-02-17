#include "gpu.cuh"
#include "../util/cuda_util.cuh"
#include "../util/state.cuh"
#include <iostream>
#include <chrono>
#include <stdio.h>

namespace GPU {
  __device__ double calcPValue(double r, int sampleSize) {
    r = isnan(r) ? 0.0 : fmin(0.9999999, fabs(r));
    double absz = sqrt(sampleSize - 3.0) * 0.5 * log1p(2.0 * r / (1.0 - r));
    return 2.0 * (1.0 - normcdf(absz));
  }
  
  __device__ double pValL1(double x1, double x2, double x3, int sampleSize) {
    // with edge i, j given k values are:
    // x1: edge i, j
    // x2: edge i, k
    // x3: edge j, k
    double r = (x1 - x2 * x3) / sqrt((1.0 - x3 * x3) * (1.0 - x2 * x2));
    return calcPValue(r, sampleSize);
  }
  
  
  __global__ void testRowL0(MMState state, int row, int row_count) {
    size_t id = (blockIdx.x * blockDim.x) + threadIdx.x;
    size_t row_node = static_cast<size_t>(sqrt(2 * id + pow(row - 0.5, 2)) + 0.5);
    size_t col_node = id - ((row_node * (row_node - 1) / 2) - (row * (row - 1) / 2));
    size_t max_row = row + row_count;
    if (col_node < state.p && row_node < max_row) {
      int idx = state.p * row_node + col_node;
      if (col_node < row_node && state.adj[idx]) {
        int inv_idx = state.p * col_node + row_node;
        double pVal = calcPValue(state.cor[idx], state.observations);
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
  }
  
  __global__ void testRowL1(MMState state, int row, int row_count) {
    size_t id = blockIdx.x;
    size_t row_node = static_cast<size_t>(sqrt(2 * id + pow(row - 0.5, 2)) + 0.5);
    size_t col_node = id - ((row_node * (row_node - 1) / 2) - (row * (row - 1) / 2));
    size_t max_row = row + row_count;
    size_t idx = row_node * state.p + col_node;
    if (col_node < row_node && row_node < max_row && state.adj_compact[idx] != 0) {
        extern __shared__ double pVals[];
        int inv_idx = col_node * state.p + row_node;
        for (size_t offset = threadIdx.x; offset < state.p; offset += blockDim.x) {
          pVals[threadIdx.x] = -1;
          if (row_node != offset && col_node != offset) {
            // check if edge still exists and if edge
            // is available to separation set
            if (state.adj_compact[row_node * state.p + offset] != 0 ||
                state.adj_compact[col_node * state.p + offset] != 0) {
              pVals[threadIdx.x] = pValL1(
                  state.cor[idx],
                  state.cor[row_node * state.p + offset],
                  state.cor[col_node * state.p + offset], state.observations);
              //printf("Row: %llu Col: %llu otherCol: %llu pVal: %f\n",row_node, col_node, offset, pVals[threadIdx.x]);
            }
          }
          __syncthreads();
          if (threadIdx.x == 0) {
            for (size_t i = 0; i < blockDim.x; ++i) {
              if (pVals[i] > state.pMax[inv_idx]) {
                state.pMax[inv_idx] = pVals[i];
                if (pVals[i] >= state.alpha) {
                  // CAREFUL CURRENTLY LIMIT sepsets to Size 1
                  // as we only go through level 0 and 1
                  state.sepSets[col_node * state.p * state.maxCondSize +
                                row_node * state.maxCondSize] = offset + i;
                  state.adj[idx] =
                      state.adj[inv_idx] = 0;
                  break;
                }
              }
            }
          }
          __syncthreads();
          if (state.adj[idx] == 0)
            break;
        }
    }
  }
  
  template <int lvlSize, int kLvlSizeSmall>
  __global__ void testRowLN(MMState state, int row, int row_count) {
    size_t id = row + blockIdx.x;
    size_t col_node = blockIdx.y;
    if (id < state.p &&
      col_node < state.p &&
      blockIdx.x < row_count &&
      state.adj_compact[id * state.p + state.p - 1] > col_node && // col_node not available
      state.adj_compact[id * state.p + state.p - 1] >= kLvlSizeSmall) {
        double Submat[lvlSize][lvlSize];
        double SubmatPInv[lvlSize][lvlSize];
        int sepset_nodes[kLvlSizeSmall];
        // pseudo-inverse parameter
        double v[lvlSize][lvlSize];
        double w[lvlSize], rv1[lvlSize];
        double res1[lvlSize][lvlSize];
        // Determine sepsets to work on
        col_node = state.adj_compact[id * state.p + col_node]; // get actual id
        size_t row_neighbours = state.adj_compact[id * state.p + state.p - 1] -
                             1; // get number of neighbors && exclude col_node
        size_t row_test_count = binomialCoeff(row_neighbours, kLvlSizeSmall);
        for (size_t test_index = threadIdx.x; test_index < row_test_count;
             test_index += blockDim.x) {
          ithCombination(sepset_nodes, test_index, kLvlSizeSmall,
                              row_neighbours);
          for (int ind = 0; ind < kLvlSizeSmall; ++ind) {
            if (sepset_nodes[ind] - 1 >= blockIdx.y) {
              sepset_nodes[ind] =
                  state.adj_compact[id * state.p + sepset_nodes[ind]];
            } else {
              sepset_nodes[ind] =
                  state.adj_compact[id * state.p + sepset_nodes[ind] - 1];
            }
          }
          for (int i = 0; i < lvlSize; ++i) {
            // set diagonal
            Submat[i][i] = 1;
          }
          Submat[0][1] = Submat[1][0] = state.cor[id * state.p + col_node];
          for (int j = 2; j < lvlSize; ++j) {
            // set correlations of X
            Submat[0][j] = Submat[j][0] =
                state.cor[id * state.p + sepset_nodes[j - 2]];
            // set correlations of Y
            Submat[1][j] = Submat[j][1] =
                state.cor[col_node * state.p + sepset_nodes[j - 2]];
          }
          for (int i = 2; i < lvlSize; ++i) {
            for (int j = i + 1; j < lvlSize; ++j) {
              Submat[i][j] = Submat[j][i] =
                  state.cor[sepset_nodes[i - 2] * state.p + sepset_nodes[j - 2]];
            }
          }
          pseudoinverse<lvlSize>(Submat, SubmatPInv, v, rv1, w, res1);
          double r = -SubmatPInv[0][1] / sqrt(SubmatPInv[0][0] * SubmatPInv[1][1]);
          double pVal = calcPValue(r, state.observations);
          if (pVal >= state.alpha) {
            if (id < col_node) {
              if (atomicCAS(&state.lock[(state.p * id) + col_node], 0, 1) == 0) {
                state.adj[state.p * id + col_node] = 0.f;
                state.adj[state.p * col_node + id] = 0.f;
                state.pMax[state.p * id + col_node] = pVal;
                for (int j = 0; j < kLvlSizeSmall; ++j) {
                  state.sepSets[id * state.p * state.maxCondSize +
                                col_node * state.maxCondSize + j] = sepset_nodes[j];
                }
              }
            } else {
              if (atomicCAS(&state.lock[(state.p * col_node) + id], 0, 1) == 0) {
                state.adj[state.p * id + col_node] = 0.f;
                state.adj[state.p * col_node + id] = 0.f;
                state.pMax[state.p * col_node + id] = pVal;
                for (int j = 0; j < kLvlSizeSmall; ++j) {
                  state.sepSets[col_node * state.p * state.maxCondSize +
                                id * state.maxCondSize + j] = sepset_nodes[j];
                }
              }
            }
          }
        }
    }
  }
  
  TestResult executeLevel(int level, MMState *state, SplitTaskQueue *gpuQueue, int maxEdgeCount, int numberOfGPUs) {
    auto start = std::chrono::system_clock::now();
    int numthreads = NUMTHREADS;
    dim3 block, grid;
    if (level >= 1) {
      block = dim3(min(numthreads, (int)state->p));
    }
  
    int row_count = gpuQueue->size_approx();
    
    #pragma omp parallel for
    for(int i = 0; i < row_count; i++){
      SplitTask curTask;
      if(gpuQueue->try_dequeue(curTask)) {
        int deviceId = i % numberOfGPUs; // Maybe split in parts instead of RR
        cudaSetDevice(deviceId);
        switch (level) {
          case 0:
            block = dim3(numthreads);
            grid = dim3((curTask.rowCount * (curTask.row + curTask.rowCount) + numthreads) / numthreads);
            testRowL0<<<grid, block>>>(*state, curTask.row, curTask.rowCount);
            break;
          case 1:
            grid = dim3(curTask.rowCount * (curTask.row + curTask.rowCount));
            testRowL1<<<grid, block, sizeof(double) * numthreads>>>(*state, curTask.row, curTask.rowCount);
            break;
          case 2:
          grid = dim3(curTask.rowCount * (curTask.row + curTask.rowCount), state->p);
            testRowLN<4,2><<<grid, block>>>(*state, curTask.row, curTask.rowCount);
            break;
          case 3:
          grid = dim3(curTask.rowCount * (curTask.row + curTask.rowCount), state->p);
            testRowLN<5,3><<<grid, block>>>(*state, curTask.row, curTask.rowCount);
            break;
        }
        std::string msg =
        "L " + std::to_string(level) + " Kernel execution failed";
        cudaDeviceSynchronize();
        checkLastCudaError(msg.c_str());
      }
    }
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::system_clock::now() - start)
                        .count();
  
    return {static_cast<uint64_t>(duration), (state->p * (state->p - 1L)) / 2};
  }
}