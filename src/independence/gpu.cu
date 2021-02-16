#include "../util/indep_util.hpp"
#include "../util/cuda_util.cuh"
#include "../util/state.cuh"
#include <iostream>
#include <chrono>
#include <stdio.h>

__global__ void testRowL0(GPUState state, int row_node) {
  int col_node = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (col_node < state.p && row_node < state.p) {
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

__global__ void testRowL1(GPUState state, int row_node) {
  int col_node = blockIdx.x;
  if (col_node < row_node && row_node < state.p) {
    int idx = state.p * row_node + col_node;
    if (state.adj_compact[idx] != 0) {
      extern __shared__ double pVals[];
      for (size_t offset = threadIdx.x; offset < row_node; offset += blockDim.x) {
        pVals[threadIdx.x] = -1;
        if (row_node != offset && col_node != offset) {
          if (state.adj_compact[row_node * state.p + offset] != 0 ||
              state.adj_compact[col_node * state.p + offset] != 0) {
            pVals[threadIdx.x] = pValL1(
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
}

template <int lvlSize, int kLvlSizeSmall>
__global__ void testRowLN(GPUState state, int row_node) {
  if (blockIdx.x == 0)
    printf("Starting level %d kernel\n", kLvlSizeSmall);
  int col_node = blockIdx.x;
  if (col_node < state.p) {
    if (blockIdx.x == 0)
      printf("Checking adj comp\n");
    int row_count = state.adj_compact[row_node * state.p + state.p - 1];
    if (blockIdx.x == 0)
      printf("Row count is %d\n", row_count);
    if (row_node < state.p &&
      row_count > col_node && // col_node not available
      row_count >= kLvlSizeSmall) { // not enough neighbors

    if (blockIdx.x == 0)
            printf("Init erveything\n");
      
    double Submat[lvlSize][lvlSize];
    double SubmatPInv[lvlSize][lvlSize];
    int sepset_nodes[kLvlSizeSmall];
    // pseudo-inverse parameter
    double v[lvlSize][lvlSize];
    double w[lvlSize], rv1[lvlSize];
    double res1[lvlSize][lvlSize];

    // Determine sepsets to work on
    col_node = state.adj_compact[row_node * state.p + col_node]; // get actual id
    size_t row_neighbours = state.adj_compact[row_node * state.p + state.p - 1] -
                         1; // get number of neighbors && exclude col_node
    size_t row_test_count = binomialCoeff(row_neighbours, kLvlSizeSmall);


    if (blockIdx.x == 0)
      printf("Init phase done\n");
    for (size_t test_index = threadIdx.x; test_index < row_test_count;
         test_index += blockDim.x) {
      // Generate new sepset permutation
      ithCombination(sepset_nodes, test_index, kLvlSizeSmall,
                          row_neighbours);
      // Fill sepset_nodes array with actual ids
      for (int ind = 0; ind < kLvlSizeSmall; ++ind) {
        if (sepset_nodes[ind] - 1 >= blockIdx.y) {
          sepset_nodes[ind] =
              state.adj_compact[row_node * state.p + sepset_nodes[ind]];
        } else {
          sepset_nodes[ind] =
              state.adj_compact[row_node * state.p + sepset_nodes[ind] - 1];
        }
      }
      if (blockIdx.x == 0)
        printf("Created sepset\n");
      // Start filling Submat with cor values of sepset nodes
      for (int i = 0; i < lvlSize; ++i) {
        // set diagonal
        Submat[i][i] = 1;
      }
      Submat[0][1] = Submat[1][0] = state.cor[row_node * state.p + col_node];
      for (int j = 2; j < lvlSize; ++j) {
        // set correlations of X
        Submat[0][j] = Submat[j][0] =
            state.cor[row_node * state.p + sepset_nodes[j - 2]];
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

      if (blockIdx.x == 0)
        printf("Created Submat\n");
      // Submat is filled and ready for the CI test
      pseudoinverse<lvlSize>(Submat, SubmatPInv, v, rv1, w, res1);
      double r = -SubmatPInv[0][1] / sqrt(SubmatPInv[0][0] * SubmatPInv[1][1]);
      double pVal = calcPValue(r, state.observations);

      if (blockIdx.x == 0)
        printf("Calculated pVal\n");
      // Check pVal in regards to alpha and delete edge + save sepset + save pMax (Needs compare and swap to lock against other threads)
      if (pVal >= state.alpha) {
        if (row_node < col_node) {
          if (atomicCAS(&state.lock[(state.p * row_node) + col_node], 0, 1) == 0) {
            state.adj[state.p * row_node + col_node] = 0.f;
            state.adj[state.p * col_node + row_node] = 0.f;
            state.pMax[state.p * row_node + col_node] = pVal;
            for (int j = 0; j < kLvlSizeSmall; ++j) {
              state.sepSets[row_node * state.p * state.maxCondSize +
                            col_node * state.maxCondSize + j] = sepset_nodes[j];
            }
          }
        } else {
          if (atomicCAS(&state.lock[(state.p * col_node) + row_node], 0, 1) == 0) {
            state.adj[state.p * row_node + col_node] = 0.f;
            state.adj[state.p * col_node + row_node] = 0.f;
            state.pMax[state.p * col_node + row_node] = pVal;
            for (int j = 0; j < kLvlSizeSmall; ++j) {
              state.sepSets[col_node * state.p * state.maxCondSize +
              row_node * state.maxCondSize + j] = sepset_nodes[j];
            }
          }
        }
      }
      if (blockIdx.x == 0)
        printf("Checked alpha and pVal\n");
    }
  }
  }
}

TestResult gpuIndTest(int level, GPUState *state, SplitTaskQueue *gpuQueue, int maxEdgeCount, int numberOfGPUs) {
  auto start = std::chrono::system_clock::now();
  int numthreads = min((int)state->p, NUMTHREADS);
  dim3 block(numthreads), grid((state->p + numthreads - 1) / numthreads);
  if (level >= 1) {
    numthreads = min((int)state->p, NUMTHREADS);
    grid = dim3(state->p);
  }

  int row_count = gpuQueue->size_approx();
  
  #pragma omp parallel for
  for(int i = 0; i < row_count; i++){
    SplitTask curTask;
    if(gpuQueue->try_dequeue(curTask)) {
      int deviceId = i % numberOfGPUs;
      cudaSetDevice(deviceId);
      switch (level) {
        case 0:
          testRowL0<<<grid, block>>>(*state, curTask.row);
          break;
        case 1:
          testRowL1<<<grid, block, sizeof(double) * numthreads>>>(*state, curTask.row);
          break;
        case 2:
          testRowLN<4,2><<<grid, block>>>(*state, curTask.row);
          break;
        case 3:
          testRowLN<5,3><<<grid, block>>>(*state, curTask.row);
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