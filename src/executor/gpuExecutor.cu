#include "gpuExecutor.cuh"
#include "../util/cuda_util.cuh"
#include <cmath>
#include <chrono>
#include <iostream>

namespace GPU
{

  __device__ double calcPValue(double r, int sampleSize)
  {
    r = isnan(r) ? 0.0 : fmin(0.9999999, fabs(r));
    double absz = sqrt(sampleSize - 3.0) * 0.5 * log1p(2.0 * r / (1.0 - r));
    return 2.0 * (1.0 - normcdf(absz));
  }

  __device__ double pValL1(double x1, double x2, double x3, int sampleSize)
  {
    // with edge i, j given k values are:
    // x1: edge i, j
    // x2: edge i, k
    // x3: edge j, k
    double r = (x1 - x2 * x3) / sqrt((1.0 - x3 * x3) * (1.0 - x2 * x2));
    return calcPValue(r, sampleSize);
  }

  __global__ void testRowL0(MMState state, int row, int row_count)
  {
    size_t id = (blockIdx.x * blockDim.x) + threadIdx.x;
    size_t row_node = static_cast<size_t>(sqrt(2 * id + pow(row - 0.5, 2)) + 0.5);
    size_t col_node = id - ((row_node * (row_node - 1) / 2) - (row * (row - 1) / 2));
    size_t max_row = row + row_count;
    if (col_node < state.p && row_node < max_row)
    {
      int idx = state.p * row_node + col_node;
      if (col_node < row_node && state.adj[idx])
      {
        int inv_idx = state.p * col_node + row_node;
        double pVal = calcPValue(state.cor[idx], state.observations);
        state.pMax[inv_idx] = pVal;
        if (state.pMax[inv_idx] >= state.alpha)
        {
          state.adj[idx] = 0;
          state.adj[inv_idx] = 0;
          state.sepSets[(col_node * state.maxCondSize * state.p) +
                        (row_node * state.maxCondSize)] = -2;
        }
      }
    }
  }

  __global__ void testRowL1(MMState state, int *rows, int start_row, int max_row_count)
  {
    if (start_row + blockIdx.x >= max_row_count) {
      return;
    }

    size_t row_node = rows[start_row + blockIdx.x];
    size_t row_neighbours = state.adj_compact[row_node * state.p + state.p - 1];
    //printf("Row %d Col %d Neigh %d\n", (int) row_node, (int) blockIdx.y, (int) row_neighbours);
    extern __shared__ double pVals[];
    if (row_neighbours > blockIdx.y && row_neighbours >= 1)
    {
      size_t col_node = state.adj_compact[row_node * state.p + blockIdx.y];
      if (col_node > state.p)
      {
        return;
      }

      size_t subIndex = 0;
      size_t row_neighbours = state.adj_compact[row_node * state.p + state.p - 1];
      for (size_t offset = threadIdx.x; offset < row_neighbours; offset += blockDim.x)
      {
        if (offset == blockIdx.y)
        {
          pVals[threadIdx.x] = -1;
        }
        else
        {
          subIndex = state.adj_compact[row_node * state.p + offset];
          pVals[threadIdx.x] = pValL1(
              state.cor[row_node * state.p + col_node],
              state.cor[row_node * state.p + subIndex],
              state.cor[col_node * state.p + subIndex], state.observations);
        }
        //printf("my pVal: %f for thread %hd \n", pVals[threadIdx.x], threadIdx.x);
        __syncthreads();
        if (threadIdx.x == 0)
        {
          for (size_t i = 0; i < blockDim.x && i < row_neighbours; ++i)
          {
            double pVal = pVals[i];
            if (offset + i < state.p && pVal >= state.alpha)
            {
              //printf("col %d pVal %f row %d actcol %d\n", (int)(offset + i), pVal, (int)row_node, (int)col_node);
              if (row_node < col_node)
              {
                if (atomicCAS_system(&state.lock[(state.p * row_node) + col_node], 0, 1) == 0)
                {
                  state.adj[row_node * state.p + col_node] = 0;
                  state.adj[col_node * state.p + row_node] = 0;
                  state.sepSets[row_node * state.p * state.maxCondSize +
                                col_node * state.maxCondSize] =
                      state.adj_compact[row_node * state.p + offset + i];
                  state.pMax[state.p * row_node + col_node] = pVal;
                }
              }
              else
              {
                if (atomicCAS_system(&state.lock[(state.p * col_node) + row_node], 0, 1) == 0)
                {
                  state.adj[row_node * state.p + col_node] = 0;
                  state.adj[col_node * state.p + row_node] = 0;
                  state.pMax[state.p * col_node + row_node] = pVal;
                  state.sepSets[col_node * state.p * state.maxCondSize +
                                row_node * state.maxCondSize] =
                      state.adj_compact[col_node * state.p + offset + i];
                }
              }
              break;
            }
          }
        }
        __syncthreads();
        if (state.adj[row_node * state.p + col_node] == 0)
          break;
      }
    }
  }

  template <int lvlSize, int kLvlSizeSmall>
  __global__ void testRowLN(MMState state, int *rows, int start_row, int max_row_count)
  {
    if (start_row + blockIdx.x >= max_row_count) {
      return;
    }

    size_t row_node = rows[start_row + blockIdx.x];
    size_t row_count = state.adj_compact[row_node * state.p + state.p - 1];
    if (row_count > blockIdx.y && // col_node available
        row_count >= kLvlSizeSmall)
    {
      double Submat[lvlSize][lvlSize];
      double SubmatPInv[lvlSize][lvlSize];
      int sepset_nodes[kLvlSizeSmall];
      // pseudo-inverse parameter
      double v[lvlSize][lvlSize];
      double w[lvlSize], rv1[lvlSize];
      double res1[lvlSize][lvlSize];
      // Determine sepsets to work on
      size_t col_node = state.adj_compact[row_node * state.p + blockIdx.y]; // get actual id
      int row_neighbours = row_count - 1;                                   // get number of neighbors && exclude col_node
      size_t row_test_count = binomialCoeff(row_neighbours, kLvlSizeSmall);
      for (size_t test_index = threadIdx.x; test_index < row_test_count;
           test_index += blockDim.x)
      {
        ithCombination(sepset_nodes, test_index, kLvlSizeSmall,
                       row_neighbours);
        for (int ind = 0; ind < kLvlSizeSmall; ++ind)
        {
          if (sepset_nodes[ind] - 1 >= blockIdx.y)
          {
            sepset_nodes[ind] =
                state.adj_compact[row_node * state.p + sepset_nodes[ind]];
          }
          else
          {
            sepset_nodes[ind] =
                state.adj_compact[row_node * state.p + sepset_nodes[ind] - 1];
          }
        }
        for (int i = 0; i < lvlSize; ++i)
        {
          // set diagonal
          Submat[i][i] = 1;
        }
        Submat[0][1] = Submat[1][0] = state.cor[row_node * state.p + col_node];
        for (int j = 2; j < lvlSize; ++j)
        {
          // set correlations of X
          Submat[0][j] = Submat[j][0] =
              state.cor[row_node * state.p + sepset_nodes[j - 2]];
          // set correlations of Y
          Submat[1][j] = Submat[j][1] =
              state.cor[col_node * state.p + sepset_nodes[j - 2]];
        }
        for (int i = 2; i < lvlSize; ++i)
        {
          for (int j = i + 1; j < lvlSize; ++j)
          {
            Submat[i][j] = Submat[j][i] =
                state.cor[sepset_nodes[i - 2] * state.p + sepset_nodes[j - 2]];
          }
        }
        pseudoinverse<lvlSize>(Submat, SubmatPInv, v, rv1, w, res1);
        double r = -SubmatPInv[0][1] / sqrt(SubmatPInv[0][0] * SubmatPInv[1][1]);
        double pVal = calcPValue(r, state.observations);
        if (pVal >= state.alpha)
        {
          if (row_node < col_node)
          {
            if (atomicCAS(&state.lock[(state.p * row_node) + col_node], 0, 1) == 0)
            {
              state.adj[state.p * row_node + col_node] = 0.f;
              state.adj[state.p * col_node + row_node] = 0.f;
              state.pMax[state.p * row_node + col_node] = pVal;
              for (int j = 0; j < kLvlSizeSmall; ++j)
              {
                state.sepSets[row_node * state.p * state.maxCondSize +
                              col_node * state.maxCondSize + j] = sepset_nodes[j];
              }
            }
          }
          else
          {
            if (atomicCAS(&state.lock[(state.p * col_node) + row_node], 0, 1) == 0)
            {
              state.adj[state.p * row_node + col_node] = 0.f;
              state.adj[state.p * col_node + row_node] = 0.f;
              state.pMax[state.p * col_node + row_node] = pVal;
              for (int j = 0; j < kLvlSizeSmall; ++j)
              {
                state.sepSets[col_node * state.p * state.maxCondSize +
                              row_node * state.maxCondSize + j] = sepset_nodes[j];
              }
            }
          }
        }
      }
    }
  }
}
TestResult GPUExecutor::executeLevel(int level, bool verbose)
{
  if (tasks.size() == 0) {
    return {0, 0};
  }

  int *rows;
  int row_count = 0;
  checkCudaErrors(cudaMallocManaged(&rows, (uint64_t)sizeof(int) * state->p));
  for (auto task : tasks) {
    for (auto i = task.row; i < task.row + task.rowCount; i++) {
      rows[row_count] = i;
      row_count++;
    }
  }

  auto start = std::chrono::system_clock::now();
  int numthreads = NUMTHREADS;
  dim3 block = dim3(numthreads);
  dim3 grid;
  int rowsPerGPU = (int)std::ceil((float)row_count / (float)numberOfGPUs);

  if (level == 0)
  {
    int max_rows = (int)std::ceil((float)rowsPerGPU * (float)state->p / (float)numthreads);
    grid = dim3(max_rows);
  }
  else
  {
    grid = dim3(rowsPerGPU, (int)state->p);
  }

#pragma omp parallel for num_threads(numberOfGPUs) if(numberOfGPUs > 1)
  for (int deviceId = 0; deviceId < numberOfGPUs; deviceId++)
  {
    int start_row = deviceId * rowsPerGPU;

    checkCudaErrors(cudaSetDevice(deviceId));
      switch (level)
      {
      case 0:
        GPU::testRowL0<<<grid, block>>>(*state, start_row, rowsPerGPU);
        break;
      case 1:
        GPU::testRowL1<<<grid, block, sizeof(double) * numthreads>>>(*state, rows, start_row, row_count);
        break;
      case 2:
        GPU::testRowLN<4, 2><<<grid, block>>>(*state, rows, start_row, row_count);
        break;
      case 3:
        GPU::testRowLN<5, 3><<<grid, block>>>(*state, rows, start_row, row_count);
        break;
      }
  }
  std::string msg = "L " + std::to_string(level) + " Kernel execution failed";
  cudaDeviceSynchronize();
  checkLastCudaError(msg.c_str());
  auto duration = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                            std::chrono::system_clock::now() - start)
                                            .count());
  if (verbose)
  {
    std::cout << "\tGPU is done. Time: " << (int)duration << " \u03BCs." << std::endl;
  }
  return TestResult{duration, (int)(state->p * (state->p - 1L)) / 2};
}
