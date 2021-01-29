#include "../util/cudaUtil.cuh"
#include "../util/matrixPrint.cuh"
#include "../util/memInfo.cuh"
#include "mm_indepTests.cuh"
#include "mm_test.cuh"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

__device__ double MMcalcPValueLN(double r, int sampleSize);
__device__ double MMcalcPValueLN(double r, int sampleSize) {
  r = isnan(r) ? 0.0 : fmin(0.9999999, fabs(r));
  double absz = sqrt(sampleSize - 3.0) * 0.5 * log1p(2.0 * r / (1.0 - r));
  return 2.0 * (1.0 - normcdf(absz));
}

__device__ double MMPYTHAG(double a, double b);
__device__ double MMPYTHAG(double a, double b) {
  double aAbs = fabs(a);
  double bAbs = fabs(b);
  double c;
  double result;
  if (aAbs > bAbs) {
    c = bAbs / aAbs;
    result = aAbs * sqrt(1.0 + c * c);
  } else if (bAbs > 0.0) {
    c = aAbs / bAbs;
    result = bAbs * sqrt(1.0 + c * c);
  } else {
    result = 0.0;
  }
  return (result);
}

__device__ int MMbinomialCoeff(int n, int k);
__device__ int MMbinomialCoeff(int n, int k) {
  if (n < k) {
    return 0;
  }
  int res = 1;
  if (k > n - k)
    k = n - k;
  for (int i = 0; i < k; ++i) {
    res *= (n - i);
    res /= (i + 1);
  }
  return res;
}

__device__ void MMith_combination(int *sepset_nodes, int comb_number, int level,
                                  int neighbours);
__device__ void MMith_combination(int *sepset_nodes, int comb_number, int level,
                                  int neighbours) {
  int sum = 0;
  int tmp = 0;
  for (int i = 0; i < level; i++) {
    sepset_nodes[i] = 0;
    if (i > 0) {
      sepset_nodes[i] = sepset_nodes[i - 1];
    }
    while (sum <= comb_number) {
      sepset_nodes[i]++;
      tmp = MMbinomialCoeff(neighbours - sepset_nodes[i], level - (i + 1));
      sum = sum + tmp;
    }
    sum = sum - tmp;
  }
}

template <int lvlSize>
__device__ void MMpseudoinverse(double Submat[][lvlSize],
                                double SubmatPInv[][lvlSize],
                                double v[][lvlSize], double *rv1, double *w,
                                double res[][lvlSize]);
template <int lvlSize>
__device__ void MMpseudoinverse(double Submat[][lvlSize],
                                double SubmatPInv[][lvlSize],
                                double v[][lvlSize], double *rv1, double *w,
                                double res[][lvlSize]) {
  int flag, its, i, j, jj, k, l, nm;
  double c, f, h, s, x, y, z;
  double anorm = 0.0, g = 0.0, scale = 0.0;
  /* Householder reduction to bidiagonal form */
  for (i = 0; i < lvlSize; i++) {
    /* left-hand reduction */
    l = i + 1;
    rv1[i] = scale * g;
    g = s = scale = 0.0;
    if (i < lvlSize) {
      for (k = i; k < lvlSize; k++)
        scale += fabs(Submat[k][i]);
      if (scale) {
        for (k = i; k < lvlSize; k++) {
          Submat[k][i] = (Submat[k][i] / scale);
          s += (Submat[k][i] * Submat[k][i]);
        }
        f = Submat[i][i];
        g = -SIGN(sqrt(s), f);
        h = f * g - s;
        Submat[i][i] = f - g;
        if (i != lvlSize - 1) {
          for (j = l; j < lvlSize; j++) {
            for (s = 0.0, k = i; k < lvlSize; k++)
              s += (Submat[k][i] * Submat[k][j]);
            f = s / h;
            for (k = i; k < lvlSize; k++)
              Submat[k][j] += (f * Submat[k][i]);
          }
        }
        for (k = i; k < lvlSize; k++)
          Submat[k][i] = (Submat[k][i] * scale);
      }
    }
    w[i] = scale * g;

    /* right-hand reduction */
    g = s = scale = 0.0;
    if (i < lvlSize && i != lvlSize - 1) {
      for (k = l; k < lvlSize; k++)
        scale += fabs(Submat[i][k]);
      if (scale) {
        for (k = l; k < lvlSize; k++) {
          Submat[i][k] = (Submat[i][k] / scale);
          s += (Submat[i][k] * Submat[i][k]);
        }
        f = Submat[i][l];
        g = -SIGN(sqrt(s), f);
        h = f * g - s;
        Submat[i][l] = f - g;
        for (k = l; k < lvlSize; k++)
          rv1[k] = Submat[i][k] / h;
        if (i != lvlSize - 1) {
          for (j = l; j < lvlSize; j++) {
            for (s = 0.0, k = l; k < lvlSize; k++)
              s += (Submat[j][k] * Submat[i][k]);
            for (k = l; k < lvlSize; k++)
              Submat[j][k] += (s * rv1[k]);
          }
        }
        for (k = l; k < lvlSize; k++)
          Submat[i][k] = Submat[i][k] * scale;
      }
    }
    anorm = MAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
  }

  /* accumulate the right-hand transformation */
  for (i = lvlSize - 1; i >= 0; i--) {
    if (i < lvlSize - 1) {
      if (g) {
        for (j = l; j < lvlSize; j++)
          v[j][i] = (Submat[i][j] / Submat[i][l]) / g;
        /* double division to avoid underflow */
        for (j = l; j < lvlSize; j++) {
          for (s = 0.0, k = l; k < lvlSize; k++)
            s += (Submat[i][k] * v[k][j]);
          for (k = l; k < lvlSize; k++)
            v[k][j] += (s * v[k][i]);
        }
      }
      for (j = l; j < lvlSize; j++)
        v[i][j] = v[j][i] = 0.0;
    }
    v[i][i] = 1.0;
    g = rv1[i];
    l = i;
  }

  /* accumulate the left-hand transformation */
  for (i = lvlSize - 1; i >= 0; i--) {
    l = i + 1;
    g = w[i];
    if (i < lvlSize - 1) {
      for (j = l; j < lvlSize; j++)
        Submat[i][j] = 0.0;
    }
    if (g) {
      g = 1.0 / g;
      if (i != lvlSize - 1) {
        for (j = l; j < lvlSize; j++) {
          for (s = 0.0, k = l; k < lvlSize; k++)
            s += (Submat[k][i] * Submat[k][j]);
          f = (s / Submat[i][i]) * g;
          for (k = i; k < lvlSize; k++)
            Submat[k][j] += (f * Submat[k][i]);
        }
      }
      for (j = i; j < lvlSize; j++)
        Submat[j][i] = (Submat[j][i] * g);
    } else {
      for (j = i; j < lvlSize; j++)
        Submat[j][i] = 0.0;
    }
    ++Submat[i][i];
  }

  /* diagonalize the bidiagonal form */
  for (k = lvlSize - 1; k >= 0; k--) { /* loop over singular values */
    for (its = 0; its < 30; its++) {   /* loop over allowed iterations */
      flag = 1;
      for (l = k; l >= 0; l--) { /* test for splitting */
        nm = l - 1;
        if (fabs(rv1[l]) + anorm == anorm) {
          flag = 0;
          break;
        }
        if (fabs(w[nm]) + anorm == anorm)
          break;
      }
      if (flag) {
        c = 0.0;
        s = 1.0;
        for (i = l; i <= k; i++) {
          f = s * rv1[i];
          if (fabs(f) + anorm != anorm) {
            g = w[i];
            h = MMPYTHAG(f, g);
            w[i] = h;
            h = 1.0 / h;
            c = g * h;
            s = (-f * h);
            for (j = 0; j < lvlSize; j++) {
              y = Submat[j][nm];
              z = Submat[j][i];
              Submat[j][nm] = (y * c + z * s);
              Submat[j][i] = (z * c - y * s);
            }
          }
        }
      }
      z = w[k];
      if (l == k) {    /* convergence */
        if (z < 0.0) { /* make singular value nonnegative */
          w[k] = (-z);
          for (j = 0; j < lvlSize; j++)
            v[j][k] = (-v[j][k]);
        }
        break;
      }
      if (its >= 30) {
        printf("Not converged\n");
      }

      /* shift from bottom 2 x 2 minor */
      x = w[l];
      nm = k - 1;
      y = w[nm];
      g = rv1[nm];
      h = rv1[k];
      f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
      g = MMPYTHAG(f, 1.0);
      f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;

      /* next QR transformation */
      c = s = 1.0;
      for (j = l; j <= nm; j++) {
        i = j + 1;
        g = rv1[i];
        y = w[i];
        h = s * g;
        g = c * g;
        z = MMPYTHAG(f, h);
        rv1[j] = z;
        c = f / z;
        s = h / z;
        f = x * c + g * s;
        g = g * c - x * s;
        h = y * s;
        y = y * c;
        for (jj = 0; jj < lvlSize; jj++) {
          x = v[jj][j];
          z = v[jj][i];
          v[jj][j] = (x * c + z * s);
          v[jj][i] = (z * c - x * s);
        }
        z = MMPYTHAG(f, h);
        w[j] = z;
        if (z) {
          z = 1.0 / z;
          c = f * z;
          s = h * z;
        }
        f = (c * g) + (s * y);
        x = (c * y) - (s * g);
        for (jj = 0; jj < lvlSize; jj++) {
          y = Submat[jj][j];
          z = Submat[jj][i];
          Submat[jj][j] = (y * c + z * s);
          Submat[jj][i] = (z * c - y * s);
        }
      }
      rv1[l] = 0.0;
      rv1[k] = f;
      w[k] = x;
    }
  }

  // start compute inverse matrix

  for (int rowNumber = 0; rowNumber < lvlSize; rowNumber++) {
    for (int colNumber = 0; colNumber < lvlSize; colNumber++) {
      res[rowNumber][colNumber] = v[rowNumber][colNumber] / w[colNumber];
    }
  }

  for (int rowNumber = 0; rowNumber < lvlSize; rowNumber++) {
    for (int colNumber = 0; colNumber < lvlSize; colNumber++) {
      SubmatPInv[rowNumber][colNumber] = 0;
      for (int thirdIndex = 0; thirdIndex < lvlSize; thirdIndex++) {
        SubmatPInv[rowNumber][colNumber] =
            SubmatPInv[rowNumber][colNumber] +
            res[rowNumber][thirdIndex] * Submat[colNumber][thirdIndex];
      }
    }
  }
}

template <int lvlSize>
__global__ void MMtestLNKernel(MMGPUState state, int edgesPerGPU, int deviceId);

template <int lvlSize>
__global__ void MMtestLNKernel(MMGPUState state, int edgesPerGPU,
                               int deviceId) {
  // start with n (bx) x n_max (by) blocks
  int id = deviceId * edgesPerGPU + blockIdx.x;
  int col_node = blockIdx.y;
  if (id < state.p &&
      col_node < state.p && // id or col_node larger than variables
      id < ((deviceId + 1) *
            edgesPerGPU) && // id larger than range of current GPU
      state.adj_compact[id * state.p + state.p - 1] >
          col_node && // col_node not available
      state.adj_compact[id * state.p + state.p - 1] >=
          lvlSize - 2) { // not enough neighbors
    double Submat[lvlSize][lvlSize];
    double SubmatPInv[lvlSize][lvlSize];
    int sepset_nodes[lvlSize - 2];
    // pseudo-inverse parameter
    double v[lvlSize][lvlSize];
    double w[lvlSize], rv1[lvlSize];
    double res1[lvlSize][lvlSize];
    // Determine sepsets to work on
    col_node = state.adj_compact[id * state.p + col_node]; // get actual id
    int row_neighbours = state.adj_compact[id * state.p + state.p - 1] -
                         1; // get number of neighbors && exclude col_node
    int row_test_count = MMbinomialCoeff(row_neighbours, lvlSize - 2);
    for (int test_index = threadIdx.x; test_index < row_test_count;
         test_index += blockDim.x) {
      MMith_combination(sepset_nodes, test_index, lvlSize - 2, row_neighbours);
      for (int ind = 0; ind < lvlSize - 2; ++ind) {
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
      MMpseudoinverse<lvlSize>(Submat, SubmatPInv, v, rv1, w, res1);
      double r = -SubmatPInv[0][1] / sqrt(SubmatPInv[0][0] * SubmatPInv[1][1]);
      double pVal = MMcalcPValueLN(r, state.observations);
      
      if (pVal >= state.alpha) {
        if (id < col_node) {
          if (atomicCAS(&state.lock[(state.p * id) + col_node], 0, 1) == 0) {
            state.adj[state.p * id + col_node] = 0.f;
            state.adj[state.p * col_node + id] = 0.f;
            state.pMax[state.p * id + col_node] = pVal;
            for (int j = 0; j < lvlSize - 2; ++j) {
              state.sepSets[id * state.p * state.maxCondSize +
                            col_node * state.maxCondSize + j] = sepset_nodes[j];
            }
          }
        } else {
          if (atomicCAS(&state.lock[(state.p * col_node) + id], 0, 1) == 0) {
            state.adj[state.p * id + col_node] = 0.f;
            state.adj[state.p * col_node + id] = 0.f;
            state.pMax[state.p * col_node + id] = pVal;
            for (int j = 0; j < lvlSize - 2; ++j) {
              state.sepSets[col_node * state.p * state.maxCondSize +
                            id * state.maxCondSize + j] = sepset_nodes[j];
            }
          }
        }
      }
    }
  }
}

TestResult MMtestL0(MMGPUState *state, int blockSize, int gpusUsed) {
  auto start = std::chrono::system_clock::now();
  std::vector<std::thread> threads;
  int edgesPerGPU =
      (state->p * (state->p - 1L) / 2 + (gpusUsed - 1)) / gpusUsed;
  int numthreads = min(edgesPerGPU, NUMTHREADS);
  dim3 block(numthreads), grid((edgesPerGPU + numthreads - 1) / numthreads);

  for(int i = 0; i < gpusUsed; i++){
    threads.push_back(std::thread([&, i]() 
    {
      cudaSetDevice(i);
      MMtestL0Triangle<<<grid, block>>>(*state, edgesPerGPU, i);
      cudaDeviceSynchronize();
      getLastCudaError("L0 Kernel execution failed");
    }));
  }

  std::for_each(threads.begin(), threads.end(), [](std::thread &t) 
  {
      t.join();
  });

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::system_clock::now() - start)
                      .count();

  std::unordered_map<std::string, uint64_t> subTimes(
      {{"Copy", 0}, {"Test", 0}});
  return {static_cast<uint64_t>(duration), (state->p * (state->p - 1L)) / 2,
          subTimes};
}

TestResult MMtestL1(MMGPUState *state, int blockSize, int gpusUsed) {
  auto start = std::chrono::system_clock::now();
  std::vector<std::thread> threads;
  int edgesPerGPU =
      (state->p * (state->p - 1L) / 2 + (gpusUsed - 1)) / gpusUsed;
  int numthreads = min(state->p, (uint64_t)NUMTHREADS);
  dim3 block(numthreads), grid(edgesPerGPU);

  for(int i = 0; i < gpusUsed; i++){
    threads.push_back(std::thread([&, i]() 
    {
      cudaSetDevice(i);
      MMtestL1Triangle<<<grid, block, sizeof(double) * numthreads>>>(*state, edgesPerGPU, i);
      cudaDeviceSynchronize();
      getLastCudaError("L1 Kernel execution failed");
    }));
  }

  std::for_each(threads.begin(), threads.end(), [](std::thread &t) 
  {
      t.join();
  });


  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::system_clock::now() - start)
                      .count();

  std::unordered_map<std::string, uint64_t> subTimes(
      {{"Copy", 0}, {"Test", 0}});
  return {static_cast<uint64_t>(duration), 0, subTimes};
}

TestResult MMtestLN(MMGPUState *state, int blockSize, int gpusUsed, int lvl) {
  auto start_compact_time = std::chrono::system_clock::now();
  std::vector<std::thread> threads;
  int device_row_count = state->p / gpusUsed;
  int max_additional_row_index = state->p % gpusUsed;

  for(int i = 0; i < gpusUsed; i++){
    threads.push_back(std::thread([&, i]() 
    {
      cudaSetDevice(i);
      int actual_device_row_count = device_row_count + (i < max_additional_row_index);
      compact<<<actual_device_row_count, 1024, state->p * sizeof(int)>>>(*state, i, gpusUsed);
      cudaDeviceSynchronize();
    }));
  }

  std::for_each(threads.begin(), threads.end(), [](std::thread &t) 
  {
      t.join();
  });

  auto compact_time = std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::system_clock::now() - start_compact_time)
                          .count();

  auto start = std::chrono::system_clock::now();
  std::vector<std::thread> threads2;
  int edgesPerGPU = (state->p + gpusUsed - 1) / gpusUsed;
  int numthreads = min(state->p, (uint64_t)NUMTHREADS);
  dim3 block(numthreads), grid(edgesPerGPU, state->max_adj[0]);

  for (int i = 0; i < gpusUsed; ++i) {
    threads2.push_back(std::thread([&, i]() 
    {
      cudaSetDevice(i);
      switch (lvl) {
        case 2:
          MMtestLNKernel<4><<<grid, block>>>(*state, edgesPerGPU, i);
          break;
        case 3:
          MMtestLNKernel<5><<<grid, block>>>(*state, edgesPerGPU, i);
          break;
          /*
        case 4:
          MMtestLNKernel<6><<<grid, block>>>(*state, edgesPerGPU, i);
          break;
        case 5:
          MMtestLNKernel<7><<<grid, block>>>(*state, edgesPerGPU, i);
          break;
        case 6:
          MMtestLNKernel<8><<<grid, block>>>(*state, edgesPerGPU, i);
          break;
        case 7:
          MMtestLNKernel<9><<<grid, block>>>(*state, edgesPerGPU, i);
          break;
        case 8:
          MMtestLNKernel<10><<<grid, block>>>(*state, edgesPerGPU, i);
          break;
        case 9:
          MMtestLNKernel<11><<<grid, block>>>(*state, edgesPerGPU, i);
          break;
        case 10:
          MMtestLNKernel<12><<<grid, block>>>(*state, edgesPerGPU, i);
          break;
        case 11:
          MMtestLNKernel<13><<<grid, block>>>(*state, edgesPerGPU, i);
          break;
        case 12:
          MMtestLNKernel<14><<<grid, block>>>(*state, edgesPerGPU, i);
          break;
        case 13:
          MMtestLNKernel<15><<<grid, block>>>(*state, edgesPerGPU, i);
          break;
        case 14:
          MMtestLNKernel<16><<<grid, block>>>(*state, edgesPerGPU, i);
          break;
          */
        }
      cudaDeviceSynchronize();
    }));
  }

  std::for_each(threads2.begin(), threads2.end(), [](std::thread &t) 
  {
      t.join();
  });

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::system_clock::now() - start)
                      .count();

  std::unordered_map<std::string, uint64_t> subTimes(
      {{"Copy", 0}, {"Test", 0}, {"Compact_Time", compact_time}});

  return {static_cast<uint64_t>(duration), 0, subTimes};
}
