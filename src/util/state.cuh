#pragma once
#include "./cuda_util.cuh"
#include <stdint.h>
#include<vector>
#include <cuda/atomic>

/**
Saves every important value/structure needed for the indepTests.

@param pMax Partial correlation values. Gets updated with every additional level
of the tests.
@param adj Adjacency matrix. It is used to reduce the problem, in case we have
knowledge of the structure.
@param cor Correlation matrix.
@param sepSet Separation sets to determine which nodes are separating two
others. (Data structure maybe change.)
@param observations Number of observations.
@param p number of variables.

*/

struct MMState {
  double *pMax;
  int *adj;
  int *adj_compact;
  double *cor;
  int *sepSets;
  uint64_t p;
  int observations;
  double alpha;
  int maxCondSize;
  int *max_adj;
  int *lock;
  int maxLevel;
  cuda::atomic<bool> *node_status;

  MMState(uint64_t p, int observations, double alpha, int maxLevel, int mainDeviceId);

  void adviceReadonlyCor(std::vector<int> gpuList);

  void memAdvise(std::vector<int> gpuList);

  void prefetchRows(int startRow, int rowCount, int deviceId);

  void destroy();
};