#ifndef R_PACKAGE_SRC_UTIL_STATE_CUH_
#define R_PACKAGE_SRC_UTIL_STATE_CUH_
#include "./cudaUtil.cuh"
#include <stdint.h>

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
struct State {
  double *pMax;
  int *adj;
  double *cor;
  int *sepSets;
  uint64_t p;
  int observations;
  double alpha;
  int maxCondSize;

  State(uint64_t p, int observations, double alpha, int maxCondSize);

  State(const State &state);

  ~State();
};

struct GPUState {
  double *pMax;
  int *adj;
  double *cor;
  int *sepSets;
  uint64_t p;
  int observations;
  double alpha;
  int maxCondSize;
  int *lock;
  int *rowMapping;
  int *colMapping;

  GPUState(uint64_t p, int observations, double alpha, int maxCondSize);

  void destroy();
};

struct MMGPUState {
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
  MMGPUState(uint64_t p, int observations, double alpha, int maxCondSize);

  void destroy();
};
#endif // R_PACKAGE_SRC_UTIL_STATE_CUH_
