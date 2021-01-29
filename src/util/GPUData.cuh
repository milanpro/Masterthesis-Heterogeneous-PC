#ifndef R_PACKAGE_SRC_UTIL_GPUDATA_CUH_
#define R_PACKAGE_SRC_UTIL_GPUDATA_CUH_
#include "./cudaUtil.cuh"
#include <stdint.h>
#include <vector>

struct GPUData {
  bool _blocks;
  int _level;
  int *adjCopy;
  // mappings stores the mapping information of each seperation set block in
  // row of the main block
  int *mappings;
  // Sepadjs stores the adjacency information of each seperation set block in
  // row of the main block and row of main blocks columns
  int *sepAdjs;
  // Sepblocks stores the correlation information of the seperation set blocks
  // for first 2*level entries it stores the blocks for row/column to each sep
  // set block alternating, after 2*level blocks the blocks contain the data
  // of correleation between sepset blocks
  double *sepBlocks;

  GPUData(int blockSize, int level, bool blocks = true);

  void destroy();
};

struct SepData {
private:
  int _level;

public:
  int *sepAdjs;
  double *sepBlocks;
  int *mappings;

  explicit SepData(int blockSize, int level);

  ~SepData();

  SepData(const SepData &d) = delete;
};
#endif // R_PACKAGE_SRC_UTIL_GPUDATA_CUH_
