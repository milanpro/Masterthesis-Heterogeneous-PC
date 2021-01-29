#include "GPUData.cuh"

GPUData::GPUData(int blockSize, int level, bool blocks)
    : _blocks(blocks), _level(level) {
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&adjCopy),
                             sizeof(int) * blockSize * blockSize));
  // Note mapping is no longer available if blocks is false
  if (_blocks) {
    // Only needed if block-based approach is used
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&sepBlocks),
                               sizeof(double) * blockSize * blockSize *
                                   ((((_level + 1) * (_level + 2)) / 2) - 1)));
    checkCudaErrors(
        cudaMalloc(reinterpret_cast<void **>(&sepAdjs),
                   sizeof(int) * blockSize * blockSize * (_level * 2)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&mappings),
                               sizeof(int) * blockSize * _level));
  }
}

void GPUData::destroy() {
  checkCudaErrors(cudaFree(adjCopy));
  if (_blocks) {
    checkCudaErrors(cudaFree(sepBlocks));
    checkCudaErrors(cudaFree(sepAdjs));
    checkCudaErrors(cudaFree(mappings));
  }
}

SepData::SepData(int blockSize, int level) : _level(level) {

  sepBlocks = new double[blockSize * blockSize *
                         ((((_level + 1) * (_level + 2)) / 2) - 1)];
  cudaHostRegister(sepBlocks,
                   (uint64_t)sizeof(double) * blockSize * blockSize *
                       ((((_level + 1) * (_level + 2)) / 2) - 1),
                   0);

  sepAdjs = new int[blockSize * blockSize * _level * 2];
  cudaHostRegister(
      sepAdjs, (uint64_t)sizeof(int) * blockSize * blockSize * _level * 2, 0);

  mappings = new int[blockSize * _level];
  cudaHostRegister(mappings, (uint64_t)sizeof(int) * blockSize * _level, 0);
}

SepData::~SepData() {
  cudaHostUnregister(sepBlocks);
  cudaHostUnregister(sepAdjs);
  cudaHostUnregister(mappings);
  delete[] sepBlocks;
  delete[] sepAdjs;
  delete[] mappings;
}
