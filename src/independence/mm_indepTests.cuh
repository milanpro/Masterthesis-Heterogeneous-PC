#pragma once
#include "../util/indepUtil.hpp"
#include <vector>

__global__ void MMtestL0Triangle(MMGPUState state, int edgesPerGPU,
                                 int deviceId);

__global__ void MMtestL1Triangle(MMGPUState state, int edgesPerGPU,
                                 int deviceId);

__global__ void compact(MMGPUState state, int deviceId, int gpusUsed);

__host__ __device__ double mm_calcPValue(double r, int sampleSize);

__host__ __device__ double mm_pValL1(double x1, double x2, double x3,
                                     int sampleSize);
