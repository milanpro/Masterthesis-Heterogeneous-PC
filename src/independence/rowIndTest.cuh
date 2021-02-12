#include "../util/State.cuh"

__device__ void testRowL0TriangluarGPU(MMGPUState state, int row_node, int col_node);

__device__ void testRowL1TriangluarGPU(MMGPUState state, int row_node, int col_node);

void testRowL0TriangluarCPU(MMGPUState *state, int row_node, int col_node);

void testRowL1TriangluarCPU(MMGPUState *state, int row_node, int col_node);