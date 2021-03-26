#include "../../util/state.cuh"

__global__ void testRowL0(MMState state, int row, int row_count);

__global__ void testRowL1(MMState state, int *rows, int start_row, int max_row_count);

template <int lvlSize, int kLvlSizeSmall>
__global__ void testRowLN(MMState state, int *rows, int start_row, int max_row_count);
