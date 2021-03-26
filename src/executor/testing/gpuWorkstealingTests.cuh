#include "../../util/state.cuh"

__global__ void testRowWorkstealingL1(MMState state, int *rows, int start_row, int max_row_count);

template <int lvlSize, int kLvlSizeSmall>
__global__ void testRowWorkstealingLN(MMState state, int *rows, int start_row, int max_row_count);