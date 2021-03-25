#include "../../util/state.cuh"
#include "../cpuExecutor.hpp"

void testEdgeWorkstealingL1(MMState *state, int row_node, int col_node, int actual_col_node, std::shared_ptr<EdgeQueue> eQueue, int row_count);

void testEdgeWorkstealingLN(MMState *state, int row_node, int col_node, int actual_col_node, std::shared_ptr<EdgeQueue> eQueue, int row_count, bool edge_done, int level);