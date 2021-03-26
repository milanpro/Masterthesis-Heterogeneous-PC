#include "../../util/state.cuh"
#include "../cpuExecutor.hpp"
#include <atomic>

void testEdgeWorkstealingL1(MMState *state, int row_node, int col_node, int actual_col_node, std::shared_ptr<EdgeQueue> eQueue, int row_count, std::atomic<int> &edges_done);

void testEdgeWorkstealingLN(MMState *state, int row_node, int col_node, int actual_col_node, std::shared_ptr<EdgeQueue> eQueue, int row_count, std::atomic<int> &edges_done, bool edge_done, int level);