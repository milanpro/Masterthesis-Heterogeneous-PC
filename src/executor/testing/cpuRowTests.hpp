#include "../../util/state.cuh"
#include "../cpuExecutor.hpp"

void testEdge(int level, MMState *state, int row_node, int col_node, std::shared_ptr<EdgeQueue> eQueue);