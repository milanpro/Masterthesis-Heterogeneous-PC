#include "../util/state.cuh"
#include <memory>

/**
  * Compact adjencency matrix into adj_comp as adjecency lists
  */
void callCompact(MMState *state, int deviceId, int idx, int numberOfGPUs,
                 int d_rowCount);
