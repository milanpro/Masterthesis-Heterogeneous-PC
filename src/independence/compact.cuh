#include "../util/state.cuh"
#include <memory>

void callCompact(MMState *state, int deviceId, int numberOfGPUs,
                 int d_rowCount);
