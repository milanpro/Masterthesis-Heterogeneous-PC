#include "../util/state.cuh"
#include <memory>

void callCompact(GPUState *state, int deviceId, int numberOfGPUs,
                 int d_rowCount);
