#pragma once
#include "./state.cuh"

void assertAdjCompactIsAdj(MMState *state);

void assertNodeStatus(MMState *state, int level);
