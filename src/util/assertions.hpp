#pragma once
#include "./state.cuh"

/**
 * Compare the compacted adjencency matrix and the non compacted one.
 * Assert graph equality
 * Only used in Debug builds
 */
void assertAdjCompactIsAdj(MMState *state);

/**
 * Assert node status bits set for all prior existing edges.
 * Asserts that every edge in the graph has been processed
 * Only used in Debug builds
 */
void assertNodeStatus(MMState *state, int level);
