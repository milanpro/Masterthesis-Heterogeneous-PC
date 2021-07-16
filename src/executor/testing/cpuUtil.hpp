#pragma once
#include <vector>
#include "../cpuExecutor.hpp"

/**
 * Delete edge in level 0. No seperation set needed.
 */
void deleteEdgeLevel0(MMState *state, int col_node, int row_node, double pMax);

/**
 * Delete edge in level 1+.
 */
void deleteEdge(MMState *state, int level, int col_node, int row_node, double pMax, std::vector<int> sepSet);

/**
 * If edge deletions should be migrated in between level, this is used instead of delete by the CPU execution
 * Deleted edges and their sepsets are enqueued for later
 */
void enqueueEdgeDeletion(std::shared_ptr<EdgeQueue> deletedEdges, int col_node, int row_node, double pMax, std::vector<int> sepSet);