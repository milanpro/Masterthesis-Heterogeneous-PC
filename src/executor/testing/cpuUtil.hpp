#pragma once
#include <vector>
#include "../cpuExecutor.hpp"

void deleteEdgeLevel0(MMState *state, int col_node, int row_node, double pMax);

void deleteEdge(MMState *state, int level, int col_node, int row_node, double pMax, std::vector<int> sepSet);

void enqueueEdgeDeletion(std::shared_ptr<EdgeQueue> deletedEdges, int col_node, int row_node, double pMax, std::vector<int> sepSet);