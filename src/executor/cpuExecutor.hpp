#pragma once
#include "executor.hpp"
#include "../util/concurrentqueue.h"

struct DeletedEdge
{
  int row;
  int col;
  double pMax;
  std::vector<int> sepSet;
};

typedef moodycamel::ConcurrentQueue<DeletedEdge> EdgeQueue;

struct CPUExecutor : Executor
{
  MMState *state;
  std::shared_ptr<EdgeQueue> deletedEdges;
  std::vector<std::tuple<int, int>> rowLengthMap;

  TestResult executeLevel(int level, int numThreads, bool verbose = false);
  TestResult workstealingExecuteLevel(int level, int numThreads, bool verbose);

  void calculateRowLengthMap(int level);
  void migrateEdges(int level, bool verbose = false);

  CPUExecutor(MMState *state) : state(state)
  {
    deletedEdges = std::make_shared<EdgeQueue>();
  }
};
