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
  TestResult executeLevel(int level, bool verbose = false);
  TestResult workstealingExecuteLevel(int level, bool verbose);
  void migrateEdges(int level, bool verbose = false);
  CPUExecutor(MMState *state) : state(state)
  {
    deletedEdges = std::make_shared<EdgeQueue>();
  }
};
