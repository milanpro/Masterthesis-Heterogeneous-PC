#pragma once

// Task to be balanced on the execution units
struct SplitTask
{
  int row;
  int rowCount;
};

// Result of an execution unit containing the duration of execution
struct TestResult
{
  unsigned long long duration;
  // Test count not implemented yet
  int tests;
};
