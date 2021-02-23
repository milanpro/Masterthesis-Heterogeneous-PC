#pragma once
#define NUMTHREADS 64

struct SplitTask
{
  int row;
  int rowCount;
};

struct TestResult
{
  unsigned long long duration;
  int tests;
};
