#include "mem_info.cuh"
#include <iostream>

void printMemInfo() {
  size_t free_byte;
  size_t total_byte;
  cudaMemGetInfo(&free_byte, &total_byte);
  double free_db = static_cast<double>(free_byte);
  double total_db = static_cast<double>(total_byte);
  double used_db = total_db - free_db;
  printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
         used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0,
         total_db / 1024.0 / 1024.0);
}
