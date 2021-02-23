#include <stdint.h>

void gpuPMCC(const double *h_mat, uint64_t n, int dim, double *cors, bool verbose);

void gpuPMCCShared(const double *h_mat, uint64_t n, int dim, double *cors);
