#ifndef R_PACKAGE_SRC_UTIL_MATRIXPRINT_CUH_
#define R_PACKAGE_SRC_UTIL_MATRIXPRINT_CUH_
#include <iostream>

template <typename T>
void printMatrix(int m, int n, const T *A, const char *name) {
  for (int row = 0; row < m; row++) {
    for (int col = 0; col < n; col++) {
      T Areg = A[row * m + col];
      std::cout << name << "(" << row << ", " << col << ") = " << Areg
                << std::endl;
    }
  }
}

template <typename T>
void printGPUMatrix(int m, int n, const T *d_A, const char *name) {
  T *h_A = new T[m * n];
  cudaMemcpy(h_A, d_A, sizeof(T) * m * n, cudaMemcpyDeviceToHost);
  for (int row = 0; row < m; row++) {
    for (int col = 0; col < n; col++) {
      T Areg = h_A[row * m + col];
      std::cout << name << "(" << row << ", " << col << ") = " << Areg
                << std::endl;
    }
  }
  delete[] h_A;
}
#endif // R_PACKAGE_SRC_UTIL_MATRIXPRINT_CUH_
