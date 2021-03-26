
namespace GPU {
#define CUT_THR 0.9999999
  __device__ double calcPValue(double r, int sampleSize)
  {
    r = isnan(r) ? 0.0 : fmin(CUT_THR, fabs(r));
    double absz = sqrt(sampleSize - 3.0) * 0.5 * log1p(2.0 * r / (1.0 - r));
    return 2.0 * (1.0 - normcdf(absz));
  }

  __device__ double pValL1(double x1, double x2, double x3, int sampleSize)
  {
    // with edge i, j given k values are:
    // x1: edge i, j
    // x2: edge i, k
    // x3: edge j, k
    double r = (x1 - x2 * x3) / sqrt((1.0 - x3 * x3) * (1.0 - x2 * x2));
    return calcPValue(r, sampleSize);
  }
}