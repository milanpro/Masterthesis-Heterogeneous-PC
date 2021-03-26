namespace GPU
{
  __device__ double calcPValue(double r, int sampleSize);

  __device__ double pValL1(double x1, double x2, double x3, int sampleSize);
}