#include "armadillo"

namespace CPU
{
  double calcPValue(double r_in, int sampleSize);

  double pValL1(double x1, double x2, double x3, int sampleSize);

  double pValLN(arma::dmat Submat, int observations);
}
