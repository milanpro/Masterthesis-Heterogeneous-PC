#include "cpuInd.hpp"
#include "boost/math/distributions/normal.hpp"
#include "boost/math/special_functions/log1p.hpp"

namespace CPU
{
#define CUT_THR 0.9999999
  double calcPValue(double r_in, int sampleSize)
  {
    double r = boost::math::isnan(r_in) ? 0.0 : std::min(CUT_THR, std::abs(r_in));
    double absz = sqrt(sampleSize - 3.0) * 0.5 * boost::math::log1p(2 * r / (1 - r));
    boost::math::normal distN;
    return (2 * boost::math::cdf(boost::math::complement(distN, absz)));
  }

  double pValL1(double x1, double x2, double x3, int sampleSize)
  {
    // with edge i, j given k values are:
    // x1: edge i, j
    // x2: edge i, k
    // x3: edge j, k
    double r = (x1 - x2 * x3) / sqrt((1.0 - x3 * x3) * (1.0 - x2 * x2));
    if (boost::math::isnan(x1) || boost::math::isnan(x2) || boost::math::isnan(x3)) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    return calcPValue(r, sampleSize);
  }

  double pValLN(arma::dmat Submat, int observations)
  {
    for (auto i = 0; i < Submat.n_rows; ++i)
      for (auto j = 0; j < Submat.n_cols; ++j)
        if ((boost::math::isnan)(Submat(i, j))) return std::numeric_limits<double>::quiet_NaN();

    arma::mat SubmatPInv = arma::pinv(Submat);
    double r = -SubmatPInv(0, 1) / sqrt(SubmatPInv(0, 0) * SubmatPInv(1, 1));
    return calcPValue(r, observations);
  }
}
