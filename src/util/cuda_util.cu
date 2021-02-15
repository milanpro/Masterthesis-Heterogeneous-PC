#include "cuda_util.cuh"

__device__ double PYTHAG(double a, double b) {
  double aAbs = fabs(a);
  double bAbs = fabs(b);
  double c;
  double result;
  if (aAbs > bAbs) {
    c = bAbs / aAbs;
    result = aAbs * sqrt(1.0 + c * c);
  } else if (bAbs > 0.0) {
    c = aAbs / bAbs;
    result = bAbs * sqrt(1.0 + c * c);
  } else {
    result = 0.0;
  }
  return (result);
}

__host__ __device__ double calcPValue(double r, int sampleSize) {
  r = isnan(r) ? 0.0 : fmin(0.9999999, fabs(r));
  double absz = sqrt(sampleSize - 3.0) * 0.5 * log1p(2.0 * r / (1.0 - r));
  return 2.0 * (1.0 - normcdf(absz));
}

__host__ __device__ double pValL1(double x1, double x2, double x3, int sampleSize) {
  // with edge i, j given k values are:
  // x1: edge i, j
  // x2: edge i, k
  // x3: edge j, k
  double r = (x1 - x2 * x3) / sqrt((1.0 - x3 * x3) * (1.0 - x2 * x2));
  return calcPValue(r, sampleSize);
}

__host__ __device__ size_t binomialCoeff(int n, int k) {
  if (n < k) {
    return 0;
  }
  size_t res = 1;
  if (k > n - k)
    k = n - k;
  for (int i = 0; i < k; ++i) {
    res *= (n - i);
    res /= (i + 1);
  }
  return res;
}

__host__ __device__ void ithCombination(int *sepset_nodes, size_t comb_number,
                                    int level, size_t neighbours) {
  int sum = 0;
  int tmp = 0;
  for (int i = 0; i < level; i++) {
    sepset_nodes[i] = 0;
    if (i > 0) {
      sepset_nodes[i] = sepset_nodes[i - 1];
    }
    while (sum <= comb_number) {
      sepset_nodes[i]++;
      tmp = binomialCoeff(neighbours - sepset_nodes[i], level - (i + 1));
      sum = sum + tmp;
    }
    sum = sum - tmp;
  }
}

template <int lvlSize>
__device__ void pseudoinverse(double Submat[][lvlSize],
                                double SubmatPInv[][lvlSize],
                                double v[][lvlSize], double *rv1, double *w,
                                double res[][lvlSize]) {
  int flag, its, i, j, jj, k, l, nm;
  double c, f, h, s, x, y, z;
  double anorm = 0.0, g = 0.0, scale = 0.0;
  /* Householder reduction to bidiagonal form */
  for (i = 0; i < lvlSize; i++) {
    /* left-hand reduction */
    l = i + 1;
    rv1[i] = scale * g;
    g = s = scale = 0.0;
    if (i < lvlSize) {
      for (k = i; k < lvlSize; k++)
        scale += fabs(Submat[k][i]);
      if (scale) {
        for (k = i; k < lvlSize; k++) {
          Submat[k][i] = (Submat[k][i] / scale);
          s += (Submat[k][i] * Submat[k][i]);
        }
        f = Submat[i][i];
        g = -SIGN(sqrt(s), f);
        h = f * g - s;
        Submat[i][i] = f - g;
        if (i != lvlSize - 1) {
          for (j = l; j < lvlSize; j++) {
            for (s = 0.0, k = i; k < lvlSize; k++)
              s += (Submat[k][i] * Submat[k][j]);
            f = s / h;
            for (k = i; k < lvlSize; k++)
              Submat[k][j] += (f * Submat[k][i]);
          }
        }
        for (k = i; k < lvlSize; k++)
          Submat[k][i] = (Submat[k][i] * scale);
      }
    }
    w[i] = scale * g;

    /* right-hand reduction */
    g = s = scale = 0.0;
    if (i < lvlSize && i != lvlSize - 1) {
      for (k = l; k < lvlSize; k++)
        scale += fabs(Submat[i][k]);
      if (scale) {
        for (k = l; k < lvlSize; k++) {
          Submat[i][k] = (Submat[i][k] / scale);
          s += (Submat[i][k] * Submat[i][k]);
        }
        f = Submat[i][l];
        g = -SIGN(sqrt(s), f);
        h = f * g - s;
        Submat[i][l] = f - g;
        for (k = l; k < lvlSize; k++)
          rv1[k] = Submat[i][k] / h;
        if (i != lvlSize - 1) {
          for (j = l; j < lvlSize; j++) {
            for (s = 0.0, k = l; k < lvlSize; k++)
              s += (Submat[j][k] * Submat[i][k]);
            for (k = l; k < lvlSize; k++)
              Submat[j][k] += (s * rv1[k]);
          }
        }
        for (k = l; k < lvlSize; k++)
          Submat[i][k] = Submat[i][k] * scale;
      }
    }
    anorm = MAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
  }

  /* accumulate the right-hand transformation */
  for (i = lvlSize - 1; i >= 0; i--) {
    if (i < lvlSize - 1) {
      if (g) {
        for (j = l; j < lvlSize; j++)
          v[j][i] = (Submat[i][j] / Submat[i][l]) / g;
        /* double division to avoid underflow */
        for (j = l; j < lvlSize; j++) {
          for (s = 0.0, k = l; k < lvlSize; k++)
            s += (Submat[i][k] * v[k][j]);
          for (k = l; k < lvlSize; k++)
            v[k][j] += (s * v[k][i]);
        }
      }
      for (j = l; j < lvlSize; j++)
        v[i][j] = v[j][i] = 0.0;
    }
    v[i][i] = 1.0;
    g = rv1[i];
    l = i;
  }

  /* accumulate the left-hand transformation */
  for (i = lvlSize - 1; i >= 0; i--) {
    l = i + 1;
    g = w[i];
    if (i < lvlSize - 1) {
      for (j = l; j < lvlSize; j++)
        Submat[i][j] = 0.0;
    }
    if (g) {
      g = 1.0 / g;
      if (i != lvlSize - 1) {
        for (j = l; j < lvlSize; j++) {
          for (s = 0.0, k = l; k < lvlSize; k++)
            s += (Submat[k][i] * Submat[k][j]);
          f = (s / Submat[i][i]) * g;
          for (k = i; k < lvlSize; k++)
            Submat[k][j] += (f * Submat[k][i]);
        }
      }
      for (j = i; j < lvlSize; j++)
        Submat[j][i] = (Submat[j][i] * g);
    } else {
      for (j = i; j < lvlSize; j++)
        Submat[j][i] = 0.0;
    }
    ++Submat[i][i];
  }

  /* diagonalize the bidiagonal form */
  for (k = lvlSize - 1; k >= 0; k--) { /* loop over singular values */
    for (its = 0; its < ITERATIONS; its++) {   /* loop over allowed iterations */
      flag = 1;
      for (l = k; l >= 0; l--) { /* test for splitting */
        nm = l - 1;
        if (fabs(rv1[l]) + anorm == anorm) {
          flag = 0;
          break;
        }
        if (fabs(w[nm]) + anorm == anorm)
          break;
      }
      if (flag) {
        c = 0.0;
        s = 1.0;
        for (i = l; i <= k; i++) {
          f = s * rv1[i];
          if (fabs(f) + anorm != anorm) {
            g = w[i];
            h = PYTHAG(f, g);
            w[i] = h;
            h = 1.0 / h;
            c = g * h;
            s = (-f * h);
            for (j = 0; j < lvlSize; j++) {
              y = Submat[j][nm];
              z = Submat[j][i];
              Submat[j][nm] = (y * c + z * s);
              Submat[j][i] = (z * c - y * s);
            }
          }
        }
      }
      z = w[k];
      if (l == k) {    /* convergence */
        if (z < 0.0) { /* make singular value nonnegative */
          w[k] = (-z);
          for (j = 0; j < lvlSize; j++)
            v[j][k] = (-v[j][k]);
        }
        break;
      }
      if (its >= ITERATIONS) {
        printf("Not converged\n");
      }

      /* shift from bottom 2 x 2 minor */
      x = w[l];
      nm = k - 1;
      y = w[nm];
      g = rv1[nm];
      h = rv1[k];
      f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
      g = PYTHAG(f, 1.0);
      f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;

      /* next QR transformation */
      c = s = 1.0;
      for (j = l; j <= nm; j++) {
        i = j + 1;
        g = rv1[i];
        y = w[i];
        h = s * g;
        g = c * g;
        z = PYTHAG(f, h);
        rv1[j] = z;
        c = f / z;
        s = h / z;
        f = x * c + g * s;
        g = g * c - x * s;
        h = y * s;
        y = y * c;
        for (jj = 0; jj < lvlSize; jj++) {
          x = v[jj][j];
          z = v[jj][i];
          v[jj][j] = (x * c + z * s);
          v[jj][i] = (z * c - x * s);
        }
        z = PYTHAG(f, h);
        w[j] = z;
        if (z) {
          z = 1.0 / z;
          c = f * z;
          s = h * z;
        }
        f = (c * g) + (s * y);
        x = (c * y) - (s * g);
        for (jj = 0; jj < lvlSize; jj++) {
          y = Submat[jj][j];
          z = Submat[jj][i];
          Submat[jj][j] = (y * c + z * s);
          Submat[jj][i] = (z * c - y * s);
        }
      }
      rv1[l] = 0.0;
      rv1[k] = f;
      w[k] = x;
    }
  }

  // start compute inverse matrix

  for (int rowNumber = 0; rowNumber < lvlSize; rowNumber++) {
    for (int colNumber = 0; colNumber < lvlSize; colNumber++) {
      res[rowNumber][colNumber] = v[rowNumber][colNumber] / w[colNumber];
    }
  }

  for (int rowNumber = 0; rowNumber < lvlSize; rowNumber++) {
    for (int colNumber = 0; colNumber < lvlSize; colNumber++) {
      SubmatPInv[rowNumber][colNumber] = 0;
      for (int thirdIndex = 0; thirdIndex < lvlSize; thirdIndex++) {
        SubmatPInv[rowNumber][colNumber] =
            SubmatPInv[rowNumber][colNumber] +
            res[rowNumber][thirdIndex] * Submat[colNumber][thirdIndex];
      }
    }
  }
}

template
__device__ void pseudoinverse(double Submat[][4],
                                double SubmatPInv[][4],
                                double v[][4], double *rv1, double *w,
                                double res[][4]);

template
__device__ void pseudoinverse(double Submat[][5],
                                double SubmatPInv[][5],
                                double v[][5], double *rv1, double *w,
                                double res[][5]);

template
__device__ void pseudoinverse(double Submat[][6],
                                double SubmatPInv[][6],
                                double v[][6], double *rv1, double *w,
                                double res[][6]);

template
__device__ void pseudoinverse(double Submat[][7],
                                double SubmatPInv[][7],
                                double v[][7], double *rv1, double *w,
                                double res[][7]);

template
__device__ void pseudoinverse(double Submat[][8],
                                double SubmatPInv[][8],
                                double v[][8], double *rv1, double *w,
                                double res[][8]);

template
__device__ void pseudoinverse(double Submat[][9],
                                double SubmatPInv[][9],
                                double v[][9], double *rv1, double *w,
                                double res[][9]);

template
__device__ void pseudoinverse(double Submat[][10],
                                double SubmatPInv[][10],
                                double v[][10], double *rv1, double *w,
                                double res[][10]);

template
__device__ void pseudoinverse(double Submat[][11],
                                double SubmatPInv[][11],
                                double v[][11], double *rv1, double *w,
                                double res[][11]);

template
__device__ void pseudoinverse(double Submat[][12],
                                double SubmatPInv[][12],
                                double v[][12], double *rv1, double *w,
                                double res[][12]);

template
__device__ void pseudoinverse(double Submat[][13],
                                double SubmatPInv[][13],
                                double v[][13], double *rv1, double *w,
                                double res[][13]);

template
__device__ void pseudoinverse(double Submat[][14],
                                double SubmatPInv[][14],
                                double v[][14], double *rv1, double *w,
                                double res[][14]);

template
__device__ void pseudoinverse(double Submat[][15],
                                double SubmatPInv[][15],
                                double v[][15], double *rv1, double *w,
                                double res[][15]);

template
__device__ void pseudoinverse(double Submat[][16],
                                double SubmatPInv[][16],
                                double v[][16], double *rv1, double *w,
                                double res[][16]);
