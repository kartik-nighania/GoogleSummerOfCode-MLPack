/**
 * @file cmaes.hpp
 * @author Marcus Edel
 * @author Kartik Nighania
 *
 * Covariance Matrix Adaptation Evolution Strategy
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_CMAES_CMAES_HPP
#define MLPACK_CORE_OPTIMIZERS_CMAES_CMAES_HPP

#include <mlpack/core/math/random.hpp>

#include "random.hpp"
#include "timings.hpp"

namespace mlpack {
namespace optimization {

/**
*CMA-ES stands for Covariance Matrix Adaptation Evolution Strategy.
*Evolution strategies (ES) are stochastic, derivative-free methods for 
*numerical optimization of non-linear or non-convex continuous optimization
*problems. They belong to the class of evolutionary algorithms and 
*evolutionary computation.
*
*An evolutionary algorithm is broadly based on the principle
*of biological evolution, namely the repeated interplay
*of variation (via recombination and mutation) and selection: in each 
*generation (iteration) new individuals (candidate solutions, denoted as
*x) are generated by variation, usually in a stochastic
*way, of the current parental individuals. Then, some individuals are
*selected to become the parents in the next generation based on their
*fitness or objective function value f(x). Like
*this, over the generation sequence, individuals with better and better
*f-values are generated.
*/
class CMAES
{
 public:
/**
* Constructor for the CMAES optimizer.
* 
* @param objectDim the dimension of the object
* @param start the initial start point of the optimizer
*        in armadillo mat or vec
* @param stdDivs the intial standard deviation to choose for the
*        gaussian distribution.
* @param iters the maximum number of iterations to reach the minimum.
*        it may happen that the function gets terminated by reaching 
*        the condition and not using the remaining iterations
* @param evalDiff the change in function value to see if flat fitness
*        is matched which is the condition mostly when minima is reached. 
*/ 
CMAES(int objectDim = 0,
double start = 0, double stdDivs = 0,
double iters = 0, double evalEnd = 0);

/**
* Optimize the given function using CMAES. The function will be given
* as a parameter along with a armadillo matrix of vector in which
* final function value coordinates will be copied. It also return
* the final objective function value in double after the complete
* optimization process.
*
* @param  function the function to be optimized
* @param  arr to put the final coordinates that are found of
*         each dimension.
* @return inal objective value obtained.
*/
template<typename funcType>
double Optimize(funcType& function, double* arr);

 private:
//! stores the fitness values of functions
//arma::vec arFunvals;
//! Problem dimension, must stay constant.
/* Input parameters. */
  //! Problem dimension, must stay constant.
  double* arFunvals;
  double countevals;
  Timing eigenTimings;

  int N;
  //! Initial search space vector.
  double* xstart;
  //! Initial standard deviations.
  arma::vec rgInitialStds;

  /* Termination parameters. */
  //! Maximal number of objective function evaluations.
  double stopMaxFunEvals;
  double facmaxeval;
  //! Maximal number of iterations.
  double stopMaxIter;
  //! Minimal fitness value. Only activated if flg is true.
  struct { bool flg; double val; } stStopFitness;
  //! Minimal value difference.
  double stopTolFun;
  //! Minimal history value difference.
  double stopTolFunHist;
  //! Minimal search space step size.
  double stopTolX;
  //! Defines the maximal condition number.
  double stopTolUpXFactor;

  /* internal evolution strategy parameters */
  /**
   * Population size. Number of samples per iteration, at least two,
   * generally > 4.
   */
  int lambda;
  /**
   * Number of individuals used to recompute the mean.
   */
  int mu;
  double mucov;
  /**
   * Variance effective selection mass, should be lambda/4.
   */
  double mueff;
  /**
   * Weights used to recombinate the mean sum up to one.
   */
  double* weights;
  /**
   * Damping parameter for step-size adaption, d = inifinity or 0 means adaption
   * is turned off, usually close to one.
   */
  double damps;
  /**
   * cs^-1 (approx. n/3) is the backward time horizon for the evolution path
   * ps and larger than one.
   */
  double cs;
  double ccumcov;
  /**
   * ccov^-1 (approx. n/4) is the backward time horizon for the evolution path
   * pc and larger than one.
   */
  double ccov;
  double diagonalCov;
  struct { double modulo; double maxtime; } updateCmode;
  double facupdateCmode;

  //!< Random number generator.
  Random<double> rand;

  //! Step size.
  double sigma;
  //! Mean x vector, "parent".
  double* xmean;
  //! Best sample ever.
  arma::vec xBestEver;
  //! x-vectors, lambda offspring.
  double** population;
  //! Sorting index of sample population.
  int* index;
  //! History of function values.
  double* funcValueHistory;

  double chiN;
  //! Lower triangular matrix: i>=j for C[i][j].
  double** C;
  //! Matrix with normalize eigenvectors in columns.
  double** B;
  //! Axis lengths.
  double* rgD;
  //! Anisotropic evolution path (for covariance).
  arma::vec pc;
  //! Isotropic evolution path (for step length).
  arma::vec ps;
  //! Last mean.
  arma::vec xold;
  //! B*D*z.
  arma::vec BDz;
  //! Temporary (random) vector used in different places.
  double* tempRandom;
  //! Objective function values of the population.
  double* functionValues;
  //!< Public objective function value array returned by init().
  double* publicFitness;

  //! Generation number.
  double gen;
  //! Algorithm state.
  enum {INITIALIZED, SAMPLED, UPDATED} state;

  // repeatedly used for output
  double maxdiagC;
  double mindiagC;
  double maxEW;
  double minEW;

  bool eigensysIsUptodate;
  bool doCheckEigen; //!< control via signals.par
  double genOfEigensysUpdate;

  double dMaxSignifKond;
  double dLastMinEWgroesserNull;

//! eigen value decomposition and update
void updateEigensystem(bool force);
//! adapt the covariance matrix to the new distribution
void adaptC2(const int hsig);
//! initialize all the variables used in CMAES with default values
void init(double* func);
//! creates the population from a gaussian normal distribution = lambda
void samplePopulation();
//! updates the distribution according to the best fitness value selected.
void updateDistribution(double* fitnessValues);
//! test for termination of the algorithm if the condition values are reached.
bool testForTermination();

double square(double d)
{
  return d*d;
}

double maxElement(const double* rgd, int len)
{
  return *std::max_element(rgd, rgd + len);
}

double minElement(const double* rgd, int len)
{
  return *std::min_element(rgd, rgd + len);
}

  void sortIndex(const double* rgFunVal, int* iindex, int n)
  {
    int i, j;
    for(i = 1, iindex[0] = 0; i < n; ++i)
    {
      for(j = i; j > 0; --j)
      {
        if(rgFunVal[iindex[j - 1]] < rgFunVal[i])
          break;
        iindex[j] = iindex[j - 1]; // shift up
      }
      iindex[j] = i;
    }
  }

  /**
   * Calculating eigenvalues and vectors.
   * @param rgtmp (input) N+1-dimensional vector for temporal use. 
   * @param diag (output) N eigenvalues. 
   * @param Q (output) Columns are normalized eigenvectors.
   */
  void eigen(double* diag, double** Q, double* rgtmp)
  {
    assert(rgtmp && "eigen(): input parameter rgtmp must be non-NULL");

    if(C != Q) // copy C to Q
    {
      for(int i = 0; i < N; ++i)
        for(int j = 0; j <= i; ++j)
          Q[i][j] = Q[j][i] = C[i][j];
    }

    householder(Q, diag, rgtmp);
    ql(diag, rgtmp, Q);
  }

  /** 
   * Exhaustive test of the output of the eigendecomposition, needs O(n^3)
   * operations writes to error file.
   * @return number of detected inaccuracies
   */
  int checkEigen(double* diag, double** Q)
  {
    // compute Q diag Q^T and Q Q^T to check
    int res = 0;
    for(int i = 0; i < N; ++i)
      for(int j = 0; j < N; ++j) {
        double cc = 0., dd = 0.;
        for(int k = 0; k < N; ++k)
        {
          cc += diag[k]*Q[i][k]*Q[j][k];
          dd += Q[i][k]*Q[j][k];
        }
        // check here, is the normalization the right one?
        const bool cond1 = fabs(cc - C[i > j ? i : j][i > j ? j : i]) / sqrt(C[i][i]* C[j][j]) > double(1e-10);
        const bool cond2 = fabs(cc - C[i > j ? i : j][i > j ? j : i]) > double(3e-14);
        if(cond1 && cond2)
        {
          
         std::cout << "eigen(): imprecise result detected " 
                << std::endl;
          ++res;
        }
        if(std::fabs(dd - (i == j)) > double(1e-10))
        {
        
          std::cout << "eigen(): imprecise result detected (Q not orthog.)"
                 << std::endl;
          ++res;
        }
      }
    return res;
  }

  /**
   * Symmetric tridiagonal QL algorithm, iterative.
   * Computes the eigensystem from a tridiagonal matrix in roughtly 3N^3 operations
   * code adapted from Java JAMA package, function tql2.
   * @param d input: Diagonale of tridiagonal matrix. output: eigenvalues.
   * @param e input: [1..n-1], off-diagonal, output from Householder
   * @param V input: matrix output of Householder. output: basis of
   *          eigenvectors, according to d
   */
  void ql(double* d, double* e, double** V)
  {
   
    double f(0);
    double tst1(0);
    const double eps(2.22e-16); // 2.0^-52.0 = 2.22e-16

    // shift input e
    double* ep1 = e;
    for(double *ep2 = e+1, *const end = e+N; ep2 != end; ep1++, ep2++)
      *ep1 = *ep2;
    *ep1 = double(0); // never changed again

    for(int l = 0; l < N; l++)
    {
      // find small subdiagonal element
      double& el = e[l];
      double& dl = d[l];
      const double smallSDElement = std::fabs(dl) + std::fabs(el);
      if(tst1 < smallSDElement)
        tst1 = smallSDElement;
      const double epsTst1 = eps*tst1;
      int m = l;
      while(m < N)
      {
        if(std::fabs(e[m]) <= epsTst1) break;
        m++;
      }

      // if m == l, d[l] is an eigenvalue, otherwise, iterate.
      if(m > l)
      {
        do {
          double h, g = dl;
          double& dl1r = d[l+1];
          double p = (dl1r - g) / (double(2)*el);
          double r = myhypot(p, double(1));

          // compute implicit shift
          if(p < 0) r = -r;
          const double pr = p+r;
          dl = el/pr;
          h = g - dl;
          const double dl1 = el*pr;
          dl1r = dl1;
          for(int i = l+2; i < N; i++) d[i] -= h;
          f += h;

          // implicit QL transformation.
          p = d[m];
          double c(1);
          double c2(1);
          double c3(1);
          const double el1 = e[l+1];
          double s(0);
          double s2(0);
          for(int i = m-1; i >= l; i--)
          {
            c3 = c2;
            c2 = c;
            s2 = s;
            const double& ei = e[i];
            g = c*ei;
            h = c*p;
            r = myhypot(p, ei);
            e[i+1] = s*r;
            s = ei/r;
            c = p/r;
            const double& di = d[i];
            p = c*di - s*g;
            d[i+1] = h + s*(c*g + s*di);

            // accumulate transformation.
            for(int k = 0; k < N; k++)
            {
              double& Vki1 = V[k][i+1];
              h = Vki1;
              double& Vki = V[k][i];
              Vki1 = s*Vki + c*h;
              Vki *= c; Vki -= s*h;
            }
          }
          p = -s*s2*c3*el1*el/dl1;
          el = s*p;
          dl = c*p;
        } while(std::fabs(el) > epsTst1);
      }
      dl += f;
      el = 0.0;
    }
  }

  /**
   * Householder transformation of a symmetric matrix V into tridiagonal form.
   * Code slightly adapted from the Java JAMA package, function private tred2().
   * @param V input: symmetric nxn-matrix. output: orthogonal transformation
   *          matrix: tridiag matrix == V* V_in* V^t.
   * @param d output: diagonal
   * @param e output: [0..n-1], off diagonal (elements 1..n-1)
   */
  void householder(double** V, double* d, double* e)
  {

    for(int j = 0; j < N; j++)
    {
      d[j] = V[N - 1][j];
    }

    // Householder reduction to tridiagonal form

    for(int i = N - 1; i > 0; i--)
    {
      // scale to avoid under/overflow
      double scale = 0.0;
      double h = 0.0;
      for(double *pd = d, *const dend = d+i; pd != dend; pd++)
      {
        scale += std::fabs(*pd);
      }
      if(scale == 0.0)
      {
        e[i] = d[i-1];
        for(int j = 0; j < i; j++)
        {
          d[j] = V[i-1][j];
          V[i][j] = 0.0;
          V[j][i] = 0.0;
        }
      }
      else
      {
        // generate Householder vector
        for(double *pd = d, *const dend = d+i; pd != dend; pd++)
        {
          *pd /= scale;
          h += *pd * *pd;
        }
        double& dim1 = d[i-1];
        double f = dim1;
        double g = f > 0 ? -std::sqrt(h) : std::sqrt(h);
        e[i] = scale*g;
        h = h - f* g;
        dim1 = f - g;
        memset((void *) e, 0, (size_t)i*sizeof(double));

        // apply similarity transformation to remaining columns
        for(int j = 0; j < i; j++)
        {
          f = d[j];
          V[j][i] = f;
          double& ej = e[j];
          g = ej + V[j][j]* f;
          for(int k = j + 1; k <= i - 1; k++)
          {
            double& Vkj = V[k][j];
            g += Vkj*d[k];
            e[k] += Vkj*f;
          }
          ej = g;
        }
        f = 0.0;
        for(int j = 0; j < i; j++)
        {
          double& ej = e[j];
          ej /= h;
          f += ej* d[j];
        }
        double hh = f / (h + h);
        for(int j = 0; j < i; j++)
        {
          e[j] -= hh*d[j];
        }
        for(int j = 0; j < i; j++)
        {
          double& dj = d[j];
          f = dj;
          g = e[j];
          for(int k = j; k <= i - 1; k++)
          {
            V[k][j] -= f*e[k] + g*d[k];
          }
          dj = V[i-1][j];
          V[i][j] = 0.0;
        }
      }
      d[i] = h;
    }

    // accumulate transformations
    const int nm1 = N-1;
    for(int i = 0; i < nm1; i++)
    {
      double h;
      double& Vii = V[i][i];
      V[N-1][i] = Vii;
      Vii = 1.0;
      h = d[i+1];
      if(h != 0.0)
      {
        for(int k = 0; k <= i; k++)
        {
          d[k] = V[k][i+1] / h;
        }
        for(int j = 0; j <= i; j++) {
          double g = 0.0;
          for(int k = 0; k <= i; k++)
          {
            double* Vk = V[k];
            g += Vk[i+1]* Vk[j];
          }
          for(int k = 0; k <= i; k++)
          {
            V[k][j] -= g*d[k];
          }
        }
      }
      for(int k = 0; k <= i; k++)
      {
        V[k][i+1] = 0.0;
      }
    }
    for(int j = 0; j < N; j++)
    {
      double& Vnm1j = V[N-1][j];
      d[j] = Vnm1j;
      Vnm1j = 0.0;
    }
    V[N-1][N-1] = 1.0;
    e[0] = 0.0;
  }

  double myhypot(double a, double b)
{
  const register double fabsa = std::fabs(a), fabsb = std::fabs(b);
  if(fabsa > fabsb)
  {
    const register double r = b / a;
    return fabsa*std::sqrt(double(1)+r*r);
  }
  else if(b != double(0))
  {
    const register double r = a / b;
    return fabsb*std::sqrt(double(1)+r*r);
  }
  else
    return double(0);
}


};
} // namespace optimization
} // namespace mlpack

// Include implementation.
#include "cmaes_impl.hpp"

#endif
