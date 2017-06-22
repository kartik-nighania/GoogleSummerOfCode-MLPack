/**
 * @file cmaes.h
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

#include "random.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <armadillo>
#include <iostream>

//2 eigen values check for column values
//index sort mabye we can use sort
//get stop message
//return value in a specific way

//random number generator
// enum work
// eignevalue column vs row check

namespace mlpack {
namespace optimization {

template<typename T>
class CMAES
{
public:

  /* Input parameters. */
  //! Problem dimension, must stay constant.
  int N;
  //! Initial search space vector.
  T* xstart;
  //! A typical value for a search space vector.
  T* typicalX;
  //! Indicates that the typical x is the initial point.
  bool typicalXcase;
  //! Initial standard deviations.
  T* rgInitialStds;
  T* rgDiffMinChange;

  /* Termination parameters. */
  //! Maximal number of objective function evaluations.
  T stopMaxFunEvals;
  T facmaxeval;
  //! Maximal number of iterations.
  T stopMaxIter;
  //! Minimal fitness value. Only activated if flg is true.
  struct { bool flg; T val; } stStopFitness;
  //! Minimal value difference.
  T stopTolFun;
  //! Minimal history value difference.
  T stopTolFunHist;
  //! Minimal search space step size.
  T stopTolX;
  //! Defines the maximal condition number.
  T stopTolUpXFactor;

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
  T mucov;
  /**
   * Variance effective selection mass, should be lambda/4.
   */
  T mueff;
  /**
   * Weights used to recombinate the mean sum up to one.
   */
  T* weights;
  /**
   * Damping parameter for step-size adaption, d = inifinity or 0 means adaption
   * is turned off, usually close to one.
   */
  T damps;
  /**
   * cs^-1 (approx. n/3) is the backward time horizon for the evolution path
   * ps and larger than one.
   */
  T cs;
  T ccumcov;
  /**
   * ccov^-1 (approx. n/4) is the backward time horizon for the evolution path
   * pc and larger than one.
   */
  T ccov;
  T diagonalCov;
  struct { T modulo; T maxtime; } updateCmode;
  T facupdateCmode;

  /**
   * Determines the method used to initialize the weights.
   */
  enum Weights
  {
    UNINITIALIZED_WEIGHTS, LINEAR_WEIGHTS, EQUAL_WEIGHTS, LOG_WEIGHTS
  } weightMode;

  //! File that contains an optimization state that should be resumed.
  std::string resumefile;

  //! Set to true to activate logging warnings.
  bool logWarnings;
  //! Output stream that is used to log warnings, usually std::cerr.
  std::ostream& logStream;


  CMAES(int dimension = 0, const T* inxstart = 0, const T* inrgsigma = 0)
      : N(-1),
        xstart(0),
        typicalX(0),
        typicalXcase(false),
        rgInitialStds(0),
        rgDiffMinChange(0),
        stopMaxFunEvals(-1),
        facmaxeval(1.0),
        stopMaxIter(-1.0),
        stopTolFun(1e-12),
        stopTolFunHist(1e-13),
        stopTolX(0), // 1e-11*insigma would also be reasonable
        stopTolUpXFactor(1e3),
        lambda(-1),
        mu(-1),
        mucov(-1),
        mueff(-1),
        weights(0),
        damps(-1),
        cs(-1),
        ccumcov(-1),
        ccov(-1),
        facupdateCmode(1),
        weightMode(UNINITIALIZED_WEIGHTS),
        resumefile(""),
        logWarnings(false),
        logStream(std::cerr)
  {
    stStopFitness.flg = false;
    stStopFitness.val = -std::numeric_limits<T>::max();
    updateCmode.modulo = -1;
    updateCmode.maxtime = -1;
  
    if(logWarnings)
    {
      if(!(xstart || inxstart || typicalX))
        logStream << "Warning: initialX undefined. typicalX = 0.5...0.5." << std::endl;
      if(!(rgInitialStds || inrgsigma))
        logStream << "Warning: initialStandardDeviations undefined. 0.3...0.3." << std::endl;
    }

    if(dimension <= 0 && N <= 0)
      throw std::runtime_error("Problem dimension N undefined.");
    else if(dimension > 0)
      N = dimension;

    if(weightMode == UNINITIALIZED_WEIGHTS)
      weightMode = LOG_WEIGHTS;

    diagonalCov = 0; // default is 0, but this might change in future

    if(!xstart)
    {
      xstart = new T[N];
      if(inxstart)
      {
        for(int i = 0; i < N; ++i)
          xstart[i] = inxstart[i];
      }
      else if(typicalX)
      {
        typicalXcase = true;
        for(int i = 0; i < N; ++i)
          xstart[i] = typicalX[i];
      }
      else
      {
        typicalXcase = true;
        for(int i = 0; i < N; i++)
          xstart[i] = 0.5;
      }
    }

    if(!rgInitialStds)
    {
      rgInitialStds = new T[N];
      if(inrgsigma)
      {
        for(int i = 0; i < N; ++i)
          rgInitialStds[i] = inrgsigma[i];
      }
      else
      {
        for(int i = 0; i < N; ++i)
          rgInitialStds[i] = T(0.3);
      }
    }

    if(lambda < 2)
      lambda = 4 + (int) (3.0*log((double) N));
    if(mu <= 0)
      mu = lambda / 2;
    if(!weights)
     {
     if(weights)
        delete[] weights;
      weights = new T[mu];
      switch(weightMode)
      {
      case LINEAR_WEIGHTS:
        for(int i = 0; i < mu; ++i) weights[i] = mu - i;
        break;
      case EQUAL_WEIGHTS:
        for(int i = 0; i < mu; ++i) weights[i] = 1;
        break;
      case LOG_WEIGHTS:
      default:
        for(int i = 0; i < mu; ++i) weights[i] = log(mu + 1.) - log(i + 1.);
        break;
      }

      // normalize weights vector and set mueff
      T s1 = 0, s2 = 0;
      for(int i = 0; i < mu; ++i)
      {
        s1 += weights[i];
        s2 += weights[i]*weights[i];
      }
      mueff = s1*s1/s2;
      for(int i = 0; i < mu; ++i)
        weights[i] /= s1;

      if(mu < 1 || mu > lambda || (mu == lambda && weights[0] == weights[mu - 1]))
        throw std::runtime_error("setWeights(): invalid setting of mu or lambda");
    }

    if(cs > 0)
      cs *= (mueff + 2.) / (N + mueff + 3.);
    if(cs <= 0 || cs >= 1)
      cs = (mueff + 2.) / (N + mueff + 3.);

    if(ccumcov <= 0 || ccumcov > 1)
      ccumcov = 4. / (N + 4);

    if(mucov < 1)
      mucov = mueff;
    T t1 = 2. / ((N + 1.4142)*(N + 1.4142));
    T t2 = (2.* mueff - 1.) / ((N + 2.)*(N + 2.) + mueff);
    t2 = (t2 > 1) ? 1 : t2;
    t2 = (1. / mucov)* t1 + (1. - 1. / mucov)* t2;
    if(ccov >= 0)
      ccov *= t2;
    if(ccov < 0 || ccov > 1)
      ccov = t2;

    if(diagonalCov < 0)
      diagonalCov = 2 + 100. * N / sqrt((double) lambda);

    if(stopMaxFunEvals <= 0)
      stopMaxFunEvals = facmaxeval * 900 * (N + 3)*(N + 3);
    else
      stopMaxFunEvals *= facmaxeval;

    if(stopMaxIter <= 0)
      stopMaxIter = ceil((double) (stopMaxFunEvals / lambda));

    if(damps < T(0))
      damps = T(1);
    damps = damps
        * (T(1) + T(2)*std::max(T(0), std::sqrt((mueff - T(1)) / (N + T(1))) - T(1)))
        * (T) std::max(T(0.3), T(1) - // modify for short runs
          (T) N / (T(1e-6) + std::min(stopMaxIter, stopMaxFunEvals / lambda)))
        + cs;

    if(updateCmode.modulo < 0)
      updateCmode.modulo = 1. / ccov / (double) N / 10.;
    updateCmode.modulo *= facupdateCmode;
    if(updateCmode.maxtime < 0)
      updateCmode.maxtime = 0.20; // maximal 20% of CPU-time
  
  }

  /**
   * Keys for get().
   */
  enum GetScalar
  {
    NoScalar = 0,
    AxisRatio = 1,
    Eval = 2, Evaluations = 2,
    FctValue = 3, FuncValue = 3, FunValue = 3, Fitness = 3,
    FBestEver = 4,
    Generation = 5, Iteration = 5,
    MaxEval = 6, MaxFunEvals = 6, StopMaxFunEvals = 6,
    MaxGen = 7, MaxIter = 7, StopMaxIter = 7,
    MaxAxisLength = 8,
    MinAxisLength = 9,
    MaxStdDev = 10,
    MinStdDev = 11,
    Dim = 12, Dimension = 12,
    Lambda = 13, SampleSize = 13, PopSize = 13,
    Sigma = 14
  };

  /**
   * Keys for getPtr()
   */
  enum GetVector
  {
    NoVector = 0,
    DiagC = 1,
    DiagD = 2,
    StdDev = 3,
    XBestEver = 4,
    XBest = 5,
    XMean = 6
  };

private:

  Random<T> rand;

  //! Step size.
  T sigma;
  //! Mean x vector, "parent".
  T* xmean;
  //! Best sample ever.
  T* xBestEver;
  //! x-vectors, lambda offspring.
  T** population;
  //! Sorting index of sample population.
  int* index;
  //! History of function values.
  T* funcValueHistory;

  T chiN;
  //! Lower triangular matrix: i>=j for C[i][j].
  T** C;
  //! Matrix with normalize eigenvectors in columns.
  T** B;
  //! Axis lengths.
  T* rgD;
  //! Anisotropic evolution path (for covariance).
  T* pc;
  //! Isotropic evolution path (for step length).
  T* ps;
  //! Last mean.
  T* xold;
  //! Output vector.
  T* output;
  //! B*D*z.
  T* BDz;
  //! Temporary (random) vector used in different places.
  T* tempRandom;
  //! Objective function values of the population.
  T* functionValues;
  //!< Public objective function value array returned by init().
  T* publicFitness;

  //! Generation number.
  T gen;
  //! Algorithm state.
  enum {INITIALIZED, SAMPLED, UPDATED} state;

  // repeatedly used for output
  T maxdiagC;
  T mindiagC;
  T maxEW;
  T minEW;

  bool eigensysIsUptodate;
  bool doCheckEigen;
  T genOfEigensysUpdate;

  T dMaxSignifKond;

  T dLastMinEWgroesserNull;

  std::string stopMessage; //!< A message that contains all matched stop criteria.

  /**
   * Calculating eigenvalues and vectors.
   * Also checks for successful eigen decomposition.
   * @param diag (output) N eigenvalues.
   * @param Q (output) Columns are normalized eigenvectors.
   */
  void eigen(T* diag, T** Q)
  { 

     arma::vec eV;
     arma::mat eigMat;

     arma::mat cov(N,N);
     for(int i=0; i<N; i++)
      for(int j=0; j<=i; j++) cov(i,j)=cov(j,i)=C[i][j];


   if (!arma::eig_sym(eV, eigMat, cov))
        assert("eigen decomposition failed in neuro_cmaes::eigen()");

     for (int i=0; i<N; i++)
     {
      diag[i]=eV(i);

        for (int j=0; j<N; j++)
        Q[i][j]=eigMat(i,j);
      
     }
  
  }

  /** 
   * Exhaustive test of the output of the eigendecomposition, needs O(n^3)
   * operations writes to error file.
   * @return number of detected inaccuracies
   */
  int checkEigen(T* diag, T** Q)
  {
    // compute Q diag Q^T and Q Q^T to check
    int res = 0;
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j) {
        T cc = 0., dd = 0.;
        for(int k = 0; k < N; ++k)
        {
          cc += diag[k]*Q[i][k]*Q[j][k];
          dd += Q[i][k]*Q[j][k];
        }
        // check here, is the normalization the right one?
        const bool cond1 = fabs(cc - C[i > j ? i : j][i > j ? j : i]) / sqrt(C[i][i]* C[j][j]) > T(1e-10);
        const bool cond2 = fabs(cc - C[i > j ? i : j][i > j ? j : i]) > T(3e-14);
        if(cond1 && cond2)
        {
          std::stringstream s;
          s << i << " " << j << ": " << cc << " " << C[i > j ? i : j][i > j ? j : i]
              << ", " << cc - C[i > j ? i : j][i > j ? j : i];
          if(logWarnings)
            logStream << "eigen(): imprecise result detected " << s.str()
                << std::endl;
          ++res;
        }
        if(std::fabs(dd - (i == j)) > T(1e-10))
        {
          std::stringstream s;
          s << i << " " << j << " " << dd;
          if(logWarnings)
            logStream << "eigen(): imprecise result detected (Q not orthog.)"
                << s.str() << std::endl;
          ++res;
        }
      }
    return res;
  }


  void sortIndex(const T* rgFunVal, int* iindex, int n)
  {
    int i, j;
    for(i = 1, iindex[0] = 0; i < n; ++i)
    {
      for(j = i; j > 0; --j)
      {
        if(rgFunVal[iindex[j - 1]] < rgFunVal[i])
          break;
        iindex[j] = iindex[j - 1];
      }
      iindex[j] = i;
    }
  }

  void adaptC2(const int hsig)
  {
    bool diag = diagonalCov == 1 || diagonalCov >= gen;

    if(ccov != T(0))
    {
      // definitions for speeding up inner-most loop
      const T mucovinv = T(1)/mucov;
      const T commonFactor = ccov * (diag ? (N + T(1.5)) / T(3) : T(1));
      const T ccov1 = std::min(commonFactor*mucovinv, T(1));
      const T ccovmu = std::min(commonFactor*(T(1)-mucovinv), T(1)-ccov1);
      const T sigmasquare = sigma*sigma;
      const T onemccov1ccovmu = T(1)-ccov1-ccovmu;
      const T longFactor = (T(1)-hsig)*ccumcov*(T(2)-ccumcov);

      eigensysIsUptodate = false;

      // update covariance matrix
      for(int i = 0; i < N; ++i)
        for(int j = diag ? i : 0; j <= i; ++j)
        {
          T& Cij = C[i][j];
          Cij = onemccov1ccovmu*Cij + ccov1 * (pc[i]*pc[j] + longFactor*Cij);
          for(int k = 0; k < mu; ++k)
          { // additional rank mu update
            const T* rgrgxindexk = population[index[k]];
            Cij += ccovmu*weights[k] * (rgrgxindexk[i] - xold[i])
                * (rgrgxindexk[j] - xold[j]) / sigmasquare;
          }
        }
      // update maximal and minimal diagonal value
      maxdiagC = mindiagC = C[0][0];
      for(int i = 1; i < N; ++i)
      {
        const T& Cii = C[i][i];
        if(maxdiagC < Cii)
          maxdiagC = Cii;
        else if(mindiagC > Cii)
          mindiagC = Cii;
      }
    }
  }

  /**
   * Treats minimal standard deviations and numeric problems. Increases sigma.
   */
  void testMinStdDevs(void)
  {
    if(!this->rgDiffMinChange)
      return;

    for(int i = 0; i < N; ++i)
      while(this->sigma*std::sqrt(this->C[i][i]) < this->rgDiffMinChange[i])
        this->sigma *= std::exp(T(0.05) + this->cs / this->damps);
  }

  /**
   * Adds the mutation sigma*B*(D*z).
   * @param x Search space vector.
   * @param eps Mutation factor.
   */
  void addMutation(T* x, T eps = 1.0)
  {
    for(int i = 0; i < N; ++i)
      tempRandom[i] = rgD[i]*rand.gauss();
    for(int i = 0; i < N; ++i)
    {
      T sum = 0.0;
      for(int j = 0; j < N; ++j)
        sum += B[i][j]*tempRandom[j];
      x[i] = xmean[i] + eps*sigma*sum;
    }
  }


public:

  T countevals; //!< objective function evaluations

  /**
   * Free the memory.
   */
  ~CMAES()
  {
    delete[] pc;
    delete[] ps;
    delete[] tempRandom;
    delete[] BDz;
    delete[] --xmean;
    delete[] --xold;
    delete[] --xBestEver;
    delete[] --output;
    delete[] rgD;
    for(int i = 0; i < N; ++i)
    {
      delete[] C[i];
      delete[] B[i];
    }
    for(int i = 0; i < lambda; ++i)
      delete[] --population[i];
    delete[] population;
    delete[] C;
    delete[] B;
    delete[] index;
    delete[] publicFitness;
    delete[] --functionValues;
    delete[] --funcValueHistory;

    if(xstart)
      delete[] xstart;
    if(typicalX)
      delete[] typicalX;
    if(rgInitialStds)
      delete[] rgInitialStds;
    if(rgDiffMinChange)
      delete[] rgDiffMinChange;
    if(weights)
      delete[] weights;
  }

  /**
   * Initializes the CMA-ES algorithm.
   * @param parameters The CMA-ES parameters in the parameters.h file
   * @return Array of size lambda that can be used to assign fitness values and
   *         pass them to updateDistribution()
   */
  T* init()
  {

    stopMessage = "";

    T trace(0);
    for(int i = 0; i < N; ++i)
      trace += rgInitialStds[i]*rgInitialStds[i];
    sigma = std::sqrt(trace/N);

    chiN = std::sqrt((T) N) * (T(1) - T(1)/(T(4)*N) + T(1)/(T(21)*N*N));
    eigensysIsUptodate = true;
    doCheckEigen = false;
    genOfEigensysUpdate = 0;

    T dtest;
    for(dtest = T(1); dtest && dtest < T(1.1)*dtest; dtest *= T(2))
      if(dtest == dtest + T(1))
        break;
    dMaxSignifKond = dtest / T(1000);

    gen = 0;
    countevals = 0;
    state = INITIALIZED;
    dLastMinEWgroesserNull = T(1);

    pc = new T[N];
    ps = new T[N];
    tempRandom = new T[N+1];
    BDz = new T[N];
    xmean = new T[N+2];
    xmean[0] = N;
    ++xmean;
    xold = new T[N+2];
    xold[0] = N;
    ++xold;
    xBestEver = new T[N+3];
    xBestEver[0] = N;
    ++xBestEver;
    xBestEver[N] = std::numeric_limits<T>::max();
    output = new T[N+2];
    output[0] = N;
    ++output;
    rgD = new T[N];
    C = new T*[N];
    B = new T*[N];
    publicFitness = new T[lambda];
    functionValues = new T[lambda+1];
    functionValues[0] = lambda;
    ++functionValues;
    const int historySize = 10 + (int) ceil(3.*10.*N/lambda);
    funcValueHistory = new T[historySize + 1];
    funcValueHistory[0] = (T) historySize;
    funcValueHistory++;

    for(int i = 0; i < N; ++i)
    {
      C[i] = new T[i+1];
      B[i] = new T[N];
    }
    index = new int[lambda];
    for(int i = 0; i < lambda; ++i)
        index[i] = i;
    population = new T*[lambda];
    for(int i = 0; i < lambda; ++i)
    {
      population[i] = new T[N+2];
      population[i][0] = N;
      population[i]++;
      for(int j = 0; j < N; j++)
        population[i][j] = 0.0;
    }

    for(int i = 0; i < lambda; i++)
    {
      functionValues[i] = std::numeric_limits<T>::max();
    }
    for(int i = 0; i < historySize; i++)
    {
      funcValueHistory[i] = std::numeric_limits<T>::max();
    }
    for(int i = 0; i < N; ++i)
      for(int j = 0; j < i; ++j)
        C[i][j] = B[i][j] = B[j][i] = 0.;

    for(int i = 0; i < N; ++i)
    {
      B[i][i] = T(1);
      C[i][i] = rgD[i] = rgInitialStds[i]*std::sqrt(N/trace);
      C[i][i] *= C[i][i];
      pc[i] = ps[i] = T(0);
    }
    minEW = minElement(rgD, N);
    minEW = minEW*minEW;
    maxEW = maxElement(rgD, N);
    maxEW = maxEW*maxEW;

    maxdiagC = C[0][0];
    for(int i = 1; i < N; ++i) if(maxdiagC < C[i][i]) maxdiagC = C[i][i];
    mindiagC = C[0][0];
    for(int i = 1; i < N; ++i) if(mindiagC > C[i][i]) mindiagC = C[i][i];

    for(int i = 0; i < N; ++i)
      xmean[i] = xold[i] = xstart[i];
    
    if(typicalXcase)
      for(int i = 0; i < N; ++i)
        xmean[i] += sigma*rgD[i]*rand.gauss();

    return publicFitness;
  }

	T maxElement(const T* rgd, int len)
	{
	  return *std::max_element(rgd, rgd + len);
	}

	T minElement(const T* rgd, int len)
	{
	  return *std::min_element(rgd, rgd + len);
	}

  /**
   * The search space vectors have a special form: they are arrays with N+1
   * entries. Entry number -1 is the dimension of the search space N.
   * @return A pointer to a "population" of lambda N-dimensional multivariate
   * normally distributed samples.
   */
  T* const* samplePopulation()
  {
    bool diag = diagonalCov == 1 || diagonalCov >= gen;

    // calculate eigensystem
    if(!eigensysIsUptodate)
    {
      if(!diag)
        updateEigensystem(false);
      else
      {
        for(int i = 0; i < N; ++i)
          rgD[i] = std::sqrt(C[i][i]);
        minEW = minElement(rgD, N);
        minEW *= minEW;
        maxEW = maxElement(rgD, N);
        maxEW *= maxEW;
        eigensysIsUptodate = true;
      }
    }

    testMinStdDevs();

    for(int iNk = 0; iNk < lambda; ++iNk)
    { // generate scaled random vector D*z
      T* rgrgxink = population[iNk];
      for(int i = 0; i < N; ++i)
        if(diag)
          rgrgxink[i] = xmean[i] + sigma*rgD[i]*rand.gauss();
        else
          tempRandom[i] = rgD[i]*rand.gauss();
      if(!diag)
        for(int i = 0; i < N; ++i) // add mutation sigma*B*(D*z)
        {
          T sum = 0.0;
          for(int j = 0; j < N; ++j)
            sum += B[i][j]*tempRandom[j];
          rgrgxink[i] = xmean[i] + sigma*sum;
        }
    }

    if(state == UPDATED || gen == 0)
      ++gen;
    state = SAMPLED;

    return population;
  }

  /**
   * Can be called after samplePopulation() to resample single solutions of the
   * population as often as desired. Useful to implement a box constraints
   * (boundary) handling.
   * @param i Index to an element of the returned value of samplePopulation().
   *          population[index] will be resampled where \f$0\leq i<\lambda\f$
   *          must hold.
   * @return A pointer to the resampled "population".
   */
  T* const* reSampleSingle(int i)
  {
    T* x;
    assert(i >= 0 && i < lambda &&
        "reSampleSingle(): index must be between 0 and sp.lambda");
    x = population[i];
    addMutation(x);
    return population;
  }

  /**
   * Can be called after samplePopulation() to resample single solutions. In
   * general, the function can be used to sample as many independent
   * mean+sigma*Normal(0,C) distributed vectors as desired.
   *
   * Input x can be a pointer to an element of the vector returned by
   * samplePopulation() but this is inconsistent with the const qualifier of the
   * returned value and therefore rather reSampleSingle() should be used.
   * @param x Solution vector that gets sampled a new value. If x == NULL new
   *          memory is allocated and must be released by the user using
   *          delete[].
   * @return A pointer to the resampled solution vector, equals input x for
   *         x != NULL on input.
   */
  T* sampleSingleInto(T* x)
  {
    if(!x)
      x = new T[N];
    addMutation(x);
    return x;
  }

  /**
   * Can be called after samplePopulation() to resample single solutions. In
   * general, the function can be used to sample as many independent
   * mean+sigma*Normal(0,C) distributed vectors as desired.
   * @param x Element of the return value of samplePopulation(), that is
   *          pop[0..\f$\lambda\f$]. This solution vector of the population gets
   *          sampled a new value.
   * @return A pointer to the resampled "population" member.
   */
  T const* reSampleSingleOld(T* x)
  {
    assert(x && "reSampleSingleOld(): Missing input x");
    addMutation(x);
    return x;
  }

  /**
   * Used to reevaluate a slightly disturbed solution for an uncertaintly
   * measurement. In case if x == NULL on input, the memory of the returned x
   * must be released.
   * @param x Solution vector that gets sampled a new value. If x == NULL new
   *          memory is allocated and must be released by the user using
   *          delete[] x.
   * @param pxmean Mean vector \f$\mu\f$ for perturbation.
   * @param eps Scale factor \f$\epsilon\f$ for perturbation:
   *            \f$x \sim \mu + \epsilon \sigma N(0,C)\f$.
   * @return A pointer to the perturbed solution vector, equals input x for
   *         x != NULL.
   */
  T* perturbSolutionInto(T* x, T const* pxmean, T eps)
  {
    if(!x)
      x = new T[N];
    assert(pxmean && "perturbSolutionInto(): pxmean was not given");
    addMutation(x, eps);
    return x;
  }

  /**
   * Core procedure of the CMA-ES algorithm. Sets a new mean value and estimates
   * the new covariance matrix and a new step size for the normal search
   * distribution.
   * @param fitnessValues An array of \f$\lambda\f$ function values.
   * @return Mean value of the new distribution.
   */
  T* updateDistribution(const T* fitnessValues)
  {

    bool diag = diagonalCov == 1 || diagonalCov >= gen;

    assert(state != UPDATED && "updateDistribution(): You need to call "
          "samplePopulation() before update can take place.");
    assert(fitnessValues && "updateDistribution(): No fitness function value array input.");

    if(state == SAMPLED) // function values are delivered here
      countevals += lambda;
    else std::cout<<  "updateDistribution(): unexpected state" << std::endl;

    // assign function values
    for(int i = 0; i < lambda; ++i)
      population[i][N] = functionValues[i] = fitnessValues[i];

    // Generate index
    sortIndex(fitnessValues, index, lambda);

    // Test if function values are identical, escape flat fitness
    if(fitnessValues[index[0]] == fitnessValues[index[(int) lambda / 2]])
    {
      sigma *= std::exp(T(0.2) + cs / damps);
     
        std::cout << "Warning: sigma increased due to equal function values"
         << std::endl << "   Reconsider the formulation of the objective function";
  
    }

    // update function value history
    for(int i = (int) *(funcValueHistory - 1) - 1; i > 0; --i)
      funcValueHistory[i] = funcValueHistory[i - 1];
    funcValueHistory[0] = fitnessValues[index[0]];

    // update xbestever
    if(xBestEver[N] > population[index[0]][N] || gen == 1)
      for(int i = 0; i <= N; ++i)
      {
        xBestEver[i] = population[index[0]][i];
        xBestEver[N+1] = countevals;
      }

    const T sqrtmueffdivsigma = std::sqrt(mueff) / sigma;
    // calculate xmean and rgBDz~N(0,C)
    for(int i = 0; i < N; ++i)
    {
      xold[i] = xmean[i];
      xmean[i] = 0.;
      for(int iNk = 0; iNk < mu; ++iNk)
        xmean[i] += weights[iNk]*population[index[iNk]][i];
      BDz[i] = sqrtmueffdivsigma*(xmean[i]-xold[i]);
    }

    // calculate z := D^(-1)* B^(-1)* rgBDz into rgdTmp
    for(int i = 0; i < N; ++i)
    {
      T sum;
      if(diag)
        sum = BDz[i];
      else
      {
        sum = 0.;
        for(int j = 0; j < N; ++j)
          sum += B[j][i]*BDz[j];
      }
      tempRandom[i] = sum/rgD[i];
    }

    // cumulation for sigma (ps) using B*z
    const T sqrtFactor = std::sqrt(cs*(T(2)-cs));
    const T invps = T(1)-cs;
    for(int i = 0; i < N; ++i)
    {
      T sum;
      if(diag)
        sum = tempRandom[i];
      else
      {
        sum = T(0);
        T* Bi = B[i];
        for(int j = 0; j < N; ++j)
          sum += Bi[j]*tempRandom[j];
      }
      ps[i] = invps*ps[i] + sqrtFactor*sum;
    }

    // calculate norm(ps)^2
    T psxps(0);
    for(int i = 0; i < N; ++i)
    {
      const T& rgpsi = ps[i];
      psxps += rgpsi*rgpsi;
    }

    // cumulation for covariance matrix (pc) using B*D*z~N(0,C)
    int hsig = std::sqrt(psxps) / std::sqrt(T(1) - std::pow(T(1) - cs, T(2)* gen))
        / chiN < T(1.4) + T(2) / (N + 1);
    const T ccumcovinv = 1.-ccumcov;
    const T hsigFactor = hsig*std::sqrt(ccumcov*(T(2)-ccumcov));
    for(int i = 0; i < N; ++i)
      pc[i] = ccumcovinv*pc[i] + hsigFactor*BDz[i];

    // update of C
    adaptC2(hsig);

    // update of sigma
    sigma *= std::exp(((std::sqrt(psxps) / chiN) - T(1))* cs / damps);

    state = UPDATED;
    return xmean;
  }

    /**
   * Request a scalar parameter from CMA-ES.
   * @param key Key of the requested scalar.
   * @return The desired value.
   */
  T get(GetScalar key)
  {
    switch(key)
    {
      case AxisRatio:
        return maxElement(rgD, N) / minElement(rgD, N);
      case Eval:
        return countevals;
      case Fitness:
        return functionValues[index[0]];
      case FBestEver:
        return xBestEver[N];
      case Generation:
        return gen;
      case MaxEval:
        return stopMaxFunEvals;
      case MaxIter:
        return std::ceil(stopMaxIter);
      case MaxAxisLength:
        return sigma*std::sqrt(maxEW);
      case MinAxisLength:
        return sigma*std::sqrt(minEW);
      case MaxStdDev:
        return sigma*std::sqrt(maxdiagC);
      case MinStdDev:
        return sigma*std::sqrt(mindiagC);
      case Dimension:
        return N;
      case SampleSize:
        return lambda;
      case Sigma:
        return sigma;
      default:
        throw std::runtime_error("get(): No match found for key");
    }
  }

  /**
   * Request a vector parameter from CMA-ES.
   * @param key Key of the requested vector.
   * @return Pointer to the desired value array. Its content might be
   *         overwritten during the next call to any member functions other
   *         than get().
   */
  const T* getPtr(GetVector key)
  {
    switch(key)
    {
      case DiagC:
      {
        for(int i = 0; i < N; ++i)
          output[i] = C[i][i];
        return output;
      }
      case DiagD:
        return rgD;
      case StdDev:
      {
        for(int i = 0; i < N; ++i)
          output[i] = sigma*std::sqrt(C[i][i]);
        return output;
      }
      case XBestEver:
        return xBestEver;
      case XBest:
        return population[index[0]];
      case XMean:
        return xmean;
      default:
        throw std::runtime_error("getPtr(): No match found for key");
    }
  }

  /**
   * Request a vector parameter from CMA-ES.
   * @param key Key of the requested vector.
   * @return Pointer to the desired value array with unlimited reading and
   *         writing access to its elements. The memory must be explicitly
   *         released using delete[].
   */
  T* getNew(GetVector key)
  {
    return getInto(key, 0);
  }

  /**
   * Request a vector parameter from CMA-ES.
   * @param key Key of the requested vector.
   * @param res Memory of size N == dimension, where the desired values are
   *            written into. For mem == NULL new memory is allocated as with
   *            calling getNew() and must be released by the user at some point.
   */
  T* getInto(GetVector key, T* res)
  {
    T const* res0 = getPtr(key);
    if(!res)
      res = new T[N];
    for(int i = 0; i < N; ++i)
      res[i] = res0[i];
    return res;
  }

  /**
   * Some stopping criteria can be set in initials.par, with names starting
   * with stop... Internal stopping criteria include a maximal condition number
   * of about 10^15 for the covariance matrix and situations where the numerical
   * discretisation error in x-space becomes noticeably. You can get a message
   * that contains the matched stop criteria via getStopMessage().
   * @return Does any stop criterion match?
   */
  bool testForTermination()
  {
    T range, fac;
    int iAchse, iKoo;
    int diag = diagonalCov == 1 || diagonalCov >= gen;

    std::stringstream message;

    if(stopMessage != "")
    {
      message << stopMessage << std::endl;
    }

    // function value reached
    if((gen > 1 || state > SAMPLED) && stStopFitness.flg &&
        functionValues[index[0]] <= stStopFitness.val)
    {
      message << "Fitness: function value " << functionValues[index[0]]
          << " <= stopFitness (" << stStopFitness.val << ")" << std::endl;
    }

    // TolFun
    range = std::max(maxElement(funcValueHistory, (int) std::min(gen, *(funcValueHistory - 1))),
        maxElement(functionValues, lambda)) -
        std::min(minElement(funcValueHistory, (int) std::min(gen, *(funcValueHistory - 1))),
        minElement(functionValues, lambda));

    if(gen > 0 && range <= stopTolFun)
    {
      message << "TolFun: function value differences " << range
          << " < stopTolFun=" << stopTolFun << std::endl;
    }

    // TolFunHist
    if(gen > *(funcValueHistory - 1))
    {
      range = maxElement(funcValueHistory, (int) *(funcValueHistory - 1))
          - minElement(funcValueHistory, (int) *(funcValueHistory - 1));
      if(range <= stopTolFunHist)
        message << "TolFunHist: history of function value changes " << range
            << " stopTolFunHist=" << stopTolFunHist << std::endl;
    }

    // TolX
    int cTemp = 0;
    for(int i = 0; i < N; ++i)
    {
      cTemp += (sigma*std::sqrt(C[i][i]) < stopTolX) ? 1 : 0;
      cTemp += (sigma*pc[i] < stopTolX) ? 1 : 0;
    }
    if(cTemp == 2*N)
    {
      message << "TolX: object variable changes below " << stopTolX << std::endl;
    }

    // TolUpX
    for(int i = 0; i < N; ++i)
    {
      if(sigma*std::sqrt(C[i][i]) > stopTolUpXFactor*rgInitialStds[i])
      {
        message << "TolUpX: standard deviation increased by more than "
            << stopTolUpXFactor << ", larger initial standard deviation recommended."
            << std::endl;
        break;
      }
    }

    // Condition of C greater than dMaxSignifKond
    if(maxEW >= minEW* dMaxSignifKond)
    {
      message << "ConditionNumber: maximal condition number " << dMaxSignifKond
          << " reached. maxEW=" << maxEW <<  ",minEW=" << minEW << ",maxdiagC="
          << maxdiagC << ",mindiagC=" << mindiagC << std::endl;
    }

    // Principal axis i has no effect on xmean, ie. x == x + 0.1* sigma* rgD[i]* B[i]
    if(!diag)
    {
      for(iAchse = 0; iAchse < N; ++iAchse)
      {
        fac = 0.1* sigma* rgD[iAchse];
        for(iKoo = 0; iKoo < N; ++iKoo)
        {
          if(xmean[iKoo] != xmean[iKoo] + fac* B[iKoo][iAchse])
            break;
        }
        if(iKoo == N)
        {
          message << "NoEffectAxis: standard deviation 0.1*" << (fac / 0.1)
              << " in principal axis " << iAchse << " without effect" << std::endl;
          break;
        }
      }
    }
    // Component of xmean is not changed anymore
    for(iKoo = 0; iKoo < N; ++iKoo)
    {
      if(xmean[iKoo] == xmean[iKoo] + sigma*std::sqrt(C[iKoo][iKoo])/T(5))
      {
        message << "NoEffectCoordinate: standard deviation 0.2*"
            << (sigma*std::sqrt(C[iKoo][iKoo])) << " in coordinate " << iKoo
            << " without effect" << std::endl;
        break;
      }
    }

    if(countevals >= stopMaxFunEvals)
    {
      message << "MaxFunEvals: conducted function evaluations " << countevals
          << " >= " << stopMaxFunEvals << std::endl;
    }
    if(gen >= stopMaxIter)
    {
      message << "MaxIter: number of iterations " << gen << " >= "
          << stopMaxIter << std::endl;
    }

    stopMessage = message.str();
    return stopMessage != "";
  }

  /**
   * A message that contains a detailed description of the matched stop
   * criteria.
   */
  std::string getStopMessage()
  {
    return stopMessage;
  }


  /**
   * Conducts the eigendecomposition of C into B and D such that
   * \f$C = B \cdot D \cdot D \cdot B^T\f$ and \f$B \cdot B^T = I\f$
   * and D diagonal and positive.
   * @param force For force == true the eigendecomposion is conducted even if
   *              eigenvector and values seem to be up to date.
   */
  void updateEigensystem(bool force)
  {
    if(!force)
    {
      if(eigensysIsUptodate)
        return;
      // return on modulo generation number
      if(gen < genOfEigensysUpdate + updateCmode.modulo)
        return;
    }

    eigen(rgD, B);

    // find largest and smallest eigenvalue, they are supposed to be sorted anyway
    minEW = minElement(rgD, N);
    maxEW = maxElement(rgD, N);

    if(doCheckEigen) // needs O(n^3)! writes, in case, error message in error file
      checkEigen(rgD, B);

    for(int i = 0; i < N; ++i)
      rgD[i] = std::sqrt(rgD[i]);

    eigensysIsUptodate = true;
    genOfEigensysUpdate = gen;
  }

  /**
   * Distribution mean could be changed before samplePopulation(). This might
   * lead to unexpected behaviour if done repeatedly.
   * @param newxmean new mean, if it is NULL, it will be set to the current mean
   * @return new mean
   */
  T const* setMean(const T* newxmean)
  {
    assert(state != SAMPLED && "setMean: mean cannot be set inbetween the calls"
        "of samplePopulation and updateDistribution");

    if(newxmean && newxmean != xmean)
      for(int i = 0; i < N; ++i)
        xmean[i] = newxmean[i];
    else
      newxmean = xmean;

    return newxmean;
  }
}; //CLASS

} //OPTIMIZER
}
#endif