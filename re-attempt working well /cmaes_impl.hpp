/**
 * @file cmaes_impl.hpp
 * @author Kartik Nighania (GSoC 17 mentor Marcus Edel)
 *
 * Covariance Matrix Adaptation Evolution Strategy
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_CMAES_CMAES_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_CMAES_CMAES_IMPL_HPP

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

#include "cmaes.hpp"
#include "random.hpp"

namespace mlpack {
namespace optimization {

  template<typename funcType>
  CMAES<funcType>::CMAES(funcType& function, arma::mat& start, arma::mat& stdDivs)
      : function(function),
        N(-1),
        typicalXcase(false),
        rgDiffMinChange(0),
        stopMaxFunEvals(-1),
        facmaxeval(1.0),
        stopMaxIter(-1.0),
        stopTolFun(1e-15),
        stopTolFunHist(1e-15),
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
        weightMode(UNINITIALIZED_WEIGHTS)
  {
    stStopFitness.flg = false;
    stStopFitness.val = -std::numeric_limits<double>::max();
    updateCmode.modulo = -1;
    updateCmode.maxtime = -1;

     N = function.NumFunctions();
    if ( N <= 0)
      throw std::runtime_error("Problem dimension N undefined.");
    
    bool startP  = true;
    bool initDev = true;

    for (int i=0; i<N; i++)
    {
     if (start[i]   < 1.0e-200) startP  = false;
     if (stdDivs[i] < 1.0e-200) initDev = false;
    }

  
   if (!startP)
        std::cout << " WARNING: initial start point undefined. Please specify if incorrect results detected. DEFAULT = 0.5...0.5." << std::endl;
   if (!initDev)
        std::cout << "WARNING: initialStandardDeviations undefined. Please specify if incorrect results detected. DEFAULT = 0.3...0.3." << std::endl;
    
   

    if (weightMode == UNINITIALIZED_WEIGHTS)
      weightMode = LOG_WEIGHTS;

    diagonalCov = 0; // default is 0, but this might change in future

      xstart = new double[N];
      if (startP)
      {
        for (int i = 0; i < N; ++i) xstart[i] = start[i];
      }
     else
      {
        typicalXcase = true;
        for (int i = 0; i < N; i++) xstart[i] = 0.5;
      }
  

    rgInitialStds.set_size(N);
    if (initDev)
      {
        for (int i = 0; i < N; ++i) rgInitialStds[i] = stdDivs[i];
      }
      else
      {
        for (int i = 0; i < N; ++i) rgInitialStds[i] = double(0.3);
      }


    if (lambda < 2)
      lambda = 4 + (int) (3.0*log((double) N));
    if (mu <= 0)
      mu = lambda / 2;
    if (!weights)
     {
     if (weights)
        delete[] weights;
      weights = new double[mu];
      switch(weightMode)
      {
      case LINEAR_WEIGHTS:
        for (int i = 0; i < mu; ++i) weights[i] = mu - i;
        break;
      case EQUAL_WEIGHTS:
        for (int i = 0; i < mu; ++i) weights[i] = 1;
        break;
      case LOG_WEIGHTS:
      default:
        for (int i = 0; i < mu; ++i) weights[i] = log(mu + 1.) - log(i + 1.);
        break;
      }

      // normalize weights vector and set mueff
      double s1 = 0, s2 = 0;
      for (int i = 0; i < mu; ++i)
      {
        s1 += weights[i];
        s2 += weights[i]*weights[i];
      }
      mueff = s1*s1/s2;
      for (int i = 0; i < mu; ++i)
        weights[i] /= s1;

      if (mu < 1 || mu > lambda || (mu == lambda && weights[0] == weights[mu - 1]))
        throw std::runtime_error("setWeights(): invalid setting of mu or lambda");
    }

    if (cs > 0)
      cs *= (mueff + 2.) / (N + mueff + 3.);
    if (cs <= 0 || cs >= 1)
      cs = (mueff + 2.) / (N + mueff + 3.);

    if (ccumcov <= 0 || ccumcov > 1)
      ccumcov = 4. / (N + 4);

    if (mucov < 1)
      mucov = mueff;
    double t1 = 2. / ((N + 1.4142)*(N + 1.4142));
    double t2 = (2.* mueff - 1.) / ((N + 2.)*(N + 2.) + mueff);
    t2 = (t2 > 1) ? 1 : t2;
    t2 = (1. / mucov)* t1 + (1. - 1. / mucov)* t2;
    if (ccov >= 0)
      ccov *= t2;
    if (ccov < 0 || ccov > 1)
      ccov = t2;

    if (diagonalCov < 0)
      diagonalCov = 2 + 100. * N / sqrt((double) lambda);

    if (stopMaxFunEvals <= 0)
      stopMaxFunEvals = facmaxeval * 900 * (N + 3)*(N + 3);
    else
      stopMaxFunEvals *= facmaxeval;

    if (stopMaxIter <= 0)
      stopMaxIter = ceil((double) (stopMaxFunEvals / lambda));

    if (damps < double(0))
      damps = double(1);
    damps = damps
        * (double(1) + double(2)*std::max(double(0), std::sqrt((mueff - double(1)) / (N + double(1))) - double(1)))
        * (double) std::max(double(0.3), double(1) - // modify for short runs
          (double) N / (double(1e-6) + std::min(stopMaxIter, stopMaxFunEvals / lambda)))
        + cs;

    if (updateCmode.modulo < 0)
      updateCmode.modulo = 1. / ccov / (double) N / 10.;
    updateCmode.modulo *= facupdateCmode;
    if (updateCmode.maxtime < 0)
      updateCmode.maxtime = 0.20; // maximal 20% of CPU-time
  
  }

  template<typename funcType>
  double CMAES<funcType>::Optimize(arma::mat& arr)
  {

  arFunvals = init();

  while(!testForTermination())
  {
    // Generate lambda new search points, sample population
    pop = samplePopulation();

    arma::mat fit(1,N);

    // evaluate the new search points using the given evaluate function by the user
    for (int i = 0; i < lambda; ++i)
    {
      for (int j=0; j<N; j++) fit(0,j) = pop[i][j];

      arFunvals[i] = function.Evaluate(fit);
    }

    // update the search distribution used for sampleDistribution()
      updateDistribution(arFunvals);
  }

  std::cout << "Stop:" << std::endl << getStopMessage();

  // get best estimator for the optimum
  for (int i=0; i<N; i++) arr[i] = xmean[i]; 

  return xBestEver[N];

  }

/**
   * Calculating eigenvalues and vectors.
   * Also checks for successful eigen decomposition.
   * @param diag (output) N eigenvalues.
   * @param Q (output) Columns are normalized eigenvectors.
   */
   template<typename funcType>
  void CMAES<funcType>::eigen(double* diag, double** Q)
  { 

     arma::vec eV;
     arma::mat eigMat;

     arma::mat cov(N,N);
     for (int i=0; i<N; i++)
      for (int j=0; j<=i; j++) cov(i,j)=cov(j,i)=C[i][j];


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
   template<typename funcType>
  int CMAES<funcType>::checkEigen(double* diag, double** Q)
  {
    // compute Q diag Q^T and Q Q^T to check
    int res = 0;
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j) {
        double cc = 0., dd = 0.;
        for (int k = 0; k < N; ++k)
        {
          cc += diag[k]*Q[i][k]*Q[j][k];
          dd += Q[i][k]*Q[j][k];
        }
        // check here, is the normalization the right one?
        const bool cond1 = fabs(cc - C[i > j ? i : j][i > j ? j : i]) / sqrt(C[i][i]* C[j][j]) > double(1e-10);
        const bool cond2 = fabs(cc - C[i > j ? i : j][i > j ? j : i]) > double(3e-14);
        if (cond1 && cond2)
        {
          std::stringstream s;
          s << i << " " << j << ": " << cc << " " << C[i > j ? i : j][i > j ? j : i]
              << ", " << cc - C[i > j ? i : j][i > j ? j : i];
          
          std::cout << "eigen(): imprecise result detected " << s.str()
                << std::endl;
          ++res;
        }
        if (std::fabs(dd - (i == j)) > double(1e-10))
        {
          std::stringstream s;
          s << i << " " << j << " " << dd;
          
            std::cout << "eigen(): imprecise result detected (Q not orthog.)"
                << s.str() << std::endl;
          ++res;
        }
      }
    return res;
  }


   template<typename funcType>
  void CMAES<funcType>::sortIndex(const double* rgFunVal, int* iindex, int n)
  {
    int i, j;
    for (i = 1, iindex[0] = 0; i < n; ++i)
    {
      for (j = i; j > 0; --j)
      {
        if (rgFunVal[iindex[j - 1]] < rgFunVal[i])
          break;
        iindex[j] = iindex[j - 1];
      }
      iindex[j] = i;
    }
  }

  template<typename funcType>
  void CMAES<funcType>::adaptC2(const int hsig)
  {
    bool diag = diagonalCov == 1 || diagonalCov >= gen;

    if (ccov != double(0))
    {
      // definitions for speeding up inner-most loop
      const double mucovinv = double(1)/mucov;
      const double commonFactor = ccov * (diag ? (N + double(1.5)) / double(3) : double(1));
      const double ccov1 = std::min(commonFactor*mucovinv, double(1));
      const double ccovmu = std::min(commonFactor*(double(1)-mucovinv), double(1)-ccov1);
      const double sigmasquare = sigma*sigma;
      const double onemccov1ccovmu = double(1)-ccov1-ccovmu;
      const double longFactor = (double(1)-hsig)*ccumcov*(double(2)-ccumcov);

      eigensysIsUptodate = false;

      // update covariance matrix
      for (int i = 0; i < N; ++i)
        for (int j = diag ? i : 0; j <= i; ++j)
        {
          double& Cij = C[i][j];
          Cij = onemccov1ccovmu*Cij + ccov1 * (pc[i]*pc[j] + longFactor*Cij);
          for (int k = 0; k < mu; ++k)
          { // additional rank mu update
            const double* rgrgxindexk = population[index[k]];
            Cij += ccovmu*weights[k] * (rgrgxindexk[i] - xold[i])
                * (rgrgxindexk[j] - xold[j]) / sigmasquare;
          }
        }
      // update maximal and minimal diagonal value
      maxdiagC = mindiagC = C[0][0];
      for (int i = 1; i < N; ++i)
      {
        const double& Cii = C[i][i];
        if (maxdiagC < Cii)
          maxdiagC = Cii;
        else if (mindiagC > Cii)
          mindiagC = Cii;
      }
    }
  }

  /**
   * Treats minimal standard deviations and numeric problems. Increases sigma.
   */
  template<typename funcType>
  void CMAES<funcType>::testMinStdDevs(void)
  {
    if (!this->rgDiffMinChange)
      return;

    for (int i = 0; i < N; ++i)
      while(this->sigma*std::sqrt(this->C[i][i]) < this->rgDiffMinChange[i])
        this->sigma *= std::exp(double(0.05) + this->cs / this->damps);
  }

  /**
   * Adds the mutation sigma*B*(D*z).
   * @param x Search space vector.
   * @param eps Mutation factor.
   */
  template<typename funcType>
  void CMAES<funcType>::addMutation(double* x, double eps)
  {
    for (int i = 0; i < N; ++i)
      tempRandom[i] = rgD[i]*rand.gauss();
    for (int i = 0; i < N; ++i)
    {
      double sum = 0.0;
      for (int j = 0; j < N; ++j)
        sum += B[i][j]*tempRandom[j];
      x[i] = xmean[i] + eps*sigma*sum;
    }
  }

  /**
   * Initializes the CMA-ES algorithm.
   * @param parameters The CMA-ES parameters in the parameters.h file
   * @return Array of size lambda that can be used to assign fitness values and
   *         pass them to updateDistribution()
   */
  template<typename funcType>
  double* CMAES<funcType>::init()
  {

    stopMessage = "";

    double trace = arma::accu(arma::pow(rgInitialStds, 2));
    sigma = std::sqrt(trace/N);

    chiN = std::sqrt((double) N) * (1 - 1/(4*N) + 1/(21*N*N));
    eigensysIsUptodate = true;
    doCheckEigen = false;
    genOfEigensysUpdate = 0;

    double dtest;
    for (dtest = double(1); dtest && dtest < double(1.1)*dtest; dtest *= double(2))
      if (dtest == dtest + double(1))
        break;
    dMaxSignifKond = dtest / double(1000);

    gen = 0;
    countevals = 0;
    state = INITIALIZED;
    dLastMinEWgroesserNull = double(1);

    pc = new double[N];
    ps = new double[N];
    tempRandom = new double[N+1];
    BDz = new double[N];
    xmean.set_size(N+2);
    xmean[0] = N;
    ++xmean;
    xold = new double[N+2];
    xold[0] = N;
    ++xold;
    xBestEver = new double[N+3];
    xBestEver[0] = N;
    ++xBestEver;
    xBestEver[N] = std::numeric_limits<double>::max();
    output = new double[N+2];
    output[0] = N;
    ++output;
    rgD = new double[N];
    C = new double*[N];
    B = new double*[N];
    publicFitness = new double[lambda];
    functionValues = new double[lambda+1];
    functionValues[0] = lambda;
    ++functionValues;
    const int historySize = 10 + (int) ceil(3.*10.*N/lambda);
    funcValueHistory = new double[historySize + 1];
    funcValueHistory[0] = (double) historySize;
    funcValueHistory++;

    for (int i = 0; i < N; ++i)
    {
      C[i] = new double[i+1];
      B[i] = new double[N];
    }
    index = new int[lambda];
    for (int i = 0; i < lambda; ++i)
        index[i] = i;
    population = new double*[lambda];
    for (int i = 0; i < lambda; ++i)
    {
      population[i] = new double[N+2];
      population[i][0] = N;
      population[i]++;
      for (int j = 0; j < N; j++)
        population[i][j] = 0.0;
    }

    for (int i = 0; i < lambda; i++)
    {
      functionValues[i] = std::numeric_limits<double>::max();
    }
    for (int i = 0; i < historySize; i++)
    {
      funcValueHistory[i] = std::numeric_limits<double>::max();
    }
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < i; ++j)
        C[i][j] = B[i][j] = B[j][i] = 0.;

    for (int i = 0; i < N; ++i)
    {
      B[i][i] = double(1);
      C[i][i] = rgD[i] = rgInitialStds[i]*std::sqrt(N/trace);
      C[i][i] *= C[i][i];
      pc[i] = ps[i] = double(0);
    }
    minEW = minElement(rgD, N);
    minEW = minEW*minEW;
    maxEW = maxElement(rgD, N);
    maxEW = maxEW*maxEW;

    maxdiagC = C[0][0];
    for (int i = 1; i < N; ++i) if (maxdiagC < C[i][i]) maxdiagC = C[i][i];
    mindiagC = C[0][0];
    for (int i = 1; i < N; ++i) if (mindiagC > C[i][i]) mindiagC = C[i][i];

    for (int i = 0; i < N; ++i)
      xmean[i] = xold[i] = xstart[i];
    
    if (typicalXcase)
      for (int i = 0; i < N; ++i)
        xmean[i] += sigma*rgD[i]*rand.gauss();

    return publicFitness;
  }

  /**
   * The search space vectors have a special form: they are arrays with N+1
   * entries. Entry number -1 is the dimension of the search space N.
   * @return A pointer to a "population" of lambda N-dimensional multivariate
   * normally distributed samples.
   */
   template<typename funcType>
  double* const* CMAES<funcType>::samplePopulation()
  {
    bool diag = diagonalCov == 1 || diagonalCov >= gen;

    // calculate eigensystem
    if (!eigensysIsUptodate)
    {
      if (!diag)
        updateEigensystem(false);
      else
      {
        for (int i = 0; i < N; ++i)
          rgD[i] = std::sqrt(C[i][i]);
        minEW = minElement(rgD, N);
        minEW *= minEW;
        maxEW = maxElement(rgD, N);
        maxEW *= maxEW;
        eigensysIsUptodate = true;
      }
    }

    testMinStdDevs();

    for (int iNk = 0; iNk < lambda; ++iNk)
    { // generate scaled random vector D*z
      double* rgrgxink = population[iNk];
      for (int i = 0; i < N; ++i)
        if (diag)
          rgrgxink[i] = xmean[i] + sigma*rgD[i]*rand.gauss();
        else
          tempRandom[i] = rgD[i]*rand.gauss();
      if (!diag)
        for (int i = 0; i < N; ++i) // add mutation sigma*B*(D*z)
        {
          double sum = 0.0;
          for (int j = 0; j < N; ++j)
            sum += B[i][j]*tempRandom[j];
          rgrgxink[i] = xmean[i] + sigma*sum;
        }
    }

    if (state == UPDATED || gen == 0)
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
   template<typename funcType>
  double* const* CMAES<funcType>::reSampleSingle(int i)
  {
    double* x;
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
   template<typename funcType>
  double* CMAES<funcType>::sampleSingleInto(double* x)
  {
    if (!x)
      x = new double[N];
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
   template<typename funcType>
  double const* CMAES<funcType>::reSampleSingleOld(double* x)
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
   template<typename funcType>
  double* CMAES<funcType>:: perturbSolutionInto(double* x, double const* pxmean, double eps)
  {
    if (!x)
      x = new double[N];
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
   template<typename funcType>
  void CMAES<funcType>::updateDistribution(const double* fitnessValues)
  {

    bool diag = diagonalCov == 1 || diagonalCov >= gen;

    assert(state != UPDATED && "updateDistribution(): You need to call "
          "samplePopulation() before update can take place.");
    assert(fitnessValues && "updateDistribution(): No fitness function value array input.");

    if (state == SAMPLED) // function values are delivered here
      countevals += lambda;
    else std::cout<<  "updateDistribution(): unexpected state" << std::endl;

    // assign function values
    for (int i = 0; i < lambda; ++i)
      population[i][N] = functionValues[i] = fitnessValues[i];

    // Generate index
    sortIndex(fitnessValues, index, lambda);

    // Test if function values are identical, escape flat fitness
    if (fitnessValues[index[0]] == fitnessValues[index[(int) lambda / 2]])
    {
      sigma *= std::exp(double(0.2) + cs / damps);
     
        std::cout << "Warning: sigma increased due to equal function values"
         << std::endl << "   Reconsider the formulation of the objective function";
  
    }

    // update function value history
    for (int i = (int) *(funcValueHistory - 1) - 1; i > 0; --i)
      funcValueHistory[i] = funcValueHistory[i - 1];
    funcValueHistory[0] = fitnessValues[index[0]];

    // update xbestever
    if (xBestEver[N] > population[index[0]][N] || gen == 1)
      for (int i = 0; i <= N; ++i)
      {
        xBestEver[i] = population[index[0]][i];
        xBestEver[N+1] = countevals;
      }

    const double sqrtmueffdivsigma = std::sqrt(mueff) / sigma;
    // calculate xmean and rgBDz~N(0,C)
    for (int i = 0; i < N; ++i)
    {
      xold[i] = xmean[i];
      xmean[i] = 0.;
      for (int iNk = 0; iNk < mu; ++iNk)
        xmean[i] += weights[iNk]*population[index[iNk]][i];
      BDz[i] = sqrtmueffdivsigma*(xmean[i]-xold[i]);
    }

    // calculate z := D^(-1)* B^(-1)* rgBDz into rgdTmp
    for (int i = 0; i < N; ++i)
    {
      double sum;
      if (diag)
        sum = BDz[i];
      else
      {
        sum = 0.;
        for (int j = 0; j < N; ++j)
          sum += B[j][i]*BDz[j];
      }
      tempRandom[i] = sum/rgD[i];
    }

    // cumulation for sigma (ps) using B*z
    const double sqrtFactor = std::sqrt(cs*(double(2)-cs));
    const double invps = double(1)-cs;
    for (int i = 0; i < N; ++i)
    {
      double sum;
      if (diag)
        sum = tempRandom[i];
      else
      {
        sum = double(0);
        double* Bi = B[i];
        for (int j = 0; j < N; ++j)
          sum += Bi[j]*tempRandom[j];
      }
      ps[i] = invps*ps[i] + sqrtFactor*sum;
    }

    // calculate norm(ps)^2
    double psxps(0);
    for (int i = 0; i < N; ++i)
    {
      const double& rgpsi = ps[i];
      psxps += rgpsi*rgpsi;
    }

    // cumulation for covariance matrix (pc) using B*D*z~N(0,C)
    int hsig = std::sqrt(psxps) / std::sqrt(double(1) - std::pow(double(1) - cs, double(2)* gen))
        / chiN < double(1.4) + double(2) / (N + 1);
    const double ccumcovinv = 1.-ccumcov;
    const double hsigFactor = hsig*std::sqrt(ccumcov*(double(2)-ccumcov));
    for (int i = 0; i < N; ++i)
      pc[i] = ccumcovinv*pc[i] + hsigFactor*BDz[i];

    // update of C
    adaptC2(hsig);

    // update of sigma
    sigma *= std::exp(((std::sqrt(psxps) / chiN) - double(1))* cs / damps);

    state = UPDATED;
  }

  /**
   * Some stopping criteria can be set in initials.par, with names starting
   * with stop... Internal stopping criteria include a maximal condition number
   * of about 10^15 for the covariance matrix and situations where the numerical
   * discretisation error in x-space becomes noticeably. You can get a message
   * that contains the matched stop criteria via getStopMessage().
   * @return Does any stop criterion match?
   */
   template<typename funcType>
   bool CMAES<funcType>::testForTermination()
  {
    double range, fac;
    int iAchse, iKoo;
    int diag = diagonalCov == 1 || diagonalCov >= gen;

    std::stringstream message;

    if (stopMessage != "")
    {
      message << stopMessage << std::endl;
    }

    // function value reached
    if ((gen > 1 || state > SAMPLED) && stStopFitness.flg &&
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

    if (gen > 0 && range <= stopTolFun)
    {
      message << "TolFun: function value differences " << range
          << " < stopTolFun=" << stopTolFun << std::endl;
    }

    // TolFunHist
    if (gen > *(funcValueHistory - 1))
    {
      range = maxElement(funcValueHistory, (int) *(funcValueHistory - 1))
          - minElement(funcValueHistory, (int) *(funcValueHistory - 1));
      if (range <= stopTolFunHist)
        message << "TolFunHist: history of function value changes " << range
            << " stopTolFunHist=" << stopTolFunHist << std::endl;
    }

    // TolX
    int cTemp = 0;
    for (int i = 0; i < N; ++i)
    {
      cTemp += (sigma*std::sqrt(C[i][i]) < stopTolX) ? 1 : 0;
      cTemp += (sigma*pc[i] < stopTolX) ? 1 : 0;
    }
    if (cTemp == 2*N)
    {
      message << "TolX: object variable changes below " << stopTolX << std::endl;
    }

    // TolUpX
    for (int i = 0; i < N; ++i)
    {
      if (sigma*std::sqrt(C[i][i]) > stopTolUpXFactor*rgInitialStds[i])
      {
        message << "TolUpX: standard deviation increased by more than "
            << stopTolUpXFactor << ", larger initial standard deviation recommended."
            << std::endl;
        break;
      }
    }

    // Condition of C greater than dMaxSignifKond
    if (maxEW >= minEW* dMaxSignifKond)
    {
      message << "ConditionNumber: maximal condition number " << dMaxSignifKond
          << " reached. maxEW=" << maxEW <<  ",minEW=" << minEW << ",maxdiagC="
          << maxdiagC << ",mindiagC=" << mindiagC << std::endl;
    }

    // Principal axis i has no effect on xmean, ie. x == x + 0.1* sigma* rgD[i]* B[i]
    if (!diag)
    {
      for (iAchse = 0; iAchse < N; ++iAchse)
      {
        fac = 0.1* sigma* rgD[iAchse];
        for (iKoo = 0; iKoo < N; ++iKoo)
        {
          if (xmean[iKoo] != xmean[iKoo] + fac* B[iKoo][iAchse])
            break;
        }
        if (iKoo == N)
        {
          message << "NoEffectAxis: standard deviation 0.1*" << (fac / 0.1)
              << " in principal axis " << iAchse << " without effect" << std::endl;
          break;
        }
      }
    }
    // Component of xmean is not changed anymore
    for (iKoo = 0; iKoo < N; ++iKoo)
    {
      if (xmean[iKoo] == xmean[iKoo] + sigma*std::sqrt(C[iKoo][iKoo])/double(5))
      {
        message << "NoEffectCoordinate: standard deviation 0.2*"
            << (sigma*std::sqrt(C[iKoo][iKoo])) << " in coordinate " << iKoo
            << " without effect" << std::endl;
        break;
      }
    }

    if (countevals >= stopMaxFunEvals)
    {
      message << "MaxFunEvals: conducted function evaluations " << countevals
          << " >= " << stopMaxFunEvals << std::endl;
    }
    if (gen >= stopMaxIter)
    {
      message << "MaxIter: number of iterations " << gen << " >= "
          << stopMaxIter << std::endl;
    }

    stopMessage = message.str();
    return stopMessage != "";
  }

  /**
   * Conducts the eigendecomposition of C into B and D such that
   * \f$C = B \cdot D \cdot D \cdot B^T\f$ and \f$B \cdot B^T = I\f$
   * and D diagonal and positive.
   * @param force For force == true the eigendecomposion is conducted even if
   *              eigenvector and values seem to be up to date.
   */
   template<typename funcType>
   void CMAES<funcType>::updateEigensystem(bool force)
  {
    if (!force)
    {
      if (eigensysIsUptodate)
        return;
      // return on modulo generation number
      if (gen < genOfEigensysUpdate + updateCmode.modulo)
        return;
    }

    eigen(rgD, B);

    // find largest and smallest eigenvalue, they are supposed to be sorted anyway
    minEW = minElement(rgD, N);
    maxEW = maxElement(rgD, N);

    if (doCheckEigen) // needs O(n^3)! writes, in case, error message in error file
      checkEigen(rgD, B);

    for (int i = 0; i < N; ++i)
      rgD[i] = std::sqrt(rgD[i]);

    eigensysIsUptodate = true;
    genOfEigensysUpdate = gen;
  }



} //namespace optimizer
} //namespace cmaes

#endif
