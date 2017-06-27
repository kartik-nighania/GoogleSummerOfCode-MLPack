/**
 * @file cmaes.hpp
 * @author Kartik Nighania (GSoC 17 mentor Marcus Edel)
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

#include "random.hpp"

namespace mlpack {
namespace optimization {

template<typename funcType>
class CMAES
{
public:

  CMAES(funcType& function, arma::mat& start, arma::mat& stdDivs);

  double Optimize(arma::mat& arr);

  size_t getDimension(void){ return N;}

  void getInitialStart(double *arr, size_t dimension)
  { 
    for (int i=0; i<N; i++) arr[i] = xstart[i];
  }
  
   void getInitialStandardDeviations(double *arr, size_t dimension)
  { 

    for (int i=0; i<N; i++) arr[i] = rgInitialStds[i];
  }

    void getStandardDeviations(double *arr)
  { 
  
    for (int i = 0; i < N; ++i)
    arr[i] = sigma*std::sqrt(C[i][i]);    
  }

  void getXBestEver(double *arr)
  { 
    for (int i = 0; i < N; ++i) arr[i] = xBestEver[i];    
  }


    void stopMaxFuncEvaluations(double evaluations)
  {
     stopMaxFunEvals = evaluations;
  }

  double getStopMaxFuncEvaluations(void)
  {
    return  stopMaxFunEvals;
  }

  void stopMaxIterations(double iterations)
  {
    stopMaxIter = iterations;
  }

  double getStopMaxIterations(void)
  {
    return stopMaxIter;
  }


  void stopMinFuntionDifference(double difference)
  {
    stopTolFun = difference;
  }

  double getStopMinFunctionDifference(void)
  {
    return stopTolFun;
  }

void stopMinFuntionHistoryDifference(double difference)
  {
    stopTolFunHist = difference;
  }

  double getStopMinFunctionHistoryDifference(void)
  {
    return stopTolFunHist;
  }

  void stopMinStepSize(double size)
  {
    stopTolX = size;
  }

  double getStopMinStepSize(void)
  {
    return stopTolX;
  }

  //! other variable parameters

void sampleSize(double l){lambda = l;}

int getSampleSize(void){ return lambda; }

void setMu(double ind){ mu = ind;}

double getMu(void){ return mu;}

void muEffective(double ind){ mueff = ind;}

double getMuEffective(void){ return mueff;}

double axisRatio() { return maxElement(rgD,N) / minElement(rgD,N);};

double evaluation(){ return countevals; }

double fitness(){ return functionValues[index[0]];}

double fitnessBestEver(){ return xBestEver[N];}

double generation(){ return gen;}

double maxAxisLength(){ return sigma*std::sqrt(maxEW);}

double minAxisLength(){ return sigma*std::sqrt(minEW); }

double maxStdDev(){return sigma*std::sqrt(maxdiagC);}

double minStdDev(){return sigma*std::sqrt(mindiagC);}

void diagonalCovariance(double *arr)
  {
     for (int i = 0; i < N; ++i) arr[i] = C[i][i];
  }

  void diagonalD(double *arr, size_t N) { for (int i = 0; i < N; ++i) arr[i] = rgD[i]; }

  void standardDeviation(double *arr, size_t N)
  {
    for (int i = 0; i < N; ++i) arr[i] = sigma*std::sqrt(C[i][i]);
  }

void getFittestMean(double *arr)
{ 
  for (int i=0; i<N; i++) arr[i] = xmean[i]; 
}

 

  double maxElement(const double* rgd, int len)
  {
    return *std::max_element(rgd, rgd + len);
  }

  double minElement(const double* rgd, int len)
  {
    return *std::min_element(rgd, rgd + len);
  }

/**
   * A message that contains a detailed description of the matched stop
   * criteria.
   */
  std::string getStopMessage(){return stopMessage;}

 /**
   * Determines the method used to initialize the weights.
   */
  enum Weights
  {
    UNINITIALIZED_WEIGHTS, LINEAR_WEIGHTS, EQUAL_WEIGHTS, LOG_WEIGHTS
  } weightMode;

  void setWeights(Weights mode);


private:

  /* Input parameters. */
  //! The instantiated function.
  funcType& function;

  double *arFunvals;

  double *const*pop;
  //! Problem dimension, must stay constant.
  int N;
  //! Initial search space vector.
  arma::vec xstart;
  //! Indicates that the typical x is the initial point.
  bool typicalXcase;
  //! Initial standard deviations.
  arma::vec rgInitialStds;
  double* rgDiffMinChange;

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

  Random<double> rand;

  //! Step size.
  double sigma;
  //! Mean x vector, "parent".
  arma::vec xmean;
  //! Best sample ever.
  double* xBestEver;
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
  double* BDz;
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
  bool doCheckEigen;
  double genOfEigensysUpdate;

  double dMaxSignifKond;

  double dLastMinEWgroesserNull;

  double countevals; //!< objective function evaluations

  std::string stopMessage; //!< A message that contains all matched stop criteria.
  
  void eigen(double* diag, double** Q);
  void updateEigensystem(bool force);
  void sortIndex(const double* rgFunVal, int* iindex, int n);
  void adaptC2(const int hsig);
  void testMinStdDevs(void);
  void addMutation(double* x, double eps = 1.0);

   double* init();
   double* const* samplePopulation();
   double* const* reSampleSingle(int i);
   double* sampleSingleInto(double* x);
   double const* reSampleSingleOld(double* x);
   double* perturbSolutionInto(double* x, double const* pxmean, double eps);
   void updateDistribution(const double* fitnessValues);

   bool testForTermination();
   int  checkEigen(double* diag, double** Q);

};
} // namespace optimization
} // namespace mlpack

// Include implementation.
#include "cmaes_impl.hpp"

#endif
