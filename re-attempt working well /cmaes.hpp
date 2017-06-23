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

  /* Input parameters. */
  //! Problem dimension, must stay constant.
  int N;
  //! Initial search space vector.
  double* xstart;
  //! A typical value for a search space vector.
  double* typicalX;
  //! Indicates that the typical x is the initial point.
  bool typicalXcase;
  //! Initial standard deviations.
  double* rgInitialStds;
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


  CMAES(int dimension = 0, const double* inxstart = 0, const double* inrgsigma = 0);

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

  Random<double> rand;

  //! Step size.
  double sigma;
  //! Mean x vector, "parent".
  double* xmean;
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
  double* pc;
  //! Isotropic evolution path (for step length).
  double* ps;
  //! Last mean.
  double* xold;
  //! Output vector.
  double* output;
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

  std::string stopMessage; //!< A message that contains all matched stop criteria.

  void eigen(double* diag, double** Q);
  int  checkEigen(double* diag, double** Q);
  void sortIndex(const double* rgFunVal, int* iindex, int n);
  void adaptC2(const int hsig);
  void testMinStdDevs(void);
  void addMutation(double* x, double eps = 1.0);

public:

  double countevals; //!< objective function evaluations

	double maxElement(const double* rgd, int len)
	{
	  return *std::max_element(rgd, rgd + len);
	}

	double minElement(const double* rgd, int len)
	{
	  return *std::min_element(rgd, rgd + len);
	}
   double* init();
   double* const* samplePopulation();
   double* const* reSampleSingle(int i);
   double* sampleSingleInto(double* x);
   double const* reSampleSingleOld(double* x);
   double* perturbSolutionInto(double* x, double const* pxmean, double eps);
   double* updateDistribution(const double* fitnessValues);
   double get(GetScalar key);
   double* getNew(GetVector key);
   double* getInto(GetVector key, double* res);
   double const* setMean(const double* newxmean);
   
   const double* getPtr(GetVector key);
   bool testForTermination();
   void updateEigensystem(bool force);
     /**
   * A message that contains a detailed description of the matched stop
   * criteria.
   */
  std::string getStopMessage()
  {
    return stopMessage;
  }

  ~CMAES();

};
} // namespace optimization
} // namespace mlpack

// Include implementation.
#include "cmaes_impl.hpp"

#endif