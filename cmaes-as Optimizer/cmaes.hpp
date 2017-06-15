/**
 * @file cmaes.h
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

#include <cmath>
#include <limits>
#include <ostream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <iostream>
#include <armadillo>
#include <iostream>

//GETTER AND SETTERS FROM 258


namespace mlpack {
namespace optimization {

/**
 * @class CMAES 
 * Holds all function prototypes that can be adjusted by the user 
 */
template<typename funcType, typename T> 
class CMAES
{
public:

    CMAES(funcType func, size_t dimension, T *start, T *stdDeviation)
  {
    int(dimension, start, stdDeviation);
  }

  size_t getDimension(void){ return N;}

  void getInitialStart(double *arr, int dimension)
  { 
    for(int i=0; i<N; i++) arr[i] = xstart[i];
  }
  
   void getInitialStandardDeviations(double *arr, int dimension)
  { 

    for(int i=0; i<N; i++) arr[i] = rgInitialStds[i];
  }

  void setWeights(Weights mode)
  {
    //if called later delete the existing ones
    delete[] weights;
    weights = new T[mu];

    switch(mode)
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
    T s1 = 0, s2 = 0;
    for (int i = 0; i < mu; ++i)
    {
      s1 += weights[i];
      s2 += weights[i] * weights[i];
    }

    mueff = s1*s1/s2;

    for (int i = 0; i < mu; ++i) weights[i] /= s1;

    if (mu < 1 || mu > lambda || (mu == lambda && weights[0] == weights[mu - 1]))
      throw std::runtime_error("setWeights(): invalid setting of mu or lambda");
  }

  //! User defined termination criterias 
  

  //GETTER AND SETTERS 
  void assign(const Parameters& p)
  {
  

    if (typicalX)
      delete[] typicalX;
    if (p.typicalX)
    {
      typicalX = new T[N];
      for (int i = 0; i < N; i++)
        typicalX[i] = p.typicalX[i];
    }

    typicalXcase = p.typicalXcase;

    if (rgInitialStds)
      delete[] rgInitialStds;
    if (p.rgInitialStds)
    {
      rgInitialStds = new T[N];
      for (int i = 0; i < N; i++)
        rgInitialStds[i] = p.rgInitialStds[i];
    }

    if (rgDiffMinChange)
      delete[] rgDiffMinChange;
    if (p.rgDiffMinChange)
    {
      rgDiffMinChange = new T[N];
      for (int i = 0; i < N; i++)
        rgDiffMinChange[i] = p.rgDiffMinChange[i];
    }

    stopMaxFunEvals = p.stopMaxFunEvals;
    facmaxeval = p.facmaxeval;
    stopMaxIter = p.stopMaxIter;

    stopTolFun = p.stopTolFun;
    stopTolFunHist = p.stopTolFunHist;
    stopTolX = p.stopTolX;
    stopTolUpXFactor = p.stopTolUpXFactor;

    lambda = p.lambda;
    mu = p.mu;
    mucov = p.mucov;
    mueff = p.mueff;
    damps = p.damps;
    cs = p.cs;
    ccumcov = p.ccumcov;
    ccov = p.ccov;
    diagonalCov = p.diagonalCov;

    updateCmode.modulo = p.updateCmode.modulo;
    updateCmode.maxtime = p.updateCmode.maxtime;

    facupdateCmode = p.facupdateCmode;

    weightMode = p.weightMode;
  }

       //N(-1),
       // xstart(0),
       // typicalX(0),
        //typicalXcase(false),
        //rgInitialStds(0),
       // rgDiffMinChange(0),
        //stopMaxFunEvals(-1),
       // facmaxeval(1.0),
       // stopMaxIter(-1.0),
        //stopTolFun(1e-12),
        //stopTolFunHist(1e-13),
        //stopTolX(0), // 1e-11*insigma would also be reasonable
       // stopTolUpXFactor(1e3),
        //lambda(-1),
        //mu(-1),
        //mucov(-1),
        //mueff(-1),
        //weights(0),
        //damps(-1),
        //cs(-1),
        //ccumcov(-1),
        //ccov(-1),
       // facupdateCmode(1),
       // weightMode(UNINITIALIZED_WEIGHTS)
       // updateCmode.modulo = -1;
      // updateCmode.maxtime = -1;


  ~CMAES()
  {
    if (xstart)
      delete[] xstart;
    if (typicalX)
      delete[] typicalX;
    if (rgInitialStds)
      delete[] rgInitialStds;
    if (rgDiffMinChange)
      delete[] rgDiffMinChange;
    if (weights)
      delete[] weights;
  }

private:

  void init(int dimension = 0, const T* inxstart = 0, const T* inrgsigma = 0)
  {
    
      if (!inxstart)
        std::cout << "Warning: initialX undefined. typicalX = 0.5...0.5." << std::endl;

      if (!inrgsigma)
        std::cout << "Warning: initialStandardDeviations undefined. 0.3...0.3." << std::endl;

      if (dimension <= 0) throw std::runtime_error("Problem dimension N undefined.");
       else 
         if (dimension > 0) N = dimension;

      diagonalCov = 0;

      xstart = new T[N];

      if (inxstart)
      {
        for (int i = 0; i < N; ++i)
          xstart[i] = inxstart[i];
      }
      else
      {
        typicalXcase = true;
        for (int i = 0; i < N; i++) xstart[i] = 0.5;
      }
  
      rgInitialStds = new T[N];

      if (inrgsigma) for (int i = 0; i < N; ++i) rgInitialStds[i] = inrgsigma[i];
      else
        for (int i = 0; i < N; ++i) rgInitialStds[i] = T(0.3);

      rgDiffMinChange(0);

      lambda = 4 + (int) (3.0*log((double) N));

      mu = lambda / 2;

      weightMode = LOG_WEIGHTS;

      setWeights(weightMode);

      if (cs <= 0 || cs >= 1) cs = (mueff + 2.) / (N + mueff + 3.);

      ccumcov = 4. / (N + 4);

      mucov = mueff = -1;

      T t1 = 2. / ((N + 1.4142)*(N + 1.4142));

      T t2 = (2.* mueff - 1.) / ((N + 2.)*(N + 2.) + mueff);
      
      t2 = (t2 > 1) ? 1 : t2;
      
      t2 = (1. / mucov)* t1 + (1. - 1. / mucov)* t2;

      ccov = t2;

      facmaxeval(1.0);
      stopMaxFunEvals = facmaxeval * 900 * (N + 3)*(N + 3);

      stopMaxIter = ceil((double) (stopMaxFunEvals / lambda));

      damps = T(1);

      damps = damps
        * (T(1) + T(2)*std::max(T(0), std::sqrt((mueff - T(1)) / (N + T(1))) - T(1)))
        * (T) std::max(T(0.3), T(1)
        - (T) N / (T(1e-6) + std::min(stopMaxIter, stopMaxFunEvals / lambda))) + cs;

      updateCmode.modulo = 1. / ccov / (double) N / 10.;

      facupdateCmode(1);
      updateCmode.modulo *= facupdateCmode;
      updateCmode.maxtime = 0.20;

      //! DEFAULT termination criterias 
      stopTolFun = 1e-12;
      stopTolFunHist = 1e-13;
      stopTolX = 0;
      stopTolUpXFactor = 1e3;

  }

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

  void stopMaxFuncEvaluations(T evaluations)
  {
     stopMaxFunEvals = evaluations;
  }

  T getStopMaxFuncEvaluations(void)
  {
    return  stopMaxFunEvals;
  }

    void stopMaxIterations(T iterations)
  {
    stopMaxIter = iterations;
  }

  T getStopMaxIterations(void)
  {
    return stopMaxIter;
  }

  

  /* Termination parameters. */
  //! Maximal number of objective function evaluations.
  T stopMaxFunEvals;
  T facmaxeval;
  //! Maximal number of iterations.
  T stopMaxIter;
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


};
} // namespace optimization
} // namespace mlpack

// Include implementation.
#include "cmaes_impl.hpp"

#endif