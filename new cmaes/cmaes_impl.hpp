/**
 * @file cmaes_impl.hpp
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
#ifndef MLPACK_CORE_OPTIMIZERS_CMAES_CMAES_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_CMAES_CMAES_IMPL_HPP

#include "cmaes.hpp"
#include "random.hpp"
#include "timings.hpp"


namespace mlpack {
namespace optimization {

CMAES::CMAES(int objectDim,
double start, double stdDivs,
double iters, double evalEnd)
      :
        N(-1),
        stopMaxFunEvals(-1),
        facmaxeval(1.0),
        stopMaxIter(-1.0),
        stopTolFun(1e-12),
        stopTolFunHist(1e-13),
        stopTolX(0),
        stopTolUpXFactor(1e3),
        lambda(-1),
        mu(-1),
        mucov(-1),
        mueff(-1),
        damps(-1),
        cs(-1),
        ccumcov(-1),
        ccov(-1),
        facupdateCmode(1)
  {
    stStopFitness.flg = false;
    stStopFitness.val = -std::numeric_limits<double>::max();
    updateCmode.modulo = -1;
    updateCmode.maxtime = -1;

     N = objectDim;
    if ( N <= 0)
    throw std::runtime_error("Problem dimension N undefined.");
    
    if(evalEnd != 0) stopTolFun = evalEnd;

    if(start == 0)
    { Log::Warn << " WARNING: initial start point undefined." <<
   "Please specify if incorrect results detected."
   << "DEFAULT = 0.5...0.5." << std::endl;
     start = 0.5; }
    
    if (stdDivs == 0)
    {
Log::Warn << "WARNING: initialStandardDeviations undefined."
<< " Please specify if incorrect results detected. DEFAULT = 0.3...0.3."
<< std::endl;
   stdDivs = 0.3;}

    xstart.set_size(N); 
    rgInitialStds.set_size(N);

          xstart.fill(start);

        for(int i = 0; i < N; ++i)
          rgInitialStds[i] = stdDivs;

    diagonalCov = 0;

    if (lambda < 2)
      lambda = 4 + (int) (3.0*log((double) N));
    if (mu <= 0)
      mu = lambda / 2;

        weights.set_size(mu);
      for (int i = 0; i < mu; ++i) weights[i] = log(mu + 1.) - log(i + 1.);


      // normalize weights vector and set mueff
    double s1 = 0, s2 = 0;
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

    if (iters <= 0)
      stopMaxIter = ceil((double) (stopMaxFunEvals / lambda));
    else
        stopMaxIter = iters;

    if (damps < double(0))
    {
    damps = double(1); damps = damps *
    (double(1) + double(2)*std::max(double(0), std::sqrt((mueff -
    double(1)) / (N + double(1))) - double(1))) * (double)
    std::max(double(0.3), double(1) - // modify for short runs
    (double) N / (double(1e-6) + std::min(stopMaxIter, stopMaxFunEvals
    / lambda))) + cs;
    }

    if (updateCmode.modulo < 0)
      updateCmode.modulo = 1. / ccov / (double) N / 10.;
    updateCmode.modulo *= facupdateCmode;
    if (updateCmode.maxtime < 0)
      updateCmode.maxtime = 0.20;
  }

  template<typename funcType>
  double CMAES::Optimize(funcType& function, arma::mat& arr)
  {
     arFunvals = new double[lambda];
    init();
   int funNo = function.NumFunctions();
    
    arma::Col<double> x(N);

  while (!testForTermination())
  {
    
    // Generate lambda new search points, sample population
    samplePopulation();
   for(int g=0; g<lambda; g++) arFunvals[g] = 0;
    // evaluate the new search points using the given evaluate
    // function by the user
    for (int i = 0; i < lambda; ++i)
   {
      x = population.submat(i, 0, i, N-1).t();

   for (int j = 0; j < funNo; j++)
     arFunvals[i] += function.Evaluate(x, j);
   }
    // update the search distribution used for sampleDistribution()
      updateDistribution(arFunvals);
  }

  // get best estimator for the optimum
  arr = xmean;

    double funs = 0;
    for (int j = 0; j < funNo; j++)
    funs += function.Evaluate(xmean, j);

    return funs;
  }

  void CMAES::adaptC2(const int hsig)
  {

    bool diag = diagonalCov == 1 || diagonalCov >= gen;

    if(ccov != double(0))
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
      for(int i = 0; i < N; ++i)
        for(int j = diag ? i : 0; j <= i; ++j)
        {
          double& Cij = C[i][j];
          Cij = onemccov1ccovmu*Cij + ccov1 * (pc[i]*pc[j] + longFactor*Cij);
          for(int k = 0; k < mu; ++k)
          { // additional rank mu update
            Cij += ccovmu*weights[k] * (population(index[k],i) - xold[i])
                * (population(index[k],j) - xold[j]) / sigmasquare;
          }
        }
      // update maximal and minimal diagonal value
      maxdiagC = mindiagC = C[0][0];
      for(int i = 1; i < N; ++i)
      {
        const double& Cii = C[i][i];
        if(maxdiagC < Cii)
          maxdiagC = Cii;
        else if(mindiagC > Cii)
          mindiagC = Cii;
      }
    }
  }

  /**
   * Initializes the CMA-ES algorithm.
   * @param parameters The CMA-ES parameters in the parameters.h file
   * @return Array of size lambda that can be used to assign fitness values and
   * pass them to updateDistribution()
   */

  void CMAES::init()
  {
    
    double trace(0);
    for(int i = 0; i < N; ++i)
      trace += rgInitialStds[i]*rgInitialStds[i];
    sigma = std::sqrt(trace/N);

    chiN = std::sqrt((double) N) * (double(1) - double(1)/(double(4)*N) + double(1)/(double(21)*N*N));
    eigensysIsUptodate = true;
    doCheckEigen = false;
    genOfEigensysUpdate = 0;

    double dtest;
    for(dtest = double(1); dtest && dtest < double(1.1)*dtest; dtest *= double(2))
      if(dtest == dtest + double(1))
        break;
    dMaxSignifKond = dtest / double(1000); // not sure whether this is really save, 100 does not work well enough

    gen = 0;
    countevals = 0;
    state = INITIALIZED;
    dLastMinEWgroesserNull = double(1);
    pc.set_size(N);
    ps.set_size(N);
    tempRandom = new double[N+1];
    BDz.set_size(N);
    xmean.set_size(N);
    xold.set_size(N);
    xBestEver.set_size(N+2);
    xBestEver[N] = std::numeric_limits<double>::max();
    rgD = new double[N];
    C = new double*[N];
    B = new double*[N];
    functionValues = new double[lambda+1];
    functionValues[0] = lambda;
    ++functionValues;
    const int historySize = 10 + (int) ceil(3.*10.*N/lambda);
    funcValueHistory = new double[historySize + 1];
    funcValueHistory[0] = (double) historySize;
    funcValueHistory++;
    
    for(int i = 0; i < N; ++i)
    {
      C[i] = new double[i+1];
      B[i] = new double[N];
    }
    index = new int[lambda];
    for(int i = 0; i < lambda; ++i)
        index[i] = i;
    population.set_size(lambda,N+1);

    // initialize newed space
    for(int i = 0; i < lambda; i++)
    {
      functionValues[i] = std::numeric_limits<double>::max();
    }
    for(int i = 0; i < historySize; i++)
    {
      funcValueHistory[i] = std::numeric_limits<double>::max();
    }
    for(int i = 0; i < N; ++i)
      for(int j = 0; j < i; ++j)
        C[i][j] = B[i][j] = B[j][i] = 0.;

    for(int i = 0; i < N; ++i)
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
    for(int i = 1; i < N; ++i) if(maxdiagC < C[i][i]) maxdiagC = C[i][i];
    mindiagC = C[0][0];
    for(int i = 1; i < N; ++i) if(mindiagC > C[i][i]) mindiagC = C[i][i];

      xmean = xold = xstart;
   
      for(int i = 0; i < N; ++i)
        xmean[i] += sigma*rgD[i]*rand.gauss();
  }

  /**
   * The search space vectors have a special form: they are arrays with N+1
   * entries. Entry number -1 is the dimension of the search space N.
   * @return A pointer to a "population" of lambda N-dimensional multivariate
   * normally distributed samples.
   */

void CMAES::samplePopulation()
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
        minEW = square(minElement(rgD, N));
        maxEW = square(maxElement(rgD, N));
        eigensysIsUptodate = true;
        eigenTimings.start();
      }
    }

  for(int iNk = 0; iNk < lambda; ++iNk)
    { // generate scaled random vector D*z
      for(int i = 0; i < N; ++i)
        if(diag)
          population(iNk,i) = xmean[i] + sigma*rgD[i]*rand.gauss();
        else
          tempRandom[i] = rgD[i]*rand.gauss();
      if(!diag)
        for(int i = 0; i < N; ++i) // add mutation sigma*B*(D*z)
        {
          double sum = 0.0;
          for(int j = 0; j < N; ++j)
            sum += B[i][j]*tempRandom[j];

          population(iNk,i) = xmean[i] + sigma*sum;
        }
    }

    if(state == UPDATED || gen == 0)
      ++gen;
    state = SAMPLED;

  }

  /** Core procedure of the CMA-ES algorithm. Sets a new mean
  value and estimates
* the new covariance matrix and a new step sizefor the normal search
  distribution.
* @param fitnessValues An array of \f$\lambda\f$ function values.
* @return Mean value of the new distribution. */

void CMAES::updateDistribution(double* fitnessValues)
{
    bool diag = diagonalCov == 1 || diagonalCov >= gen;
    assert(state != UPDATED && "updateDistribution(): You need to call "
          "samplePopulation() before update can take place.");

    if (state == SAMPLED) // function values are delivered here
      countevals += lambda;
    else Log::Warn <<  "updateDistribution(): unexpected state" << std::endl;

    // assign function values
    for(int i = 0; i < lambda; ++i)
      population(i,N) = functionValues[i] = fitnessValues[i];

    // Generate index
    sortIndex(fitnessValues, index, lambda);
    // Test if function values are identical, escape flat fitness
    if (fitnessValues[index[0]] == fitnessValues[index[(int) lambda / 2]])
    {
      sigma *= std::exp(double(0.2) + cs / damps);

      std::cout << "Warning: sigma increased due to equal function values"
      << std::endl << "Reconsider the formulation of the objective function";
    }

for(int i = (int) *(funcValueHistory - 1) - 1; i > 0; --i)
      funcValueHistory[i] = funcValueHistory[i - 1];
    funcValueHistory[0] = fitnessValues[index[0]];

    // update xbestever
    if(xBestEver[N] > population(index[0],N) || gen == 1)
      for(int i = 0; i <= N; ++i)
      {
        xBestEver[i] = population(index[0],i);
        xBestEver[N+1] = countevals;
      }

    const double sqrtmueffdivsigma = std::sqrt(mueff) / sigma;
    // calculate xmean and rgBDz~N(0,C)
    for(int i = 0; i < N; ++i)
    {
      xold[i] = xmean[i];
      xmean[i] = 0.;
      for(int iNk = 0; iNk < mu; ++iNk)
        xmean[i] += weights[iNk]*population(index[iNk],i);
      BDz[i] = sqrtmueffdivsigma*(xmean[i]-xold[i]);
    }

    // calculate z := D^(-1)* B^(-1)* rgBDz into rgdTmp
    for(int i = 0; i < N; ++i)
    {
      double sum;
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
    const double sqrtFactor = std::sqrt(cs*(double(2)-cs));
    const double invps = double(1)-cs;
    for(int i = 0; i < N; ++i)
    {
      double sum;
      if(diag)
        sum = tempRandom[i];
      else
      {
        sum = double(0);
        double* Bi = B[i];
        for(int j = 0; j < N; ++j)
          sum += Bi[j]*tempRandom[j];
      }
      ps[i] = invps*ps[i] + sqrtFactor*sum;
    }

    // calculate norm(ps)^2
    double psxps(0);
    for(int i = 0; i < N; ++i)
    {
      const double& rgpsi = ps[i];
      psxps += rgpsi*rgpsi;
    }

    // cumulation for covariance matrix (pc) using B*D*z~N(0,C)
    int hsig = std::sqrt(psxps) / std::sqrt(double(1) - std::pow(double(1) - cs, double(2)* gen))
        / chiN < double(1.4) + double(2) / (N + 1);
    const double ccumcovinv = 1.- ccumcov;
    const double hsigFactor = hsig*std::sqrt(ccumcov*(double(2)-ccumcov));
    for(int i = 0; i < N; ++i)
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

bool CMAES::testForTermination()
  {
     double range, fac;
    int iAchse, iKoo;
    int diag = diagonalCov == 1 || diagonalCov >= gen;

    // function value reached
    if((gen > 1 || state > SAMPLED) && stStopFitness.flg &&
        functionValues[index[0]] <= stStopFitness.val)
    {
      std::cout << "Fitness: function value " << functionValues[index[0]]
          << " <= stopFitness (" << stStopFitness.val << ")" << std::endl;
          return true;
    }
//std::cout << *(funcValueHistory - 1)-1 << std::endl;
    // TolFun
    range = std::max(maxElement(funcValueHistory, (int) std::min(gen, *(funcValueHistory - 1))),
        maxElement(functionValues, lambda)) -
        std::min(minElement(funcValueHistory, (int) std::min(gen, *(funcValueHistory - 1))),
        minElement(functionValues, lambda));

    if(gen > 0 && range <= stopTolFun)
    {
      std::cout << "TolFun: function value differences " << range
          << " < stopTolFun=" << stopTolFun << std::endl;
           return true;
    }
//std::cout << *(funcValueHistory - 1)-1 << std::endl;
    // TolFunHist
    if(gen > *(funcValueHistory - 1))
    {
      range = maxElement(funcValueHistory, (int) *(funcValueHistory - 1))
          - minElement(funcValueHistory, (int) *(funcValueHistory - 1));
      if(range <= stopTolFunHist)
      {
        std::cout << "TolFunHist: history of function value changes " << range
            << " stopTolFunHist=" << stopTolFunHist << std::endl;
       return true;
      }
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
      std::cout << "TolX: object variable changes below " << stopTolX << std::endl;
    }

    // TolUpX
    for(int i = 0; i < N; ++i)
    {
      if(sigma*std::sqrt(C[i][i]) > stopTolUpXFactor*rgInitialStds[i])
      {
        std::cout << "TolUpX: standard deviation increased by more than "
            << stopTolUpXFactor << ", larger initial standard deviation recommended."
            << std::endl;
             return true;
      }
    }

    // Condition of C greater than dMaxSignifKond
    if(maxEW >= minEW* dMaxSignifKond)
    {
      std::cout << "ConditionNumber: maximal condition number " << dMaxSignifKond
          << " reached. maxEW=" << maxEW <<  ",minEW=" << minEW << ",maxdiagC="
          << maxdiagC << ",mindiagC=" << mindiagC << std::endl;
           return true;
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
          std::cout << "NoEffectAxis: standard deviation 0.1*" << (fac / 0.1)
              << " in principal axis " << iAchse << " without effect" << std::endl;
           return true;
        }
      }
    }
    // Component of xmean is not changed anymore
    for(iKoo = 0; iKoo < N; ++iKoo)
    {
      if(xmean[iKoo] == xmean[iKoo] + sigma*std::sqrt(C[iKoo][iKoo])/double(5))
      {
        std::cout << "NoEffectCoordinate: standard deviation 0.2*"
            << (sigma*std::sqrt(C[iKoo][iKoo])) << " in coordinate " << iKoo
            << " without effect" << std::endl;
         return true;
      }
    }

    if(countevals >= stopMaxFunEvals)
    {
      std::cout << "MaxFunEvals: conducted function evaluations " << countevals
          << " >= " << stopMaxFunEvals << std::endl;  return true;
    }
    if(gen >= stopMaxIter)
    {
      std::cout << "MaxIter: number of iterations " << gen << " >= "
          << stopMaxIter << std::endl; return true;
    }

   return false;
  }

  /**
   * Conducts the eigendecomposition of C into B and D such that
   * \f$C = B \cdot D \cdot D \cdot B^T\f$ and \f$B \cdot B^T = I\f$
   * and D diagonal and positive.
   * @param force For force == true the eigendecomposion is conducted even if
   *              eigenvector and values seem to be up to date.
   */

void CMAES::updateEigensystem(bool force)
  {
    eigenTimings.update();

    if(!force)
    {
      if(eigensysIsUptodate)
        return;
      // return on modulo generation number
      if(gen < genOfEigensysUpdate + updateCmode.modulo)
        return;
      // return on time percentage
      if(updateCmode.maxtime < 1.00
          && eigenTimings.tictoctime > updateCmode.maxtime* eigenTimings.totaltime
          && eigenTimings.tictoctime > 0.0002)
        {
          Log::Info << " time return happened " << std::endl;
        return;
      }
    }

    eigenTimings.tic();
    eigen(rgD, B, tempRandom);
    eigenTimings.toc();

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

} // namespace optimization
} // namespace mlpack

#endif
