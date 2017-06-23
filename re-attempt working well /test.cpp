/**
 * @file test.cpp
 * Very short example of CMAES optimizer working
 */

#include <stdlib.h>
#include <iostream>
#include "cmaes.hpp"
#include <math.h>

using namespace mlpack;
using namespace mlpack::optimization;
/**
*
 */
double fitfun(double const *x, int N)
{
  double func = (pow(x[0]-5, 2))*(pow(x[1]-3, 2)) ; // function = (x-5)^2 * (y-3)^2 (minima at x=5 y=3)
  return func;
}

int main(int, char**)
 {

  double *arFunvals, *const*pop, xfinal[2];

  
  const int dim = 2;

  double xstart[dim];

  for(int i=0; i<dim; i++) xstart[i] = 0.5;
  double stddev[dim];
  for(int i=0; i<dim; i++) stddev[i] = 0.5;

  CMAES<double> evo(dim, xstart, stddev);

  arFunvals = evo.init();

  while(!evo.testForTermination())
  {
    // Generate lambda new search points, sample population
    pop = evo.samplePopulation();

    // evaluate the new search points using fitfun from above
    for (int i = 0; i < evo.getSampleSize(); ++i)
      arFunvals[i] = fitfun(pop[i], (int) evo.getDimension());

    // update the search distribution used for sampleDistribution()
    evo.updateDistribution(arFunvals);
  }

  std::cout << "Stop:" << std::endl << evo.getStopMessage();

  // get best estimator for the optimum
  evo.getFittestMean(xfinal);

  std::cout << "value for x = " << xfinal[0] << " value for y = " << xfinal[1] << std::endl;

  return 0;
}

