/**
 * @file example1.cpp
 * Very short example source code. The purpose of the example codes is to be
 * edited/extended.
 */

#include <stdlib.h>
#include <iostream>
#include "cmaes.h"
#include <math.h>

using namespace std;

/**
 * Rosenbrock's Function, generalized.
 */
double f_rosenbrock( double const *x, int di)
{
  double qualitaet = 0.0;
  int DIM = di;

  for(int i = DIM-2; i >= 0; --i)
    qualitaet += 100.*pow((pow((x[i]),2)-x[i+1]),2) + pow((1.-x[i]),2);
  return ( qualitaet);
}

/**
 * The optimization loop.
 */
int main(int, char**) 
{
  CMAES<double> evo;
  double *arFunvals, *const*pop, *xfinal;

  // Initialize everything
  const int dim = 45;
  double xstart[dim];
  for(int i=0; i<dim; i++) xstart[i] = 0.5;
  double stddev[dim];
  for(int i=0; i<dim; i++) stddev[i] = 0.3;
  Parameters<double> parameters;
  
  parameters.init(dim, xstart, stddev);
  arFunvals = evo.init(parameters);

  // Iterate until stop criterion holds
  while(!evo.testForTermination())
  {
    // Generate lambda new search points, sample population
    pop = evo.samplePopulation(); // Do not change content of pop

    // evaluate the new search points using fitfun from above
    for (int i = 0; i < evo.get(CMAES<double>::Lambda); ++i)
      arFunvals[i] = f_rosenbrock(pop[i], (int) evo.get(CMAES<double>::Dimension));

  //  for(int s=0; s<evo.get(CMAES<double>::Lambda); s++) std::cout << arFunvals[s] << " ";
    //  cout << endl;
    // update the search distribution used for sampleDistribution()
    evo.updateDistribution(arFunvals);
  }
  std::cout << "Stop:" << std::endl << evo.getStopMessage();
  
  // get best estimator for the optimum, xmean
  xfinal = evo.getNew(CMAES<double>::XMean); // "XBestEver" might be used as well

std::cout << std::endl;
for(int i=0; i<45; i++) std::cout << *(xfinal+i) << " ";
std::cout << std::endl;
  // do something with final solution and finally release memory
  delete[] xfinal;

  return 0;
}

