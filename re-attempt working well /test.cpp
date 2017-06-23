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
class chaljaPlz
{ 
public:
  
  double Evaluate(double const *x, int N)
  {
    double func = (pow(x[0]-5, 2))*(pow(x[1]-3, 2)) ; // function = (x-5)^2 * (y-3)^2 (minima at x=5 y=3)
    return func;
  }
};

int main(int, char**)
 {
  chaljaPlz plz;



  
  const int dim = 2;

  double xstart[dim];

  for(int i=0; i<dim; i++) xstart[i] = 0.5;
  double stddev[dim];
  for(int i=0; i<dim; i++) stddev[i] = 0.5;

  CMAES<chaljaPlz> evo(plz, dim, xstart, stddev);
  
  double xfinal[2];

  double low = evo.Optimize(xfinal);

  std::cout << "lowest value found being = " << low << std::endl << " value for x = " << xfinal[0] << " value for y = " << xfinal[1] << std::endl;

  return 0;
}

