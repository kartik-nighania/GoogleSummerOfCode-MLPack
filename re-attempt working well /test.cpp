/**
 * @file test.cpp
 * Very short example of CMAES optimizer working
 */

/* 
now i need dim , start, stddiv
arma::mat for test function and standard dev

*/

#include <stdlib.h>
#include <iostream>
#include "cmaes.hpp"
#include <math.h>
#include <armadillo>

using namespace std;
using namespace mlpack;
using namespace mlpack::optimization;
/**
*
 */
class testFunction
{ 
public:
    
  double Evaluate(const arma::mat& x)
  {
    double func = (pow(x[0]-5, 2))*(pow(x[1]-3, 2)) ; // function = (x-5)^2 * (y-3)^2 (minima at x=5 y=3)
    return func;
  }

  //! Return 3 (the number of functions).
  size_t NumFunctions() const { return 2; }
};

int main(int, char**)
 {
   testFunction test;

  double xstart[2];
  for(int i=0; i<2; i++) xstart[i] = 0.5;

  double stddev[2];
  for(int i=0; i<2; i++) stddev[i] = 0.5;

  CMAES<testFunction> evo(test, xstart, stddev);
  
  double xfinal[2];

  double low = evo.Optimize(xfinal);

  std::cout << "lowest value found being = " << low << std::endl << " value for x = " << xfinal[0] << " value for y = " << xfinal[1] << std::endl;

  return 0;
}

