/**
 * @file test.cpp
 * Very short example of CMAES optimizer working
 */
#include <iostream>
#include <math.h>
#include <mlpack/core.hpp>
#include <armadillo>

#include "cmaes.hpp"

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

  size_t NumFunctions() const { return 2; }
};

int main(int, char**)
 {
   testFunction test;

  arma::mat xstart("0.5; 0.5");
  arma::mat stddev("0.5; 0.5");

  CMAES<testFunction> evo(test, xstart, stddev);
  
  arma::mat values(2,1);

  double result = evo.Optimize(values);

  std::cout << "lowest value found being = " << result << std::endl 
 << " value for x = " << values(0) << " value for y = " << values(1) << std::endl;

  return 0;
}

