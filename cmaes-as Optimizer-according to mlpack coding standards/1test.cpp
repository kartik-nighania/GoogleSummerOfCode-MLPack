/**
 * @file test.cpp
 * Very short example of CMAES optimizer working
 */

#include <stdlib.h>
#include <iostream>
#include <math.h>

#include "cmaes.hpp"

using namespace mlpack;
using namespace mlpack::optimization;

class cmaesTestFunction
{
 public:
  //! Nothing to do for the constructor.
  cmaesTestFunction() { }

  //! Return 3 (the number of functions) = the variable N in CMAES class for dimension
  size_t NumFunctions() {return 3; };

  //! Get the starting point = CMAES class xstart array of dimension given by NumFunctions()
 arma::mat GetInitialPoint() { return arma::mat("6; -45.6; 6.2"); };

  //! Get the intial standard devaition = CMAES class stddiv array of dimenison given by NumFunction()
  arma::mat GetInitialStdDev() { return arma::mat("3; 3; 3"); }

  //! Evaluate a function.
  double Evaluate(double const *coordinates)
  {
  
      return -std::exp(-std::abs(coordinates[0])) 
      + std::pow(coordinates[1], 2) 
      + std::pow(coordinates[2], 4) + 3 * std::pow(coordinates[2], 2);

  } 

};

int main(int, char**)
 {

  cmaesTestFunction func;

  CMAES<cmaesTestFunction, double> fo(func);

  double arr[3];
  double s = fo.Optimize(func, arr);

   return 0;
}

