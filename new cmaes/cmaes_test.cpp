/**
 * @file cmaes_test.cpp
 * @author Ryan Curtin
 *
 * Test file for SGD (stochastic gradient descent).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <math.h>

#include "cmaes.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::optimization;


  class rosenbrock
  {
    public:
    
    double NumFunctions(){return 49;}

    double Evaluate(arma::mat& x, int i)
    {
      return 100.*pow((pow((x[i]),2)-x[i+1]),2) + pow((1.-x[i]),2);
    }

  };

    class simpleFunction
  {
    public:
    
    double NumFunctions(){return 3;}

   double Evaluate(arma::mat& coordinates, int i)
    const
    {
      switch (i)
      {
        case 0:
          return -std::exp(-std::abs(coordinates[0]));

        case 1:
          return std::pow(coordinates[1], 2);

        case 2:
          return std::pow(coordinates[2], 4) + 3 * std::pow(coordinates[2], 2);

        default:
          return 0;
      }
    }

  };

int main()
{ 
mlpack::math::RandomSeed(std::time(NULL));
  
  rosenbrock test;

  CMAES s(50,0.5, 0.3, 100000, 1e-13);

//  arma::mat coordinates(N,1);
  arma::vec coordinates(50);
  double result = s.Optimize(test, coordinates);

  cout << endl << result << endl;
for(int i=0; i<50; i++) std::cout << coordinates[i] << " ";
std::cout << std::endl;

  simpleFunction testing;

  CMAES fun(3, 0.5, 0.3, 100000, 1e-13);

//  arma::mat coordinates(N,1);
  arma::vec coordinates1(3);
  double result1 = fun.Optimize(testing, coordinates1);

  cout << endl << result1 << endl;
  for(int i=0; i<3; i++) std::cout << coordinates1[i] << " ";
  std::cout << std::endl;

return 0;
}

