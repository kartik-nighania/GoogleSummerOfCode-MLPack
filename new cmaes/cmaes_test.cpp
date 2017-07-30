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
//#include <mlpack/core.hpp>
#include <math.h>

#include "cmaes.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::optimization;


  class rosenbrock
  {
    public:
    
    double NumFunctions(){return 44;}

    double Evaluate(double *x, int i)
    {
      return 100.*pow((pow((x[i]),2)-x[i+1]),2) + pow((1.-x[i]),2);
    }

  };

int main()
{ 
mlpack::math::RandomSeed(std::time(NULL));
  
  rosenbrock test;

  CMAES s(45,0.5, 0.3, 100000, 1e-13);

//  arma::mat coordinates(N,1);
  double coordinates[45];
  double result = s.Optimize(test, coordinates);

  cout << endl << result << endl;
for(int i=0; i<45; i++) std::cout << coordinates[i] << " ";
std::cout << std::endl;


return 0;
}

