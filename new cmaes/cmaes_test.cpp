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

#include <mlpack/core/optimizers/lbfgs/test_functions.hpp>
#include <mlpack/core/optimizers/sgd/test_function.hpp>
#include <math.h>

#include "cmaes.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;


  class rosenbrock
  {
     public:
    
    int N;

    rosenbrock(int x){ N = x-1; }

    double NumFunctions(){return N;}

    double Evaluate(arma::mat& x, int i)
    {
      return 100.*pow((pow((x[i]),2)-x[i+1]),2) + pow((1.-x[i]),2);
    }

  };



int main()
{ 
mlpack::math::RandomSeed(std::time(NULL));
  
  for(int i=2; i<20; i += 1)
  {
    rosenbrock test(i);

    CMAES s(i,0.5, 0.3, 100000, 1e-16, 1e-15);

  //  arma::mat coordinates(N,1);
    arma::vec coordinates(i);
    double result = s.Optimize(test, coordinates);

    cout << endl << result << endl;
    for(int j=0; j<i; j++) std::cout << coordinates[j] << " ";
    std::cout << std::endl;
  }


  SGDTestFunction testing;

  CMAES fun(3, 0.5, 0.3, 100000, 1e-16, 1e-15);

//  arma::mat coordinates(N,1);
  arma::vec coordinates1(3);
  double result1 = fun.Optimize(testing, coordinates1);

  cout << endl << result1 << endl;
  for(int i=0; i<3; i++) std::cout << coordinates1[i] << " ";
  std::cout << std::endl;

/*
// Loop over several variants.
  for (size_t i = 10; i < 50; i += 5)
  {
    // Create the generalized Rosenbrock function.
    GeneralizedRosenbrockFunction f(i);

    CMAES t(3, 0.5, 0.3, 100000, 1e-16);

    arma::mat coordinates2 = f.GetInitialPoint();
    double result2 = t.Optimize(f, coordinates2);

   cout << endl << result2 << endl;
    for (size_t j = 0; j < i; ++j)
     std::cout << coordinates2[j] << " ";
     std::cout << std::endl;
  }
*/

return 0;
}

