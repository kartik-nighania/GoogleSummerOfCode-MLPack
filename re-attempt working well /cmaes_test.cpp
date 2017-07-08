/**
 * @file sgd_test.cpp
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

#include "cmaes.hpp"
#include "test_function.hpp"

#include <mlpack/core/optimizers/sgd/sgd.hpp>
#include <mlpack/core/optimizers/lbfgs/test_functions.hpp>
#include <mlpack/core/optimizers/sgd/test_function.hpp>

#include <armadillo>

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

int main()
{
  cmaesTestFunction test;
  
  int N = test.NumFunctions();

  arma::mat start(N,1); start.fill(0.5); 
  arma::mat initialStdDeviations(N,1); initialStdDeviations.fill(1.5);

  CMAES<cmaesTestFunction> s(test,start,initialStdDeviations);

  arma::mat coordinates(N,1);
  double result = s.Optimize(coordinates);

cout << 
  "BOOST_REQUIRE_CLOSE(result, -1.0, 0.05);   \n" <<
  "BOOST_REQUIRE_SMALL(coordinates[0], 1e-3); \n" <<
  "BOOST_REQUIRE_SMALL(coordinates[1], 1e-7); \n" <<
  "BOOST_REQUIRE_SMALL(coordinates[2], 1e-7);" << endl;

  cout << endl << result << endl;
  cout << coordinates[0] << endl;
  cout << coordinates[1] << endl;
  cout << coordinates[2] << endl;

/*
  // test results according to SGD

  SGDTestFunction f;
  StandardSGD s1(0.0003, 5000000, 1e-9, true);

  arma::mat coordinates1 = f.GetInitialPoint();
  double result1 = s1.Optimize(f, coordinates1);

cout << endl << "SGD test results " << endl;

  cout << endl << result1 << endl;
  cout << coordinates1[0] << endl;
  cout << coordinates1[1] << endl;
  cout << coordinates1[2] << endl;

*/

return 0;
}

