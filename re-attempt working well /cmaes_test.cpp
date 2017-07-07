/**
 * @file cmaes_test.cpp
 * @author Kartik Nighania (Mentor Marcus Edel)
 *
 * Test file for CMAES (Covariance Matrix Adaptation Evolution Strategy).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/lbfgs/test_functions.hpp>

#include <boost/test/unit_test.hpp>

#include "test_tools.hpp"
#include "cmaes.hpp"
#include "test_function.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

BOOST_AUTO_TEST_SUITE(CMAEStest);

BOOST_AUTO_TEST_CASE(SimpleCMAEStestFunction)
{
  cmaesTestFunction test;
  
  int N = test.NumFunctions();

  arma::mat start(N,1); start.fill(0.5); 
  arma::mat initialStdDeviations(N,1); initialStdDeviations.fill(1.5);

  CMAES<cmaesTestFunction> s(test,start,initialStdDeviations);

  arma::mat coordinates(N,1);
  double result = s.Optimize(coordinates);

  BOOST_REQUIRE_CLOSE(result, -1.0, 0.05);
  BOOST_REQUIRE_SMALL(coordinates[0], 1e-3);
  BOOST_REQUIRE_SMALL(coordinates[1], 1e-7);
  BOOST_REQUIRE_SMALL(coordinates[2], 1e-7);
}

/* CANT IMPLEMENT DUE TO CHANGE IN EVALUATE FUNCTION IN CMAES

BOOST_AUTO_TEST_CASE(GeneralizedRosenbrockTest)
{
  // Loop over several variants.
  for (size_t i = 10; i < 50; i += 5)
  {
    // Create the generalized Rosenbrock function.
    GeneralizedRosenbrockFunction f(i);

    SGD<GeneralizedRosenbrockFunction> s(f, 0.001, 0, 1e-15, true);

    arma::mat coordinates = f.GetInitialPoint();
    double result = s.Optimize(coordinates);

    BOOST_REQUIRE_SMALL(result, 1e-10);
    for (size_t j = 0; j < i; ++j)
      BOOST_REQUIRE_CLOSE(coordinates[j], (double) 1.0, 1e-3);
  }
}
*/

BOOST_AUTO_TEST_SUITE_END();
