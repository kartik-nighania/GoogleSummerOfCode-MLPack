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


#include <mlpack/methods/logistic_regression/logistic_regression.hpp>
#include <mlpack/core/optimizers/sgd/test_function.hpp>
#include "cmaes.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

using namespace mlpack::distribution;
using namespace mlpack::regression;

int main()
{ 
mlpack::math::RandomSeed(std::time(NULL));
 
 // SGD TEST CASE PASS
 
  SGDTestFunction test;
  
  size_t N = test.NumFunctions();

  arma::mat start(N,1); start.fill(0.5); 
  arma::mat initialStdDeviations(N,1); initialStdDeviations.fill(1.5);

  CMAES s(N,start,initialStdDeviations,10000,1e-18);

  arma::mat coordinates(N,1);
  double result = s.Optimize(test, coordinates);

cout << 
  "BOOST_REQUIRE_CLOSE(result, -1.0, 0.05);   \n" <<
  "BOOST_REQUIRE_SMALL(coordinates[0], 1e-3); \n" <<
  "BOOST_REQUIRE_SMALL(coordinates[1], 1e-7); \n" <<
  "BOOST_REQUIRE_SMALL(coordinates[2], 1e-7);" << endl;

  cout << endl << result << endl;
  cout << coordinates[0] << endl;
  cout << coordinates[1] << endl;
  cout << coordinates[2] << endl;

  // Generate a two-Gaussian dataset.
  GaussianDistribution g1(arma::vec("1.0 1.0 1.0"), arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("9.0 9.0 9.0"), arma::eye<arma::mat>(3, 3));

  arma::mat data(3, 1000);
  arma::Row<size_t> responses(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    data.col(i) = g1.Random();
    responses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    data.col(i) = g2.Random();
    responses[i] = 1;
  }

  // Shuffle the dataset.
  arma::uvec indices = arma::shuffle(arma::linspace<arma::uvec>(0,
      data.n_cols - 1, data.n_cols));
  arma::mat shuffledData(3, 1000);
  arma::Row<size_t> shuffledResponses(1000);
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    shuffledData.col(i) = data.col(indices[i]);
    shuffledResponses[i] = responses[indices[i]];
  }

  // Create a test set.
  arma::mat testData(3, 1000);
  arma::Row<size_t> testResponses(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    testData.col(i) = g1.Random();
    testResponses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    testData.col(i) = g2.Random();
    testResponses[i] = 1;
  }

  int dim = shuffledData.n_rows + 1;
  arma::mat start1(dim,1); start1.fill(0.5); 
  arma::mat initialStdDeviations1(dim,1); initialStdDeviations1.fill(1.5);

  CMAES test1(dim, start1, initialStdDeviations1, 50000, 1e-7);

  LogisticRegression<arma::mat> lr(shuffledData, shuffledResponses, test1, 0.5);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses);
  cout << "got this value = " << acc << " should be = 100.0 with tolerance = 0.3" << endl; // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses);
cout << "got this value = " << testAcc << " should be = 100.0 with tolerance = 0.3" << endl;

return 0;
}

