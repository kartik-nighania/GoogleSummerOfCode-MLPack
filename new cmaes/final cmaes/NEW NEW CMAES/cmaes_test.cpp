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

#include <mlpack/methods/logistic_regression/logistic_regression.hpp>

#include <mlpack/core/optimizers/sgd/update_policies/vanilla_update.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>

#include "cmaes.hpp"
#include <time.h>

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;


using namespace mlpack::distribution;
using namespace mlpack::regression;

int main()
{ 

  double start = clock();
  
mlpack::math::RandomSeed(std::time(NULL));

  SGDTestFunction test1;

  CMAES opt1(3, 0.5, 0.3, 10000, 1e-13, 1e-13);

  arma::mat coordinates1(3, 1);
  double result1 = opt1.Optimize(test1, coordinates1);

  double end1 = clock();

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

  CMAES test2(shuffledData.n_rows + 1, 0.5, 0.3, 10000, 1e-10, 1e-10);

  LogisticRegression<> lr(shuffledData, shuffledResponses, test2, 0.5);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses);
  const double testAcc = lr.ComputeAccuracy(testData, testResponses);

  double end2 = clock();

  cout << result1 << endl;
  cout << coordinates1[0] << "  " << coordinates1[1] << "  " << coordinates1[2] << endl;

  cout << endl << endl;

  cout << acc << " 100.0" << endl; // 0.3% error tolerance.
  cout << testAcc << " 100.0" << endl << endl; // 0.6% error tolerance.

    // Loop over several variants.
  for (size_t i = 5; i < 30; i += 5)
  {
    // Create the generalized Rosenbrock function.
    GeneralizedRosenbrockFunction f(i);

    CMAES s(i, 0.5, 0.3, 100000, 1e-16, 1e-16);

    arma::mat coordinates2 = f.GetInitialPoint();
    double result2 = s.Optimize(f, coordinates2);

    cout << result2 << endl;
    for (size_t j = 0; j < i; ++j)
      cout << coordinates2[j] << " ";
    cout << endl;
  }

  double end3 = clock();

  cout << "Total time = " << (end3 - start)/CLOCKS_PER_SEC << endl << endl;
  cout << "SGD Time = " << (end1 - start)/CLOCKS_PER_SEC << endl;
  cout << "log Time = " << (end2 - end1)/CLOCKS_PER_SEC << endl;
  cout << "ROS Time = " << (end3 - end2)/CLOCKS_PER_SEC << endl;

return 0;
}

