/**
 * @file cmaes_test.cpp
 * @author Kartik Nighania Mentor Marcus Edel
 *
 * Test file for CMAES (Covariance Matrix Adaptation Evolution Strategy).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include <mlpack/core/optimizers/cmaes/cmaes.hpp>
#include <mlpack/core/optimizers/lbfgs/test_functions.hpp>
#include <mlpack/core/optimizers/sgd/test_function.hpp>

#include <mlpack/methods/logistic_regression/logistic_regression.hpp>

#include <mlpack/core/optimizers/sgd/update_policies/vanilla_update.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

using namespace mlpack::distribution;
using namespace mlpack::regression;

BOOST_AUTO_TEST_SUITE(CMAESTest);

/**
 * Train and evaluate a vanilla network with the specified structure.
 */
template<typename MatType = arma::mat>
void BuildVanillaNetwork(MatType& trainData,
                         MatType& trainLabels,
                         MatType& testData,
                         MatType& testLabels,
                         const size_t outputSize,
                         const size_t hiddenLayerSize,
                         const size_t maxEpochs,
                         const double classificationErrorThreshold)
{
  /*
   * Construct a feed forward network with trainData.n_rows input nodes,
   * hiddenLayerSize hidden nodes and trainLabels.n_rows output nodes. The
   * network structure looks like:
   *
   *  Input         Hidden        Output
   *  Layer         Layer         Layer
   * +-----+       +-----+       +-----+
   * |     |       |     |       |     |
   * |     +------>|     +------>|     |
   * |     |     +>|     |     +>|     |
   * +-----+     | +--+--+     | +-----+
   *             |             |
   *  Bias       |  Bias       |
   *  Layer      |  Layer      |
   * +-----+     | +-----+     |
   * |     |     | |     |     |
   * |     +-----+ |     +-----+
   * |     |       |     |
   * +-----+       +-----+
   */

  FFN<NegativeLogLikelihood<> > model;
  model.Add<Linear<> >(trainData.n_rows, hiddenLayerSize);
  model.Add<SigmoidLayer<> >();
  model.Add<Linear<> >(hiddenLayerSize, outputSize);
  model.Add<LogSoftMax<> >();

  int dim = trainData.n_rows * hiddenLayerSize +  hiddenLayerSize * outputSize;
  arma::mat start1(dim, 1); start1.fill(0.5);
  arma::mat initialStdDeviations1(dim, 1); initialStdDeviations1.fill(0.3);

  CMAES opt(dim, start1, initialStdDeviations1, 50000, 1e-8);

  model.Train(trainData, trainLabels, opt);

  MatType predictionTemp;
  model.Predict(testData, predictionTemp);
  MatType prediction = arma::zeros<MatType>(1, predictionTemp.n_cols);

  for (size_t i = 0; i < predictionTemp.n_cols; ++i)
  {
    prediction(i) = arma::as_scalar(arma::find(
        arma::max(predictionTemp.col(i)) == predictionTemp.col(i), 1)) + 1;
  }

  size_t error = 0;
  for (size_t i = 0; i < testData.n_cols; i++)
  {
    if (int(arma::as_scalar(prediction.col(i))) ==
        int(arma::as_scalar(testLabels.col(i))))
    {
      error++;
    }
  }

  double classificationError = 1 - double(error) / testData.n_cols;
  BOOST_REQUIRE_LE(classificationError, classificationErrorThreshold);
}

BOOST_AUTO_TEST_CASE(SimpleCMAESTestFunction)
{
  SGDTestFunction test;

  size_t N = test.NumFunctions();

  arma::mat start(N, 1); start.fill(0.5);
  arma::mat initialStdDeviations(N, 1); initialStdDeviations.fill(1.5);

  CMAES s(N, start, initialStdDeviations, 10000, 1e-18);

  arma::mat coordinates(N, 1);
  double result = s.Optimize(test, coordinates);

  BOOST_REQUIRE_CLOSE(result, -1.0, 0.05);
  BOOST_REQUIRE_SMALL(coordinates[0], 1e-3);
  BOOST_REQUIRE_SMALL(coordinates[1], 1e-7);
  BOOST_REQUIRE_SMALL(coordinates[2], 1e-7);
}

// written by Marcus Edel and modified for CMAES by Kartik Nighania
// to test CMAES as an optimizer for logistic regression optimization

BOOST_AUTO_TEST_CASE(LogisticRegressionTestWithCMAES)
{
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
  arma::mat start1(dim, 1); start1.fill(0.5);
  arma::mat initialStdDeviations1(dim, 1); initialStdDeviations1.fill(1.5);

  CMAES test1(dim, start1, initialStdDeviations1, 50000, 1e-7);

  LogisticRegression<> lr(shuffledData, shuffledResponses, test1, 0.5);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses);
  BOOST_REQUIRE_CLOSE(acc, 100.0, 0.3); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses);
  BOOST_REQUIRE_CLOSE(testAcc, 100.0, 0.6); // 0.6% error tolerance.
}

BOOST_AUTO_TEST_CASE(feedForwardNetworkCMAES)
{
  arma::mat irisTrainData;
  data::Load("iris_train.csv", irisTrainData, true);
 
 //normalize train data
 double minVal=0,range=1;
 for(int i=0; i<irisTrainData.n_rows; i++)
 {
   minVal = irisTrainData.row(i).min();
   range  = irisTrainData.row(i).max() - minVal;
   irisTrainData.row(i) =  (irisTrainData.row(i) - minVal)/range;
 }

 arma::mat irisTrainLabels;
 data::Load("iris_train_labels.csv", irisTrainLabels, true);

 arma::mat irisTestData;
 data::Load("iris_test.csv", irisTestData, true);

  //normalize test data
 for(int i=0; i<irisTestData.n_rows; i++)
 {
  minVal = irisTestData.row(i).min();
  range  = irisTestData.row(i).max() - minVal;
  irisTestData.row(i) =  (irisTestData.row(i) - minVal)/range;
 }

   arma::mat irisTestLabels;
    data::Load("iris_test_labels.csv", irisTestLabels, true);

 BuildVanillaNetwork<>
(irisTrainData, irisTrainLabels, irisTestData, irisTestLabels, 3, 8, 70, 0.1);
}
BOOST_AUTO_TEST_SUITE_END();
