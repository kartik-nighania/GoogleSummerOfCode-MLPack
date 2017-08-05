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

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;


using namespace mlpack::distribution;
using namespace mlpack::regression;

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
  FFN<NegativeLogLikelihood<> > model;
  model.Add<Linear<> >(trainData.n_rows, hiddenLayerSize);
  model.Add<SigmoidLayer<> >();
  model.Add<Linear<> >(hiddenLayerSize, outputSize);
  model.Add<LogSoftMax<> >();


  int dim = trainData.n_rows * hiddenLayerSize +  hiddenLayerSize * outputSize + 2;
  CMAES opt(dim, 0.5, 0.3, 10000, 1e-16, 1e-16);

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
  cout << classificationError << " < = " << classificationErrorThreshold << endl;
}

int main()
{ 
 mlpack::math::RandomSeed(std::time(NULL));

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

return 0;
}

