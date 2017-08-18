#include <mlpack/core.hpp>

#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <time.h>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>

#include "cne.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;

using namespace mlpack::distribution;
using namespace mlpack::regression;

int main()
{
  double start = clock();
   // Load the datasets.
  arma::mat trainData;
  data::Load("iris_train.csv", trainData, true);

  arma::mat testData;
  data::Load("iris_test.csv", testData, true);

  arma::mat trainLabels;
  data::Load("iris_train_labels.csv", trainLabels, true);
  trainLabels += 1;

  arma::mat testLabels;
  data::Load("iris_test_labels.csv", testLabels, true);
  testLabels += 1;

  // Create vanilla network with 4 input, 4 hidden and 3 output nodes. 
  FFN<NegativeLogLikelihood<> > model;
  model.Add<Linear<> >(trainData.n_rows, 4);
  model.Add<SigmoidLayer<> >();
  model.Add<Linear<> >(4, 3);
  model.Add<LogSoftMax<> >();

  CNE opt(30, 200, 0.2, 0.2, 0.3);

  model.Train(trainData, trainLabels, opt);

  arma::mat predictionTemp1;
  model.Predict(testData, predictionTemp1);
  arma::mat prediction1 = arma::zeros<arma::mat>(1, predictionTemp1.n_cols);

  for (size_t i = 0; i < predictionTemp1.n_cols; ++i)
  {
    prediction1(i) = arma::as_scalar(arma::find(
        arma::max(predictionTemp1.col(i)) == predictionTemp1.col(i), 1)) + 1;
  }

  size_t error = 0;
  for (size_t i = 0; i < testData.n_cols; i++)
  {
    if (int(arma::as_scalar(prediction1.col(i))) ==
        int(arma::as_scalar(testLabels.col(i))))
    {
      error++;
    }
  }



  arma::mat train("1,0,0,1;1,0,1,0");
  arma::mat labels("1,1,2,2");

  // network with 2 input 2 hidden and 2 output layer
  FFN<NegativeLogLikelihood<> > network;

  network.Add<Linear<> >(2, 2);
  network.Add<SigmoidLayer<> >();
  network.Add<Linear<> >(2, 2);
  network.Add<LogSoftMax<> >();

  // CNE object
  CNE opt1(50, 5000, 0.1, 0.02, 0.2, 0.1);

  // Training the network with CNE
  network.Train(train, labels, opt1);

  // Predicting for the same train data
  arma::mat predictionTemp;
  network.Predict(train, predictionTemp);

  arma::mat prediction = arma::zeros<arma::mat>(1, predictionTemp.n_cols);

  for (size_t i = 0; i < predictionTemp.n_cols; ++i)
  {
    prediction(i) = arma::as_scalar(arma::find(
        arma::max(predictionTemp.col(i)) == predictionTemp.col(i), 1)) + 1;
  }


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
  arma::mat testData1(3, 1000);
  arma::Row<size_t> testResponses(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    testData1.col(i) = g1.Random();
    testResponses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    testData1.col(i) = g2.Random();
    testResponses[i] = 1;
  }

  CNE opt2(30, 500, 0.2, 0.2, 0.3, 100);

  LogisticRegression<> lr(shuffledData, shuffledResponses, opt2, 0.5);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses);
  cout << acc << " 100.0" << endl; // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData1, testResponses);
  cout << testAcc << " 100.0" << endl;

    double classificationError = 1 - double(error) / testData.n_cols;
  cout << classificationError << " <= " << 0.1 << endl;

    // 1 means 0 and 2 means 1 as the output to XOR
  cout << " 1 = " << prediction[0] << endl;
  cout << " 1 = " << prediction[1] << endl;
  cout << " 2 = " << prediction[2] << endl;
  cout << " 2 = " << prediction[3] << endl;
  
  double end = clock();

  cout << "TIME TAKEN " << (end - start)/CLOCKS_PER_SEC << endl;

return 0;
}