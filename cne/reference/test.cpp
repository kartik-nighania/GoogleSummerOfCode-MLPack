#include <mlpack/core.hpp>

#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <time.h>

#include "cne.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;

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

  // RMSProp opt(0.01, 0.88, 1e-8, maxEpochs * trainData.n_cols, -1);
   CNE opt(30, 150, 0.2, 0.2, 0.3);

  double start = clock();
  model.Train(trainData, trainLabels, opt);
  double end = clock();

  cout << "TIME ELAPSED " << (double)(end - start)/CLOCKS_PER_SEC << endl;

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
  std::cout << classificationError << " <= " << classificationErrorThreshold << std::endl;
}

int main()
{
   // Load the dataset.
  arma::mat dataset;
  data::Load("thyroid_train.csv", dataset, true);

  arma::mat trainData = dataset.submat(0, 0, dataset.n_rows - 4,
      dataset.n_cols - 1);

  arma::mat trainLabelsTemp = dataset.submat(dataset.n_rows - 3, 0,
      dataset.n_rows - 1, dataset.n_cols - 1);
  arma::mat trainLabels = arma::zeros<arma::mat>(1, trainLabelsTemp.n_cols);
  for (size_t i = 0; i < trainLabelsTemp.n_cols; ++i)
  {
    trainLabels(i) = arma::as_scalar(arma::find(
        arma::max(trainLabelsTemp.col(i)) == trainLabelsTemp.col(i), 1)) + 1;
  }

  data::Load("thyroid_test.csv", dataset, true);

  arma::mat testData = dataset.submat(0, 0, dataset.n_rows - 4,
      dataset.n_cols - 1);

  arma::mat testLabelsTemp = dataset.submat(dataset.n_rows - 3, 0,
      dataset.n_rows - 1, dataset.n_cols - 1);

  arma::mat testLabels = arma::zeros<arma::mat>(1, testLabelsTemp.n_cols);
  for (size_t i = 0; i < testLabels.n_cols; ++i)
  {
    testLabels(i) = arma::as_scalar(arma::find(
        arma::max(testLabelsTemp.col(i)) == testLabelsTemp.col(i), 1)) + 1;
  }

  // Vanilla neural net with logistic activation function.
  // Because 92 percent of the patients are not hyperthyroid the neural
  // network must be significant better than 92%.
  BuildVanillaNetwork<>
      (trainData, trainLabels, testData, testLabels, 3, 8, 70, 0.1);

return 0;
}