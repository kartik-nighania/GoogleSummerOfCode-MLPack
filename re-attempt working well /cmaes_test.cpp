#include <mlpack/core.hpp>
#include <armadillo>

#include "cmaes.hpp"
#include "test_function.hpp"

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

  return 0;

}


