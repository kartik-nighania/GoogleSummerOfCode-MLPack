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

#include "cmaes.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::optimization;


  class rosenbrock
  {
    public:
    
    double NumFunctions(){return 2;}

    double Evaluate(arma::mat& x, int i)
    {
      if(i==0) return 100.*pow((pow((x[0]),2)-x[1]),2);
      if(i==1) return pow((6.-x[0]),2);

    }

  };

int main()
{ 
mlpack::math::RandomSeed(std::time(NULL));
  
  rosenbrock test;
  size_t N = 2;

  arma::mat start(N,1); start.fill(0.5); 
  arma::mat initialStdDeviations(N,1); initialStdDeviations.fill(0.3);

  CMAES s(N,start,initialStdDeviations,10000,1e-18);

  arma::mat coordinates(N,1);
  double result = s.Optimize(test, coordinates);

  cout << endl << result << endl;
  cout << coordinates[0] << endl;
  cout << coordinates[1] << endl;

  
return 0;
}

