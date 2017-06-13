/**
 * @file test_function.cpp
 * @author Ryan Curtin
 *
 * Implementation of very simple test for 
 * COVARIANCE MATRIX ADAPTATION EVOLUTION STRATEGY
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "test_function.hpp"

using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

size_t cmaesTestFunction::NumFunctions() const { return 3; }

arma::mat cmaesTestFunction::GetInitialPoint() const { return arma::mat("6; -45.6; 6.2"); }

arma::mat cmaesTestFunction::GetInitialStdDev() const { return arma::mat("3; 3; 3"); }

double camesTestFunction::Evaluate(const arma::mat& coordinates, const size_t i)
    const
{
  switch (i)
  {
    case 0:
      return -std::exp(-std::abs(coordinates[0]));

    case 1:
      return std::pow(coordinates[1], 2);

    case 2:
      return std::pow(coordinates[2], 4) + 3 * std::pow(coordinates[2], 2);

    default:
      return 0;
  }
}
