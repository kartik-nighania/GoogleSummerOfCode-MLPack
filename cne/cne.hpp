/**
 * @file cne.hpp
 * @author Marcus Edel
 * @author Kartik Nighania
 *
 * Conventional Neuro-evolution
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_CNE_CNE_HPP
#define MLPACK_CORE_OPTIMIZERS_CNE_CNE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

class CNE
{
 public:

  CNE(const size_t populationSize,
      const size_t maxGeneration,
      const double mutationRate, 
      const double mutationSize, 
      const double selectPercent);

  template<typename DecomposableFunctionType>
  double Optimize(DecomposableFunctionType& function, arma::mat& iterate);

 private:

  void Reproduce();
  void Mutate();
  void Cross(size_t mom, size_t dad, size_t dropout1, size_t dropout2);

  size_t populationSize;
  size_t maxGeneration;
  double mutationSize;
  double mutationRate;
  double selectPercent;

  arma::mat population;
  arma::vec fitnessValues;

  arma::uvec index;
};

} // namespace optimization
} // namespace mlpack

// Include implementation.
#include "cne_impl.hpp"

#endif
