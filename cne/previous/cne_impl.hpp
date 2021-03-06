/**
 * @file cne_impl.hpp
 * @author Marcus Edel
 * @author Kartik Nighania
 *
 * Conventional Neural Evolution
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_CNE_CNE_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_CNE_CNE_IMPL_HPP

#include "cne.hpp"

namespace mlpack {
namespace optimization {

CNE::CNE(
    const size_t populationSize,
    const size_t maxGenerations,
    const double mutationProb,
    const double mutationSize,
    const double selectPercent,
    const double finalValue,
    const double fitnessHist) :
    populationSize(populationSize),
    maxGenerations(maxGenerations),
    mutationProb(mutationProb),
    mutationSize(mutationSize),
    selectPercent(selectPercent),
    finalValue(finalValue),
    fitnessHist(fitnessHist)
{ /* Nothing to do here. */ }

//! Optimize the function
template<typename DecomposableFunctionType>
double CNE::Optimize(
    DecomposableFunctionType& function,
    arma::mat& iterate)
{
  // Get the number of functions to iterate
  const size_t numFun = function.NumFunctions();

  // Set the population size and fill random values [0,1]
  population = arma::randu(populationSize, iterate.n_rows);

  // initialize helper variables
  fitnessValues.set_size(populationSize);
  double fitness = 0;
  double lastBestFitness;

  std::cout << "CNE initialized successfully. Optimization started"
      << std::endl;

  // Iterate till max number of generations
  for (size_t gen = 1; gen <= maxGenerations; gen++)
  {
    // calculate fitness values of all candidates
    for (size_t i = 0; i < populationSize; i++)
    {
       // select a candidate and insert the parameters in the function
       iterate = population.row(i).t();

       // find the fitness
       for (size_t j = 0; j < numFun; j++)
         fitness += function.Evaluate(iterate, j);

       // Save fitness values
       fitnessValues[i] = fitness;
       fitness = 0;
    }

      std::cout << "Generation number: " << gen << " best fitness = "
          << fitnessValues.min() << std::endl;

      // create the next generation of species
      Reproduce();

      // check for termination
      if (finalValue != DBL_MIN && finalValue >= fitnessValues.min())
      {
          std::cout << "Terminating. Given fitness criteria " << finalValue
              << " > " << fitnessValues.min() << std::endl;
          break;
      }

      // check for termination
      if (fitnessHist != DBL_MIN && gen != 1 &&
        (lastBestFitness - fitnessValues.min()) < fitnessHist)
      {
          std::cout << "Terminating. Fitness History change "
              << (lastBestFitness - fitnessValues.min())
          << " < " << fitnessHist << std::endl;
          break;
      }

      // Store the best fitness of this generation before update
      lastBestFitness = fitnessValues.min();
  }

  // Set the best candidate into the network
  iterate = population.submat(index[0], 0, index[0], iterate.n_rows - 1).t();

  // find the best fitness
  for (size_t j = 0; j < numFun; j++)
      fitness += function.Evaluate(iterate, j);

  return fitness;
}

//! Reproduce candidates to create the next generation
void CNE::Reproduce()
{
  // Sort fitness value. The smaller the better
  index = arma::sort_index(fitnessValues);

  // Find the number of elite percentage
  size_t numElite = floor(selectPercent * populationSize);

  // Making sure we have even number of candidates to remove and create
  if ((populationSize - numElite) % 2 != 0) numElite++;

  for (size_t i = numElite; i < populationSize-1; i++)
  {
    // Select 2 parents from the elite group randomly [0, numElite)
    size_t mom = mlpack::math::RandInt(0, numElite);
    size_t dad = mlpack::math::RandInt(0, numElite);

    // Crossover parents to create 2 childs replacing the droped out candidates
    Crossover(mom, dad, i, i+1);
  }

  // Mutate the weights with small noise.
  // This is done to bring change in the next generation.
  Mutate();
}

//! Crossover parents to create new childs
void CNE::Crossover(size_t mom, size_t dad, size_t child1, size_t child2)
{
  // find the index of these candidates in the population matrix
  mom = index[mom];
  dad = index[dad];
  child1 = index[child1];
  child2 = index[child2];

  // Remove the cadidates and instead place the parents
  population.row(child1) = population.row(mom);
  population.row(child2) = population.row(dad);

  double rand;
  // Randomly alter mom and dad genome data to get two childs
  for (size_t i = 0; i < population.n_cols; i++)
  {
    // Select a random value between 0 and 1
    rand = mlpack::math::Random();

    // Use it to alter the weights of the childs
    if (rand > 0.5)
    {
      population(child1, i) = population(mom, i);
      population(child2, i) = population(dad, i);
    }
    else
    {
      population(child1, i) = population(dad, i);
      population(child2, i) = population(mom, i);
    }
  }
}

//! Function to modify weights for the evolution of next generation
void CNE::Mutate()
{
  // Helper variables
  double noise;
  double delta;

  // Mutate the whole matrix with the given rate and probability
  // Note: The best candidate is not altered
  for (size_t i = 1; i < populationSize; i++)
  {
    for (size_t j = 0; j < population.n_cols; j++)
    {
      noise = mlpack::math::Random();

      if (noise < mutationProb)
      {
        delta = mlpack::math::RandNormal(0, mutationSize);
        population(index[i], j) += delta;
      }
    }

  }
}

} // namespace optimization
} // namespace mlpack

#endif
