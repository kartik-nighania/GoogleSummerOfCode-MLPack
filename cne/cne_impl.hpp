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

CNE::CNE(const size_t populationSize,
         const size_t maxGenerations,
         const double mutationProb,
         const double mutationSize,
         const double selectPercent,
         const double tolerance,
         const double objectiveChange) :
    populationSize(populationSize),
    maxGenerations(maxGenerations),
    mutationProb(mutationProb),
    mutationSize(mutationSize),
    selectPercent(selectPercent),
    tolerance(tolerance),
    objectiveChange(objectiveChange),
    numElite(0)
{ /* Nothing to do here. */ }

//! Optimize the function.
template<typename DecomposableFunctionType>
double CNE::Optimize(DecomposableFunctionType& function, arma::mat& iterate)
{
  // Make sure for evolution to work at least four candidates are present.
  if (populationSize < 4)
    throw std::logic_error("Population size should be atleast 4!");

  // Find the number of elite canditates from population.
  numElite = floor(selectPercent * populationSize);

  // Making sure we have even number of candidates to remove and create.
  if ((populationSize - numElite) % 2 != 0) numElite--;

  // Terminate if two parents can not be created.
  if (numElite < 2)
    throw std::logic_error("Increase selection percentage.");

  // Terminate if at least two childs are not possible.
  if ((populationSize - numElite) < 2 )
    throw std::logic_error("Increase population size.");

  // Get the number of functions to iterate.
  const size_t numFun = function.NumFunctions();

  // Set the population size and fill random values [0,1].
  population = arma::randu(iterate.n_rows, populationSize);

  // initializing helper variables.
  fitnessValues.set_size(populationSize);
  double fitness = 0;
  double lastBestFitness = 0;

  Log::Info << "CNE initialized successfully. Optimization started"
      << std::endl;

  // Find the fitness before optimization using given iterate parameters.
  for (size_t j = 0; j < numFun; j++)
    lastBestFitness += function.Evaluate(iterate, j);

  // Iterate until maximum number of generations is obtained.
  for (size_t gen = 1; gen <= maxGenerations; gen++)
  {
    // Calculating fitness values of all candidates.
    for (size_t i = 0; i < populationSize; i++)
    {
       // Select a candidate and insert the parameters in the function.
       iterate = population.col(i);

       // Find fitness of candidate.
       for (size_t j = 0; j < numFun; j++)
         fitness += function.Evaluate(iterate, j);

       // Save fitness value of the evaluated candidate.
       fitnessValues[i] = fitness;
       fitness = 0;
    }

      Log::Info << "Generation number: " << gen << " best fitness = "
          << fitnessValues.min() << std::endl;

      // Create next generation of species.
      Reproduce();

      // Check for termination criteria.
      if (tolerance != DBL_MIN && tolerance >= fitnessValues.min())
      {
          Log::Info << "Terminating. Given fitness criteria " << tolerance
              << " > " << fitnessValues.min() << std::endl;
          break;
      }

      // Check for termination criteria.
      if (objectiveChange != DBL_MIN &&
        (lastBestFitness - fitnessValues.min()) < objectiveChange)
      {
          Log::Info << "Terminating. Fitness History change "
              << (lastBestFitness - fitnessValues.min())
              << " < " << objectiveChange << std::endl;
          break;
      }

      // Store the best fitness of present generation.
      lastBestFitness = fitnessValues.min();
  }

  // Set the best candidate into the network parameters.
  iterate = population.col(index(0));

  // Find the objective function value from the best candidate found.
  for (size_t j = 0; j < numFun; j++)
    fitness += function.Evaluate(iterate, j);

  return fitness;
}

//! Reproduce candidates to create the next generation.
void CNE::Reproduce()
{
  // Sort fitness values. Smaller fitness value means better performance.
  index = arma::sort_index(fitnessValues);

  // First parent.
  size_t mom;

  // Second parent.
  size_t dad;

  for (size_t i = numElite; i < populationSize - 1; i++)
  {
    // Select 2 different parents from elite group randomly [0, numElite).
    do {
         mom = mlpack::math::RandInt(0, numElite);
         dad = mlpack::math::RandInt(0, numElite);
       } while (mom == dad);

    // Parents generates 2 childs replacing the droped out candidates.
    // Also finding the index of these candidates in the population matrix.
    Crossover(index[mom], index[dad], index[i], index[i + 1]);
  }

  // Mutating the weights with small noise values.
  // This is done to bring change in the next generation.
  Mutate();
}

//! Crossover parents to create new childs.
void CNE::Crossover(const size_t mom,
                    const size_t dad,
                    const size_t child1,
                    const size_t child2)
{
  // Replace the cadidates with parents at their place.
  population.col(child1) = population.col(mom);
  population.col(child2) = population.col(dad);

  double rand;
  // Randomly alter mom and dad genome weights to get two different childs.
  for (size_t i = 0; i < population.n_rows; i++)
  {
    // Selecting a random value between 0 and 1.
    rand = mlpack::math::Random();

    // Using it to alter the weights of the childs.
    if (rand > 0.5)
    {
      population(i, child1) = population(i, mom);
      population(i, child2) = population(i, dad);
    }
    else
    {
      population(i, child1) = population(i, dad);
      population(i, child2) = population(i, mom);
    }
  }
}

//! Modify weights with some noise for the evolution of next generation.
void CNE::Mutate()
{
  // Helper variables.
  double noise;
  double delta;

  // Mutate the whole matrix with the given rate and probability.
  // The best candidate is not altered.
  for (size_t i = 1; i < populationSize; i++)
  {
    for (size_t j = 0; j < population.n_rows; j++)
    {
      noise = mlpack::math::Random();

      if (noise < mutationProb)
      {
        delta = mlpack::math::RandNormal(0, mutationSize);
        population(j, index[i]) += delta;
      }
    }
  }
}

} // namespace optimization
} // namespace mlpack

#endif
