/**
 * @file cne_impl.hpp
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
         const double finalValue,
         const double fitnessHist) :
         populationSize(populationSize),
         maxGenerations(maxGenerations),
         mutationProb(mutationProb),
         mutationSize(mutationSize),
         selectPercent(selectPercent),
         finalValue(finalValue),
         fitnessHist(fitnessHist)
         { }

template<typename DecomposableFunctionType>
double CNE::Optimize(
    DecomposableFunctionType& function,
    arma::mat& answer)
{
  // Get the number of functions to iterate
  size_t numFun = function.NumFunctions();

  // Set the population size and fill random values [0,1]
  populations.set_size(populationSize, answer.n_rows);
  population.randu();

  // initialize helper variables
  arma::vec parameters(answer.n_rows);
  fitnessValues.set_size(populationSize);
  double fitness = 0;

  Log::Info << "CNE initialized successfully. Optimization started "
  << std::endl;

  // Iterate till max number of generations
  for (size_t gen = 1; gen <= maxGenerations; gen++)
  {
  // calculate fitness values of all candidates
  for (size_t i = 0; i < populationSize; i++)
     {
       // select a candidate
       parameters = population.row(i).t();
       // Insert the parameters in the function
       answer = parameters;

       // find the fitness
       for (int j = 0; j < numFun; j++)
       fitness += function.Evaluate(parameters, j);

       // Save fitness values
       fitnessValues[i] = fitness;
       fitness = 0;
     }

        Log::Info << "Generation number: " << gen << " best fitness = "
        << fitnessValues.max() << std::endl;

        // see that the answer final call is one iteration back to this.

        // create the next generation of species
        Reproduce();
  }

  // Set the best candidate into the network
  answer = population.submat(index[0], 0, index[0], populationSize-1);

  // find the best fitness
  for (int j = 0; j < numFun; j++)
      fitness += function.Evaluate(answer, j);

  return fitness;
}

CNE::Reproduce()
{
  // Sort fitness value. The smaller the better
  index = arma::sort_index(fitnessValues);

  // Find the number of elite percentage
  size_t numElite = floor(selectPercent * populationSize);

  // Making sure we have even number of candidates to remove and create
  if ((populationSize - numElite) % 2 != 0) numElite++;

  for (size_t i = numElite; num < populationSize-1; i++)
  {
    // Select 2 parents from the elite group randomly [0, numElite)
    size_t mom = RandInt(0, numElite);
    size_t dad = RandInt(0, numElite);

    // Crossover parents to create 2 childs replacing the droped out candidates
    Crossover(mom, dad, i, i+1]);
  }

  // Mutate the weights with small noise.
  // This is done to bring change in the next generation.
  // Then look for improvement in the next generation and improve upon it iteratively.
  Mutate();
}

CNE::Crossover(size_t mom, size_t dad, size_t child1, size_t child2)
{
  // find the index of these candidates in the population matrix
  mom = index[mom];
  dad = index[dad];
  child1 = index[child1];
  child2 = index[child2];

  // Remove the cadidates and instead place the parents
  population.row(child1) = population.row(mom);
  population.row(child2) = population.row(dad);

  // Randomly alter mom and dad genome data to get two childs
  for (size_t i = 0; i < population.n_cols; i++)
  {
    // Select a random value from the normal distribution
    double rand = mlpack::math::RandNormal();

    // Use it to alter the weights of the childs
    if (rand > 0)
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


CNE::Mutate()
{
  // Mutate the whole matrix with the given rate and probability
  // Note: The best candidate is not altered
  for (size_t i = 1; i < populationSize; i++)
  {
    for (size_t j = 0; j < population.n_cols; j++)
   {
      double noise = mlpack::math::Random();

      if (noise < mutationProb)
      {
        double delta = mlpack::math::RandNormal(0, mutationSize);
        population(index[i], j) += delta;
      }
    }
  }
}

} // namespace optimization
} // namespace mlpack

#endif
