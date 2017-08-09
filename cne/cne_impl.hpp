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
         const size_t maxGeneration, 
         const double mutationRate, 
         const double mutationSize, 
         const double selectPercent) :
         populationSize(populationSize),
         maxGeneration(maxGeneration),
         mutationRate(mutationRate),
         mutationSize(mutationSize),
         selectPercent(selectPercent)
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

    Log::Info << "CNE initialized successfully. Optimization started " << std::endl;

    // Iterate till max number of generations
    for (size_t gen = 1; gen <= maxGeneration; gen++)
    {   
    	// calculate fitness values of all candidates
    	for (size_t i = 0; i < populationSize; i++)
         {
           	// select a candidate
           	parameters = population.row(i).t();
           	// Insert the parameters in the neural network
           	function.parameters() = parameters;

           	// find the fitness
           	for(int j = 0; j < numFun; j++)
           		fitness += function.Evaluate(parameters, j);

           	// Save fitness values
           	fitnessValues[i] = fitness;
           	fitness = 0;
         }
        
            Log::Info << "Generation number: " << gen << " best fitness = "
            << fitnessValues.max() << std::endl;
            
            // ******************** termination criteria and then output TODO AND REMAINS
            
            // create the next generation of species
            Reproduce();
    }
}

CNE::Reproduce()
{   
  // sort fitness value. The smaller the better
  index = arma::sort_index(fitnessValues);

  //find the number of elite percentage
  size_t numElite = floor(selectPercent * populationSize);
  
  // making sure we have even number of candidates to remove and create
  if((populationSize - numElite) % 2 != 0) numElite++;
  
  for(size_t i = numElite; num < populationSize-1; i++)
  {
  	// select 2 parents from the elite group randomly [0, numElite)
  	size_t mom = RandInt(0, numElite);
  	size_t dad = RandInt(0, numElite);

  	// crossover parents to create 2 childs replacing the droped out candidates
    Cross(mom, dad, i, i+1]);
  }

  // mutate the weights with small noise.
  // This is done to bring change in the next generation.
  // Then look for improvement in the next generation and improve upon it iteratively.
  Mutate();   

}

CNE::Cross(size_t mom, size_t dad, size_t child1, size_t child2)
{
  // find the index of these candidates in the population matrix
  mom = index[mom];
  dad = index[dad];
  child1 = index[child1];
  child2 = index[child2];
  
  // remove the cadidates and instead place the parents
  population.row(child1) = population.row(mom);
  population.row(child2) = population.row(dad);
  
  // randomly alter mom and dad genome data to get two childs
  for(size_t i = 0; i < population.n_rows; i++)
  {
  	// select a random value from the normal distribution
    double rand = mlpack::math::RandNormal();

    // use it to alter the weights of the childs
    if(rand > 0)
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
  
}

} // namespace optimization
} // namespace mlpack

#endif
