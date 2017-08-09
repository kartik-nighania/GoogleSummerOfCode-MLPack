#include <mlpack/core.hpp>

using namespace std;
using namespace arma; // for armadillo library
using namespace mlpack::optimization; // for the cmaes class

// User made function class
// The details about making this class is explained above in its own seperate section.

class cmaesTestFunction
{
  public:

  int NumFunctions(void) { return 3; }

   double Evaluate(arma::mat& coordinates, ssize_t i)
  {
    switch (i)
        {
        case 0: return -exp(-abs(coordinates[0]));
        break;
        
        case 1: return pow(coordinates[1], 2);
        break;
        
        case 2: return pow(coordinates[2], 4) + 3 * pow(coordinates[2], 2);
        }
  }
  
};

int main()
{
  // the above made class object
  cmaesTestFunction test;

  // the CMAES optimizer object
  // The parameters of the class are explained above in its seperate section 
  CMAES opt(3, 0.5, 0.3, 10000, 1e-12, 1e-13);
 
  // armadillo matrix to get the optimized dimension values
  arma::mat coordinates(3,1);

  // calling the Optimizer's method to optimize. 
  double result = opt.Optimize(test, coordinates);

  // printing out the results
  cout << result << endl;
  for(int i=0; i<3; i++) cout << coordinates[i] << " ";

return 0;
}