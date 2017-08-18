#include <iostream>
#include <mlpack/core.hpp>
#include <armadillo>

using namespace std;
using namespace arma;

int main()
{   
  arma::cube x(3,3,3);
  x.fill(4);

  for(int i=0; i<3; i++)
   {   for(int j=0; j<3; j++)
      {
         for(int k=0; k<3; k++)
         {
            cout << x(k,j,i) << " ";
         }

         cout << endl;
      }
     
     cout << endl << endl;
   }
   
   x.slice(0).fill(1);
   x.slice(0)(8) = x.slice(1)(1);

    for(int i=0; i<3; i++)
   {   for(int j=0; j<3; j++)
      {
         for(int k=0; k<3; k++)
         {
            cout << x(k,j,i) << " ";
         }

         cout << endl;
      }
     
     cout << endl << endl;
   }

   cout << x.n_elem << endl;
	return 0;
}