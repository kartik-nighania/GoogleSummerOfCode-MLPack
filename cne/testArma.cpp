#include <iostream>
#include <mlpack/core.hpp>

using namespace std;
using namespace arma;

int main()
{   mlpack::math::RandomSeed(std::time(NULL));
    arma::uvec indices = arma::shuffle(arma::linspace<arma::uvec>(0, 10, 11));
    arma::uvec indice1 = arma::shuffle(arma::linspace<arma::uvec>(0, 10, 11));


   arma::mat x(3,3);
   x.randu();

   cout << endl;
   for(int i=0; i<3; i++)
   {
   	for(int j=0; j<3; j++)
   	{
   		cout << x(i,j) << " ";
   	}

   	cout << endl;
   }


   arma::mat y = randu<mat>(size(x));

      cout << endl;
   for(int i=0; i<3; i++)
   {
   	for(int j=0; j<3; j++)
   	{
   		cout << y(i,j) << " ";
   	}

   	cout << endl;
   }

   y.elem(find(y > 0.3)).fill(arma::randu());

         cout << endl;
   for(int i=0; i<3; i++)
   {
   	for(int j=0; j<3; j++)
   	{
   		cout << y(i,j) << " ";
   	}

   	cout << endl;
   }
	

	return 0;
}