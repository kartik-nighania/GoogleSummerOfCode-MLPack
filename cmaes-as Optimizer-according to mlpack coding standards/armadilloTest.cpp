#include <iostream>
#include <armadillo>

using namespace std;

double *pop;

 void gauss(arma::vec& iterate)
   {
	pop[0] = 5;
	pop[1] = 3;
	pop[2] = 4443;
    
    iterate[0] = pop[0];
	iterate[1] = pop[1]; 
	iterate[2] = pop[2];
   }

int main()
{	
   arma::vec iter(3);

  gauss(iter);
  // iter(1)=55;
	cout << iter[1] << endl;

	return 0;
}
