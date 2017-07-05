#include <iostream>
#include <armadillo>
#include <cmath>

using namespace std;
using namespace arma;

 void gauss(arma::mat& x)
   {
   		//cout << endl << "here it is " << x[1] << endl;
   }

int main()
{	
	vec st(3);
	st[0] = 3; st[1] = 1; st[2] = 2;
	//cout << "this is the norm : " << pow(norm(st),2) << endl;

	for (size_t i = 0; i < 5; ++i)
    std::cout << arma::randu(1) << std::endl;
	return 0;
}
