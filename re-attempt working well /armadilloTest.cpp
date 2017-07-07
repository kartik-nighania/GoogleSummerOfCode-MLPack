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
	st = arma::sqrt(st);
	//cout << "this is the norm : " << pow(norm(st),2) << endl;
    std::cout << st << std::endl;
	return 0;
}
