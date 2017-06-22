#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

 double gauss(void)
   {
    arma::mat gauss = arma::randu<arma::mat>(1,1);
    return gauss(0);

   }
int main()
{
	double r=10;

	double d = gauss();
	double s = gauss();
	cout << d << " " << s << endl; 
	return 0;
}
