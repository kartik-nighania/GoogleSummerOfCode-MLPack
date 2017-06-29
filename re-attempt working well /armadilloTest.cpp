#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

 void gauss(arma::mat& x)
   {
   		cout << endl << "here it is " << x[1] << endl;
   }
int main()
{	

	const mat ok("0; 0.44");
	mat pk(3,2);
	cout << endl << ok[0] <<  " " << pk(0,0)<< endl  ;

	double z = 1.2e-100;

	cout << z;

    B = new double*[2];

    B[0] = new double[3];
    B[1] = new double[3];

    B[0][0] = 5;
    


	//double r=10;
	//gauss(ok);

	/* double d = gauss();
	double s = gauss();
	cout << d << " " << s << endl;
	*/

	//cout << endl << ok[1] << " " << ok[0] << endl ;

	//cout << arma::size(ok);
	return 0;
}
