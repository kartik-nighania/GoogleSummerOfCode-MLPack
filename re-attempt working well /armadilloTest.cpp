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
	vec st(6);
	st[0] = 0; st[1] = 2; st[2] = 2; st[3] = 23; st[4] = 231; st[5] = 324; st[6] = 8888;

	cout << "here it is " << st;
	
	arma::mat start(1,3); start.fill(0.5);

	start[0] = 1234321;

	cout << start[0];

    std::cout << endl <<"before " << endl << 10*st << std::endl;

	uvec x = (find(st < 5));
	double d =x.n_rows;

    std::cout <<"after "<< d << std::endl;
	return 0;
}
