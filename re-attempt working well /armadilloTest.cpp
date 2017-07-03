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
	//cout << endl << ok[0] <<  " " << pk(0,0) << endl  ;

	double z = 1.2e-100;

	//cout << z;
	double** population;
 	population = new double*[4];
    for (int i = 0; i < 4; ++i)
    {
      population[i] = new double[5];
      population[i][0] = 3.4;
      //population[i]++;
      for (int j = 0; j < 4; j++) population[i][j] = 0.664;
    }

//cout << endl;

for(int i=0; i<4; i++)
{
	for(int j=0; j<4; j++) cout << population[i][j] << " ";
	//cout << endl; 
}

arma::vec row;
row.set_size(3);

for(int i= (int)row.size(); i>=0; i--)
{
	//cout << "done " << endl;
}


	//double r=10;
	//gauss(ok);

	/* double d = gauss();
	double s = gauss();
	cout << d << " " << s << endl;
	*/

	//cout << endl << ok[1] << " " << ok[0] << endl ;

	//cout << arma::size(ok);

vec a = randu<vec>(10);
vec b = randu<vec>(10);

double sww = dot(a,b);


 cout << "gotteh dot " << sww << endl;
	return 0;
}
