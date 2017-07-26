#include <iostream>
#include <cmath>

#include <mlpack/core.hpp>

using namespace std;
using namespace arma;
using namespace mlpack;

template<typename T>
class Random
{
  // variables for uniform()
  long int startseed;
  long int aktseed;
  long int aktrand;
  long int rgrand[32];
  // variables for gauss()
  bool stored;
  T hold;

 public:
  /**
   * @param seed use clock if 0
   */
  Random(long unsigned seed = 0) : hold(0.0)
  {
    stored = false;
    if (seed < 1)
    {
      long int t = 100*time(0) + clock();
      seed = (long unsigned) (t < 0 ? -t : t);
    }
    start(seed);
  }
  /**
   * @param seed 0 == 1
   */
  void start(long unsigned seed)
  {
    stored = false;
    startseed = seed;
    if (seed < 1) seed = 1;
    aktseed = seed;
    for (int i = 39; i >= 0; --i)
    {
      long tmp = aktseed / 127773;
      aktseed = 16807* (aktseed - tmp* 127773) - 2836* tmp;
      if (aktseed < 0) aktseed += 2147483647;
      if (i < 32) rgrand[i] = aktseed;
    }
    aktrand = rgrand[0];
  }
  /**
   * @return (0,1)-normally distributed random number
   */
  T gauss(void)
  {
    if (stored)
    {
      stored = false;
      return hold;
    }
    stored = true;
    T x1, x2, rquad;
    do {
      x1 = 2.0*uniform() - 1.0;
      x2 = 2.0*uniform() - 1.0;
      rquad = x1*x1 + x2*x2;
    } while (rquad >= 1 || rquad <= 0);
    const register T fac = std::sqrt(T(-2)*std::log(rquad)/rquad);
    hold = fac*x1;
    return fac*x2;
  }
  /**
   * @return (0,1)-uniform distributed random number
   */
  T uniform(void)
  {
    long tmp = aktseed / 127773;
    aktseed = 16807 * (aktseed - tmp * 127773) - 2836 * tmp;
    if (aktseed < 0)
      aktseed += 2147483647;
    tmp = aktrand / 67108865;
    aktrand = rgrand[tmp];
    rgrand[tmp] = aktseed;
    return (T) aktrand / T(2.147483647e9);
  }
};


int main()

{	
	Random<double> rand;
	math::RandomSeed(std::time(NULL));
	cout << "here we go " << math::Random() << "  and  " << rand.gauss() << endl;
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
