#include "/home/pkarouzakis/Documents/boost/include/boost/multiprecision/gmp.hpp"
#include "/home/pkarouzakis/Documents/boost/include/boost/math/constants/constants.hpp"
#include "/home/pkarouzakis/Documents/boost/include/boost/random/uniform_real.hpp"
#include "/home/pkarouzakis/Documents/boost/include/boost/random/variate_generator.hpp"
#include "/home/pkarouzakis/Documents/boost/include/boost/random/linear_congruential.hpp"
#include "/home/pkarouzakis/Documents/boost/include/boost/generator_iterator.hpp"

#include <iostream>
#include <vector>


using namespace std;

template<typename value_type, typename function_type>
inline value_type sums_f(vector<value_type> ti, vector<value_type> freqi,
                           const value_type ppi, function_type func)
{
  unsigned N = 80U;
  value_type ret = 0.0;

  for(int k = 0; k < N; k++){
    ret += func(ti[k]*freqi[k]*ppi); 
  }
  return ret;
}



int main(int, char**){
  // seed 
  base_generator_type generator(47);  
  
  using boost::math::constants::pi;
  using namespace boost_multiprecision;
  std::vector<mpf_float_500> vf_mp, vt_mp;
  std::vector<double> vf, vt;
  typedef boost::uniform_real<> freq_dist(0.0, 10.0);
  typedef boost::variate_generator<base_generator_type&, distribution_type> gen_type;
  typedef boost::uniform_real<> time_dist(0, 1000); 

  boost::variate_generator<base_generator_type&, boost::uniform_real<> > uni_f(generator, uni_dist);
  boost::variate_generator<base_generator_type&, boost::uniform_real<> > uni_t(generator, time_dist);
  
  // Get random numbers for frequencies and time series
  for(int i = 0; i < 80; i++){
    vf.push_back(uni_f());
    vt.push_back(uni_t());
    vf_mp.push_back(uni_f());
    vf_mp.push_back(uni_t());
  } 
  
  mpf_t f;
  mpf_init(f);
  mpf_set(f, f_mp.backend().data());
  mpf_clear(f);
  return 0;

}
