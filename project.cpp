#include <iostream>
#include <fstream>
#include <ctime>            // std::time

#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/generator_iterator.hpp>


#include <boost/multiprecision/gmp.hpp>
#include <boost/math/constants/constants.hpp>

#include <vector>


typedef boost::minstd_rand base_generator_type;


template<typename value_type, typename function_type>
inline value_type sums_f(std::vector<value_type> ti, std::vector<value_type> freqi,
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
  using namespace boost::multiprecision;
  std::vector<mpf_float_500> vf_mp, vt_mp;
  std::vector<double> vfd, vtd;
  std::vector<float> vff, vtf;
  boost::uniform_real<> freq_dist(0.0, 10.0);
  
	boost::uniform_real<> time_dist(0, 1000); 

  boost::variate_generator<base_generator_type&, boost::uniform_real<> > uni_f(generator, freq_dist);
  boost::variate_generator<base_generator_type&, boost::uniform_real<> > uni_t(generator, time_dist);
 unsigned PopulationSize = 5;
 unsigned Iterations = 10;

	for(int iter = 0; iter < Iterations; iter++){
	 	for(int pop = 0; pop < PopulationSize;	pop++){
			// generate a new solution
			// Get random numbers for frequencies and time series
			for(int i = 0; i < 80; i++){
				vfd.push_back(uni_f());
				vtd.push_back(uni_t());
				vff.push_back(uni_f());
				vtf.push_back(uni_t());
				vf_mp.push_back(uni_f());
				vf_mp.push_back(uni_t());
			} 
		 
			const float sin_f = 
			 sums_f(vtf, vff, pi<float>(), 
				 static_cast<float(*)(float)>(std::sin)
			);
			
			const double sin_d = 
			 sums_f(vtd, vfd, pi<double>(), 
				 static_cast<double(*)(double)>(std::sin)
				);

			const mpf_float_500 sin_mp = 
				sums_f(vt_mp, vf_mp, pi<mpf_float_500>(), 
				[](const mpf_float_500& x) -> mpf_float_500 
				{
					return sin(x);
				}
				);
	 
	 }
 }

  return 0;

}
