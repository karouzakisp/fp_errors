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

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>


#include <vector>
#include <cmath>

#include "gnuplot-iostream/gnuplot-iostream.h"

#define NUM_PARTICLES 35
#define DIMENSION 80 

#define C1_MAX 2.5
#define C2_MAX 2.5
#define C1_MIN 0.5
#define C2_MIN 0.5

#define FREQ_MAX 10.0
#define TIME_MAX 1000.0


// HYPER-PARAMETERS
#define DELTA 0.295
#define ALPHA 3.45

const int vel_max_x = ((DELTA)*(FREQ_MAX));
const int vel_max_y = ((DELTA)*(TIME_MAX));

typedef struct{
  int successes;
  int failures;
  int e_s;
  int e_f;
  double factor; 
}scaling_factor_info_t;

typedef struct {
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> velX;
  std::vector<double> velY;
  unsigned personalBest;
}particle_t;


typedef struct {
  std::vector<boost::multiprecision::mpf_float_500> vf_mp_t, vt_mp_t;
  std::vector<double> vfd_t, vtd_t;
  std::vector<float> vff_t, vtf_t;

}vectors_t;

typedef boost::minstd_rand base_generator_type;

std::pair<std::vector<double>, std::vector<double>> getPosition(const particle_t &Particle) {
  return std::make_pair(Particle.x, Particle.y);    
}

inline double clamp_velocity(double max_vel, int iteration, int max_iterations){
  return (1 - (std::pow(iteration/max_iterations, ALPHA))) * max_vel; 
}

inline double cognitive_comp(int iteration, int max_iteration){
  return (C1_MIN - C1_MAX)*(iteration/max_iteration) + C1_MAX;
}
inline double social_comp(int iteration, int max_iteration){
  return (C2_MAX - C2_MIN)*(iteration/max_iteration) + C2_MIN;
}


template<typename value_type, typename function_type>
inline value_type sums_f(std::vector<value_type> ti, std::vector<value_type> freqi,
                           const value_type ppi, function_type func)
{
  unsigned N = DIMENSION;
  value_type ret(0.0);

  for(int k = 0; k < N; k++){
    ret += func(ti[k]*freqi[k]*ppi); 
  }
  return ret;
}

inline boost::multiprecision::mpf_float_500 sums_fssss(std::vector<boost::multiprecision::mpf_float_500> ti, 
    std::vector<boost::multiprecision::mpf_float_500> freqi,
                           const boost::multiprecision::mpf_float_500 ppi)
{
  using namespace boost::multiprecision;
  unsigned N = DIMENSION;
  mpf_float_500 ret(0.0);

  for(int k = 0; k < N; k++){
    ret += sin(ti[k]*freqi[k]*ppi); 
  }
  return ret;
}

inline double objective_fsum(std::vector<double> x, std::vector<double> y){
  
  using boost::math::constants::pi;
  using namespace boost::multiprecision;
  

  std::vector<mpf_float_500> vf_mp, vt_mp;
  std::vector<double> vfd, vtd;
  std::vector<float> vff, vtf;
  // generate a new solution
  // Get random numbers for frequencies and time series
  for(int i = 0; i < DIMENSION; i++){
    vfd.push_back(x[i]);
    vtd.push_back(y[i]);
    vff.push_back(x[i]);
    vtf.push_back(y[i]);
    mpf_float_500 xx(x[i]);
    mpf_float_500 yy(y[i]);
    vf_mp.push_back(xx);
    vt_mp.push_back(yy);
  } 
 
  const float sin_f = 
   sums_f(vtf, vff, pi<float>(), 
     static_cast<float(*)(float)>(std::sin)
  );
  
  const double sin_d = 
   sums_f(vtd, vfd, pi<double>(), 
     static_cast<double(*)(double)>(std::sin)
    );
  
  const mpf_float_500 sin_mp = sums_fssss(vf_mp, vt_mp, pi<mpf_float_500>()); 

  const mpf_float_500 sin_double(sin_d);
  const mpf_float_500 diff = sin_mp - sin_double;
  double diff_d = static_cast<double>(diff);
  const double bits_lost_double = std::log2(diff_d);
  return std::abs(diff_d);
}


// returns the global best particle
int init_particles (base_generator_type generator, 
    std::vector<particle_t>& particles, int dimension){
  boost::uniform_real<> freq_dist(0.0, FREQ_MAX);
  boost::uniform_real<> time_dist(0, TIME_MAX); 

  boost::variate_generator<base_generator_type&, boost::uniform_real<> > uni_f(generator, freq_dist);
  boost::variate_generator<base_generator_type&, boost::uniform_real<> > uni_t(generator, time_dist);

  unsigned global_best = 0; 
  for(int i = 0; i < NUM_PARTICLES; i++){
    particles[i].x.resize(dimension);
    particles[i].y.resize(dimension);
    particles[i].velX.resize(dimension);
    particles[i].velY.resize(dimension);
    for(int j = 0; j < dimension; j++){
        particles[i].x[j] = uni_f();
        particles[i].y[j] = uni_t();
        particles[i].velY[j] = vel_max_y; 
        particles[i].velX[j] = vel_max_x; 
    }
    particles[i].personalBest = objective_fsum(particles[i].x, particles[i].y);
    global_best = particles[i].personalBest > global_best ? i : global_best;
  }
  return global_best; 
}


inline double get_scaling_factor(scaling_factor_info_t scaling_factor){
   if(scaling_factor.successes > scaling_factor.e_s){
    return 2*scaling_factor.factor;
   }else if(scaling_factor.failures > scaling_factor.e_f){
    return 0.5*scaling_factor.factor;
   }else{
    return scaling_factor.factor;
   }
   return 0.0;
}

void init_scaling_factor(scaling_factor_info_t &scaling_factor){
  scaling_factor.successes = 0;
  scaling_factor.failures = 0;
  scaling_factor.e_s = 0;
  scaling_factor.e_f = 0;
  scaling_factor.factor = 1;
}



int main(int, char**){
  // seed 
  base_generator_type generator(41);  
  
  using boost::math::constants::pi;
  using namespace boost::multiprecision;

  unsigned Iterations = 1000;
  
  boost::uniform_real<> U0_1(0.0, 1);

  boost::variate_generator<base_generator_type&, boost::uniform_real<> > uni_gen(generator, U0_1);
   
  std::vector<double> bestScoreHistory;
  std::vector<particle_t> particles(NUM_PARTICLES);
  std::vector<std::vector<std::pair<std::vector<double>,std::vector<double>>>> particlesPositionHistory;
  unsigned global_bestIndex = init_particles(generator, particles, DIMENSION);
  double global_bestScore = 0.0;
  scaling_factor_info_t scaling_factor;
  init_scaling_factor(scaling_factor);
  double vel_max_y = uni_gen() * (0, FREQ_MAX);
  double vel_max_x = uni_gen() * (0, TIME_MAX);
  std::vector<double> scoresHistory; 
	bool plotPositions = false;
  for(int iter = 0; iter < Iterations; iter++){
    double r1 = uni_gen();
    double r2 = uni_gen();
    double c1 = cognitive_comp(iter, Iterations);
    double c2 = social_comp(iter, Iterations);
    double w = r1 * c1 + r2 * c2;
    double vu = uni_gen();
    bool failure_occured = true;
    std::vector<std::pair<std::vector<double>, std::vector<double>>> iPositions;
	 	for(int i = 0; i < NUM_PARTICLES;	i++){
      for (int j = 0; j < DIMENSION; j++){
        double rj = uni_gen();
        if(i != global_bestIndex){
          particles[i].velX[j] = particles[i].velX[j] + 
            c1*r1*(particles[i].personalBest - particles[i].x[j]) +
            c2*r2*(particles[global_bestIndex].x[j] - particles[i].x[j]);
          particles[i].velY[j] = particles[i].velY[j] + 
            c1*r1*(particles[i].personalBest - particles[i].y[j]) +
            c2*r2*(particles[global_bestIndex].y[j] - particles[i].y[j]);
          if(particles[i].velX[j] > vel_max_x){
            particles[i].velX[j] = vel_max_x;
          } 
          if(particles[i].velY[j] > vel_max_y){
            particles[i].velY[j] = vel_max_y;
          } 
        }else{
          particles[i].velX[j] = -particles[i].x[j] + 
            particles[global_bestIndex].x[j] + w + 
            get_scaling_factor(scaling_factor)*(1-2*rj);
          particles[i].velY[j] = -particles[i].y[j] + 
            particles[global_bestIndex].y[j] + w + 
            get_scaling_factor(scaling_factor)*(1-2*rj);
        }
        
        particles[i].x[j] += particles[i].velX[j];
        if(particles[i].x[j] > FREQ_MAX ){
          particles[i].x[j] = FREQ_MAX;
        }else if(particles[i].x[j] < 0){
          particles[i].x[j] = 0;
        }
        particles[i].y[j] += particles[i].velY[j];
        if(particles[i].y[j] > TIME_MAX ){
          particles[i].y[j] = TIME_MAX;
        }else if(particles[i].y[j] < 0){
          particles[i].y[j] = 0;
        }
      }// dimensions loop
      iPositions.push_back(getPosition(particles[i]));
      double score = objective_fsum(particles[i].x, particles[i].y);
      scoresHistory.push_back(score);
      if(score > particles[i].personalBest){
        particles[i].personalBest = score; 
      }
      if(score > global_bestScore){
        global_bestIndex = i;
        global_bestScore = score;
        failure_occured = false;
        bestScoreHistory.push_back(global_bestScore);
      }
	 }// particles loop
   if(failure_occured){
      scaling_factor.failures++;
      scaling_factor.successes = 0;
      if(scaling_factor.failures > scaling_factor.e_f){
        scaling_factor.e_s++;
      }
   }else{
      scaling_factor.successes++;
      scaling_factor.failures = 0;
      if(scaling_factor.successes > scaling_factor.e_s){
        scaling_factor.e_f++;
      }
    }
    particlesPositionHistory.push_back(iPositions);
    vel_max_x = clamp_velocity(vel_max_x, iter, Iterations);
    vel_max_y = clamp_velocity(vel_max_y, iter, Iterations);
  }
  std::cout << "Final Score: " << global_bestScore << std::endl;
  Gnuplot gp;
  
  if(plotPositions == true){
    gp << "set xrange [0:10]\n";
    gp << "set yrange [0:1000]\n";
    gp << "plot for [i=1:" << Iterations << "] '-' with lines title 'It '.i, ";
    gp << "for [i=1:" << NUM_PARTICLES << "] '' with points pointtype 7 title 'Final Pos'\n";
    for (const auto& iteration : particlesPositionHistory) {
      for (const auto& position : iteration) { 
        gp.send1d(position.first);
        gp.send1d(position.second);
        gp.send1d("e");
      }
    }
  }else{
    gp << "set xlabel 'Iteration'\n";
    gp << "set ylabel 'Fitness Score'\n";
    gp << "plot '-' with lines title 'PSO Optimization'\n";
    gp.send1d(scoresHistory);
  }
  return 0;

}
