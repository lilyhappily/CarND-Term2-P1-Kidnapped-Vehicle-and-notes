/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using namespace std;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double stddev[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */
   num_particles = 100;  // TODO: Set the number of particles

   //This line create normal (Gaussian) distribution for x, y, theta
   normal_distribution<double> dist_x(x, stddev[0]);
   normal_distribution<double> dist_y(y, stddev[1]);
   normal_distribution<double> dist_theta(theta, stddev[2]);

   // Sample from these normal distribution above
   for(int i = 0; i < num_particles; i++)
   {
       // define the particles and initialize them
       Particle P;

       P.x = dist_x(gen);
       P.y = dist_y(gen);
       P.theta = dist_theta(gen);
       P.weight = 1.0;
       P.id = i;

       particles.push_back(P);
   }
   is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

   // define noise gaussian distribution
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

   for(int i = 0; i < num_particles; i++)
   {
       //avoid division by zero
       if (fabs(yaw_rate) < 0.00001)
       {
           particles[i].x += velocity * delta_t * cos(particles[i].theta);
           particles[i].y += velocity * delta_t * sin(particles[i].theta);
       }
       else
       {
           particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
           particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) -cos(particles[i].theta + yaw_rate * delta_t));
           particles[i].theta += yaw_rate * delta_t;
       }

       // Add noise to the predicted state
       particles[i].x += dist_x(gen);
       particles[i].y += dist_y(gen);
       particles[i].theta += dist_theta(gen);
   }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */


   for(unsigned int i = 0; i < observations.size(); i++)
   {
       // initial the distance and id
       int id = -1;
       double min_dist = numeric_limits<double>::max();

       for(unsigned int j = 0; j < predicted.size(); j++)
       {
           double dist_op= dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
           if(dist_op < min_dist)
           {
               min_dist = dist_op;
               id = predicted[j].id;
           }
       }
       // find the closest predicted measurement(landmarkers on the map) with respect to the ith observation
       observations[i].id = id;
   }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

   for(int i = 0; i < num_particles; i++)
   {
       // get the ith particle position x, y and heading orientation
       double x_p = particles[i].x;
       double y_p = particles[i].y;
       double theta = particles[i].theta;

       // define the transformation of the observations
       vector<LandmarkObs> transform_to_map;

       for(unsigned int j = 0; j < observations.size(); j++)
       {
           LandmarkObs t_obs;

           //compute the observations into map's coordinate
           double x_map = x_p + cos(theta) * observations[j].x - sin(theta) * observations[j].y;
           double y_map = y_p + sin(theta) * observations[j].x + cos(theta) * observations[j].y;

           t_obs.x = x_map;
           t_obs.y = y_map;
           t_obs.id = observations[j].id;
           transform_to_map.push_back(t_obs);
       }

       // make the landmarks within the sensor range
       vector<LandmarkObs> valid_landmarkers;
       for(unsigned int n = 0; n < map_landmarks.landmark_list.size(); n++)
       {
           float map_landmark_x = map_landmarks.landmark_list[n].x_f;
           float map_landmark_y = map_landmarks.landmark_list[n].y_f;
           int map_landmark_id = map_landmarks.landmark_list[n].id_i;

           // only consider the landmarkers within the range of sensors in the center of the ith particle
           if(fabs(map_landmark_x - x_p) <= sensor_range && fabs(map_landmark_y - y_p) <= sensor_range)
           {
               valid_landmarkers.push_back(LandmarkObs{map_landmark_id, map_landmark_x, map_landmark_y});
           }
       }

       // data associations
       dataAssociation(valid_landmarkers, transform_to_map);

       //reinit weight
       particles[i].weight = 1.0;

       // compute the weight one particle
       for(unsigned int t = 0; t < transform_to_map.size(); t++)
       {
           //define the variates of Gaussian distribution
           double pos_x, pos_y, mu_x, mu_y;
           pos_x = transform_to_map[t].x;
           pos_y = transform_to_map[t].y;

           // find the mult-variate Gaussian mean according to the closest landmark
           int mu_id = transform_to_map[t].id;

           for(unsigned int k = 0; k < valid_landmarkers.size(); k++)
           {
               if(valid_landmarkers[k].id == mu_id)
               {
                   mu_x = valid_landmarkers[k].x;
                   mu_y =  valid_landmarkers[k].y;
               }
           }

           //compute the weight of one observation
           double sig_x = std_landmark[0];
           double sig_y = std_landmark[1];
           double gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);
           double enponent = pow((pos_x - mu_x), 2) / (2 * pow(sig_x, 2)) + pow((pos_y - mu_y), 2) / (2 * pow(sig_y, 2));
           double w = gauss_norm * exp(-enponent);
           particles[i].weight *= w;
       }
   }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

   // define the new particles
   vector<Particle> new_p;
   vector<double> weights;
   // get all current weights and noramlize
   for(int i = 0; i < num_particles; i++)
   {
       weights.push_back(particles[i].weight);
   }


   // resampling the wheel
   //1. generate the random start index
   uniform_int_distribution<int> uniform_int_dist(0, num_particles - 1);
   auto index = uniform_int_dist(gen);
   double beta = 0.0;

   //2.get the weight maximum
   double max_weight = *max_element(weights.begin(), weights.end());

   //3.get the random from uniform distribution[0, max_weight]
   uniform_real_distribution<double> uniform_real_dist(0, max_weight);

   //3.resample
   for(int j = 0; j < num_particles; j++)
   {
       beta += uniform_real_dist(gen) * 2.0;
       while(beta > weights[index])
       {
           beta -= weights[index];
           index = (index + 1) % num_particles;
       }
       new_p.push_back(particles[index]);
   }

   particles = new_p;
}


void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
