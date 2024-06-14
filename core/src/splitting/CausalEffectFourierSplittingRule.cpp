/*-------------------------------------------------------------------------------
 This file is part of distributional random forest (drf).

 drf is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 drf is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with drf. If not, see <http://www.gnu.org/licenses/>.
#-------------------------------------------------------------------------------*/


 // THERE NEED CHANGES
#include <algorithm>
#include <random> //n
#include <vector> //n
#include <complex> //n

#include "CausalEffectFourierSplittingRule.h"

 namespace drf {

 CausalEffectFourierSplittingRule::CausalEffectFourierSplittingRule(size_t max_num_unique_values,
                                                        double alpha,
                                                        double imbalance_penalty,
                                                        size_t dim_outcome,
                                                        size_t num_features,
                                                        double bandwidth,
                                                        unsigned int node_scaling):
 alpha(alpha),
 imbalance_penalty(imbalance_penalty),
 num_features(num_features),
 bandwidth(bandwidth),
 node_scaling(node_scaling){
   //this->counter = new size_t[max_num_unique_values];
   //this->sums = new double[max_num_unique_values];
   //s this->sums = new double[max_num_unique_values * dim_outcome];
   // pointers of pointers
   // this->sums = new double*[max_num_unique_values];
   // for (int i = 0; i < (max_num_unique_values-1); ++i) {
   //   this->sums[i] = new double[dim_outcome];
   // }

   //new std::vector<std::vector<double>>(max_num_unique_values, std::vector<double>(dim_outcome,0.0));
   //this->sums = new std::vector<std::vector<double>>(max_num_unique_values);
   //for (size_t d=0; d<=((this->sums).size()-1); ++d) {
   //  this->sums[d]->resize(dim_outcome, 0.0);
   //}
   //Free each sub-array


 }

 CausalEffectFourierSplittingRule::~CausalEffectFourierSplittingRule() {
   // if (counter != nullptr) {
   //   delete[] counter;
   // }
   // if (sums != nullptr) {
   //  // delete[] sums;
   //  // int s = sizeof(sums)/sizeof(sums[0]); // maybe need some check here
   //   //for(int i = 0; i < 10; ++i) {
   //   //  delete[] sums[i];
   //   //}
   //   //Free the array of pointers
   //   delete[] sums;
   //
   // }
 }

 bool CausalEffectFourierSplittingRule::find_best_split(const Data& data,
                                                  size_t node,
                                                  const std::vector<size_t>& possible_split_vars,
                                                  std::vector<std::vector<double> >& responses_by_sample, // std::vector<double> -> std::vector<std::vector<double>>
                                                  const std::vector<std::vector<size_t> >& samples,
                                                  std::vector<size_t>& split_vars,
                                                  std::vector<double>& split_values) {

   size_t size_node = samples[node].size();
   size_t min_child_size = std::max<size_t>(std::ceil(size_node * alpha), 1uL);

   //n initialize the fourier features
   std::vector<std::vector<std::complex<double> > > fourier_features;
   fourier_features.resize(num_features);
   std::default_random_engine generator;
   generator.seed(153 + node);
   std::normal_distribution<double> distribution(0.0, 1.0/(bandwidth*bandwidth));

   size_t d = data.get_outcome_index().size(); //int d = data[0].size(); //dimensionality of the response
   size_t n = samples[node].size(); //int n = sampleIDs.size(); //number of datapoints in the current node
   std::complex<double> i(0, 1); //imaginary unit

   std::vector<double> omega(d, 0);

   // center and scale
   if (node_scaling) {

     for(size_t k = 0; k < d; ++k) {
       double m = 0.0;
       double s = 0.0;
       // center
       for(size_t j = 0; j < n; ++j) {
         m += responses_by_sample[samples[node][j]][k];
       }
       m = m / n;
       for(size_t j = 0; j < n; ++j) {
         responses_by_sample[samples[node][j]][k] = responses_by_sample[samples[node][j]][k] - m;
       }
       // scale
       for(size_t j = 0; j < n; ++j) {
         s += responses_by_sample[samples[node][j]][k] * responses_by_sample[samples[node][j]][k];
       }
       s = sqrt(s/n);
       if (s > 10e-10) {
         for(size_t j = 0; j < n; ++j) {
           responses_by_sample[samples[node][j]][k] = responses_by_sample[samples[node][j]][k]/s;
         }
       }
     }
   }

   for(size_t l = 0; l < num_features; ++l){
     fourier_features[l].resize(n);

     for(size_t k = 0; k < d; ++k)
       omega[k] = distribution(generator);

     for(size_t j = 0; j < n; ++j){
       double theta = 0;
       for(size_t k = 0; k < d; ++k) {
         if(data.get_treatment(samples[node][j]) > 0) {
           theta += omega[k] * (responses_by_sample[samples[node][j]][k]);
         }
         else {
           theta += omega[k] * responses_by_sample[samples[node][j]][k];
         }
       }

       fourier_features[l][j] = exp(i * theta);
     }
   }

   // Initialize the variables to track the best split variable.
   size_t best_var = 0;
   double best_value = 0;
   double best_decrease = -1.0;

   // For all possible split variables
   for (auto& var : possible_split_vars) {
     std::vector<size_t> idx_in_sorted_array;

     for (auto& sample : samples[node]) {
       size_t index = data.get_index(sample, var);
       idx_in_sorted_array.push_back(index);
     }
     idx_in_sorted_array.resize(size_node);

     find_split(data, node, best_value, best_var, best_decrease, min_child_size,
                var, fourier_features, idx_in_sorted_array, samples);
   }

   // Stop if no good split found
   if (best_decrease <= 0.0) {
     return true;
   }

   // Save best values
   split_vars[node] = best_var;
   split_values[node] = best_value;

   return false;
 }


 void CausalEffectFourierSplittingRule::find_split(const Data& data,
                                             size_t node,
                                             double& best_value,
                                             size_t& best_var,
                                             double& best_decrease,
                                             size_t min_child_size,
                                             size_t var,
                                             const std::vector<std::vector<std::complex<double> > >& fourier_features,
                                             const std::vector<size_t>& idx_in_sorted_array,
                                             const std::vector<std::vector<size_t>>& samples
 ){
   int n = idx_in_sorted_array.size();

   if (n <= min_child_size) {
     return;
   }

   std::vector<std::pair<size_t, size_t>> tmp;
   tmp.resize(n);

   for(size_t i = 0; i < n; ++i) {
     tmp[i] = std::make_pair(idx_in_sorted_array[i], i);
   }
   sort(tmp.begin(), tmp.end()); //sorts by default lexicographically

   std::vector<size_t> ordering(n, 0);
   for(size_t i = 0; i < n; ++i) {
     ordering[i] = tmp[i].second;
   }
   //how are the datapoints in the current node sorted according to X_i, ordering[i] is the index of the datapoint in the sampleIDs that is at place i
   //std::cout << "survived ordering building" << std::endl;
   size_t num_features = fourier_features.size();
   //std::vector<double> MMD(0, n-1);
   std::vector<double> MMD(n-1, 0.0);

   size_t num_treatment = 0, num_control = 0;
   for(int j = 0; j < n; ++j) {
     if(data.get_treatment(samples[node][j]) > 0) {
       num_treatment++;
     }
     else {
       num_control++;
     }
   }

   std::complex<double> sum_treatment(0, 0);
   std::complex<double> sum_control(0, 0);
   for(size_t i = 0; i < num_features; ++i){
     for(int j = 0; j < n; ++j) {
       if(data.get_treatment(samples[node][j]) == 1) {
         sum_treatment += fourier_features[i][j];
       }
       else {
         sum_control += fourier_features[i][j];
       }
     }
   }

   std::complex<double> sum_left_treatment(0, 0), sum_left_control(0, 0),
                        sum_right_treatment(0, 0), sum_right_control(0, 0);

   size_t num_control_left = 0, num_treatment_left = 0,
          num_control_right = num_control, num_treatment_right = num_treatment;

   for(size_t j = 0;  j < (n-1); ++j){
     if(data.get_treatment(samples[node][ordering[j]]) > 0) {
       for(int i = 0; i < num_features; i++)
         sum_left_treatment += fourier_features[i][ordering[j]];
       sum_right_treatment = sum_treatment - sum_left_treatment;
       num_treatment_left += 1;
       num_treatment_right -= 1;
     }
     else {
       for(int i = 0; i < num_features; i++)
         sum_left_control += fourier_features[i][ordering[j]];
       sum_right_control = sum_control - sum_left_control;
       num_control_left += 1;
       num_control_right -= 1;
     }

     if(num_treatment_left == 0 || num_control_left == 0 || num_treatment_right == 0 || num_control_right == 0) {
       continue;
     }

     MMD[j] += ((double)(j+1)*(n-j-1)/(double)n) * pow(
       pow(abs(sum_left_treatment), 2)/(double)num_treatment_left - pow(abs(sum_left_control), 2)/(double)num_control_left
       - (pow(abs(sum_right_treatment), 2)/(double)num_treatment_right - pow(abs(sum_right_control), 2)/(double)num_control_right)
     , 2) / (double)num_features;
   }

   double best_dec = -1;
   double best_val = 0;
   double best_dec10 = -1;
   double best_val10 = 0;

   for(size_t i = 0; i < (n-1); ++i){
     if(tmp[i].first == tmp[i+1].first)
       continue;

     if(best_dec < MMD[i]){
       best_dec = MMD[i];
       best_val = data.get(samples[node][tmp[i].second], var);
     }
     if((i+1) < 0.1*n && (n-i-1) < 0.1*n)  {// we ensure at least 10% of datapoints is in each child to ensure logarithmic depth of the tree
       continue;
     }
     if(best_dec10 < MMD[i]){
       best_dec10 = MMD[i];
       best_val10 = data.get(samples[node][tmp[i].second], var);
     }
   }

   //std::cout << var << ": " << best_dec << " " << best_dec10 << std::endl;

   if(best_dec10 > -0.5){
     if(best_decrease < best_dec10){
       best_decrease = best_dec10;
       best_value = best_val10;
       best_var = var;
     }
   }
   else{
     if(best_decrease < best_dec) {
       best_decrease = best_dec;
       best_value = best_val;
       best_var = var;
     }
   }
   //return make_pair(sampleID, best_stat);
 }


 } // namespace drf
