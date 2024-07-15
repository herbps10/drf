/*-------------------------------------------------------------------------------
  This file is part of generalized-random-forest.

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

#include <Rcpp.h>
#include <queue>
#include <vector>

#include "analysis/SplitFrequencyComputer.h"
#include "commons/globals.h"
#include "forest/Forest.h"
#include "prediction/ObjectiveBayesDebiaser.h"
#include "prediction/collector/SampleWeightComputer.h"
#include "prediction/collector/CausalSampleWeightComputer.h"
#include "prediction/collector/TreeTraverser.h"

#include "RcppUtilities.h"

using namespace drf;

// [[Rcpp::export]]
Rcpp::NumericMatrix compute_split_frequencies(Rcpp::List forest_object,
                                              size_t max_depth) {
  Forest forest = RcppUtilities::deserialize_forest(forest_object);

  SplitFrequencyComputer computer;
  std::vector<std::vector<size_t>> split_frequencies = computer.compute(forest, max_depth);

  size_t num_variables = forest.get_num_variables();
  Rcpp::NumericMatrix result(max_depth, num_variables);
  for (size_t depth = 0; depth < split_frequencies.size(); depth++) {
    const std::vector<size_t>& frequencies = split_frequencies.at(depth);
    for (size_t var = 0; var < num_variables; var++) {
      double frequency = frequencies[var];
        result(depth, var) = frequency;
      }
    }
  return result;
}

Eigen::SparseMatrix<double> compute_sample_weights(Rcpp::List forest_object,
                                                   Rcpp::NumericMatrix train_matrix,
                                                   Eigen::SparseMatrix<double> sparse_train_matrix,
                                                   Rcpp::NumericMatrix test_matrix,
                                                   Eigen::SparseMatrix<double> sparse_test_matrix,
                                                   unsigned int num_threads,
                                                   bool oob_prediction) {
  std::unique_ptr<Data> train_data = RcppUtilities::convert_data(train_matrix, sparse_train_matrix);
  std::unique_ptr<Data> data = RcppUtilities::convert_data(test_matrix, sparse_test_matrix);
  Forest forest = RcppUtilities::deserialize_forest(forest_object);
  num_threads = ForestOptions::validate_num_threads(num_threads);

  TreeTraverser tree_traverser(num_threads);
  SampleWeightComputer weight_computer;

  std::vector<std::vector<size_t>> leaf_nodes_by_tree = tree_traverser.get_leaf_nodes(forest, *data, oob_prediction);
  std::vector<std::vector<bool>> trees_by_sample = tree_traverser.get_valid_trees_by_sample(forest, *data, oob_prediction);

  size_t num_samples = data->get_num_rows();
  size_t num_neighbors = train_data->get_num_rows();

  // From http://eigen.tuxfamily.org/dox/group__TutorialSparse.html:
  // Filling a sparse matrix effectively
  std::vector<Eigen::Triplet<double>> triplet_list;
  triplet_list.reserve(num_neighbors);
  Eigen::SparseMatrix<double> result(num_samples, num_neighbors);

  for (size_t sample = 0; sample < num_samples; sample++) {
    std::unordered_map<size_t, double> weights = weight_computer.compute_weights(
      sample, forest, leaf_nodes_by_tree, trees_by_sample);

    for (auto it = weights.begin(); it != weights.end(); it++) {
      size_t neighbor = it->first;
      double weight = it->second;
      triplet_list.emplace_back(sample, neighbor, weight);
    }
  }
  result.setFromTriplets(triplet_list.begin(), triplet_list.end());

  return result;
}


Eigen::SparseMatrix<double> compute_causal_sample_weights(Rcpp::List forest_object,
                                                   Rcpp::NumericMatrix train_matrix,
                                                   Eigen::SparseMatrix<double> sparse_train_matrix,
                                                   Rcpp::NumericMatrix test_matrix,
                                                   Eigen::SparseMatrix<double> sparse_test_matrix,
                                                   size_t treatment_index,
                                                   unsigned int num_threads,
                                                   bool oob_prediction) {
  std::unique_ptr<Data> train_data = RcppUtilities::convert_data(train_matrix, sparse_train_matrix);
  std::unique_ptr<Data> data = RcppUtilities::convert_data(test_matrix, sparse_test_matrix);

  train_data->set_treatment_index(treatment_index - 1);
  data->set_treatment_index(treatment_index - 1);

  Forest forest = RcppUtilities::deserialize_forest(forest_object);
  num_threads = ForestOptions::validate_num_threads(num_threads);

  TreeTraverser tree_traverser(num_threads);
  CausalSampleWeightComputer weight_computer;

  std::vector<std::vector<size_t>> leaf_nodes_by_tree = tree_traverser.get_leaf_nodes(forest, *data, oob_prediction);
  std::vector<std::vector<bool>> trees_by_sample = tree_traverser.get_valid_trees_by_sample(forest, *data, oob_prediction);

  size_t num_samples = data->get_num_rows();
  size_t num_neighbors = train_data->get_num_rows();

  // From http://eigen.tuxfamily.org/dox/group__TutorialSparse.html:
  // Filling a sparse matrix effectively
  std::vector<Eigen::Triplet<double>> triplet_list;
  triplet_list.reserve(num_neighbors);
  Eigen::SparseMatrix<double> result(num_samples, num_neighbors);

  for (size_t sample = 0; sample < num_samples; sample++) {
    size_t treatment = data->get_treatment(sample);
    std::unordered_map<size_t, double> weights = weight_computer.compute_weights(
      sample, treatment, train_data, forest, 0, forest.get_trees().size(), leaf_nodes_by_tree, trees_by_sample);
    for (auto it = weights.begin(); it != weights.end(); it++) {
      size_t neighbor = it->first;
      double weight = it->second;
      triplet_list.emplace_back(sample, neighbor, weight);
    }
  }
  result.setFromTriplets(triplet_list.begin(), triplet_list.end());

  return result;
}


std::vector<Eigen::SparseMatrix<double>> compute_causal_bootstrap_sample_weights(Rcpp::List forest_object,
                                                          Rcpp::NumericMatrix train_matrix,
                                                          Eigen::SparseMatrix<double> sparse_train_matrix,
                                                          Rcpp::NumericMatrix test_matrix,
                                                          Eigen::SparseMatrix<double> sparse_test_matrix,
                                                          size_t treatment_index,
                                                          size_t ci_group_size,
                                                          unsigned int num_threads,
                                                          bool oob_prediction) {
  std::unique_ptr<Data> train_data = RcppUtilities::convert_data(train_matrix, sparse_train_matrix);
  std::unique_ptr<Data> data = RcppUtilities::convert_data(test_matrix, sparse_test_matrix);

  train_data->set_treatment_index(treatment_index - 1);
  data->set_treatment_index(treatment_index - 1);

  Forest forest = RcppUtilities::deserialize_forest(forest_object);
  num_threads = ForestOptions::validate_num_threads(num_threads);

  TreeTraverser tree_traverser(num_threads);
  CausalSampleWeightComputer weight_computer;

  std::vector<std::vector<size_t>> leaf_nodes_by_tree = tree_traverser.get_leaf_nodes(forest, *data, oob_prediction);
  std::vector<std::vector<bool>> trees_by_sample = tree_traverser.get_valid_trees_by_sample(forest, *data, oob_prediction);

  size_t num_samples = data->get_num_rows();
  size_t num_neighbors = train_data->get_num_rows();

  // From http://eigen.tuxfamily.org/dox/group__TutorialSparse.html:
  // Filling a sparse matrix effectively

  std::vector<Eigen::SparseMatrix<double>> results;

  for(size_t tree_index = 0; tree_index < forest.get_trees().size(); tree_index += ci_group_size) {
    std::vector<Eigen::Triplet<double>> triplet_list;
    triplet_list.reserve(num_neighbors);
    Eigen::SparseMatrix<double> result(num_samples, num_neighbors);
    for (size_t sample = 0; sample < num_samples; sample++) {
      size_t treatment = data->get_treatment(sample);

      //std::unordered_map<size_t, double> weights = weight_computer.compute_weights(
      //  sample, treatment, train_data, forest, tree_index, tree_index + ci_group_size, leaf_nodes_by_tree, trees_by_sample);

      std::unordered_map<size_t, double> weights = weight_computer.compute_weights(
        sample, treatment, train_data, forest, tree_index, tree_index + ci_group_size, leaf_nodes_by_tree, trees_by_sample);

      for (auto it = weights.begin(); it != weights.end(); it++) {
        size_t neighbor = it->first;
        double weight = it->second;
        triplet_list.emplace_back(sample, neighbor, weight);
      }
    }
    result.setFromTriplets(triplet_list.begin(), triplet_list.end());

    results.push_back(result);
  }

  return results;
}

// [[Rcpp::export]]
Eigen::SparseMatrix<double> compute_weights(Rcpp::List forest_object,
                                            Rcpp::NumericMatrix train_matrix,
                                            Eigen::SparseMatrix<double> sparse_train_matrix,
                                            Rcpp::NumericMatrix test_matrix,
                                            Eigen::SparseMatrix<double> sparse_test_matrix,
                                            unsigned int num_threads) {
  return compute_sample_weights(forest_object, train_matrix, sparse_test_matrix,
                                test_matrix, sparse_test_matrix, num_threads, false);
}

// [[Rcpp::export]]
Eigen::SparseMatrix<double> compute_weights_oob(Rcpp::List forest_object,
                                                Rcpp::NumericMatrix test_matrix,
                                                Eigen::SparseMatrix<double> sparse_test_matrix,
                                                unsigned int num_threads) {
  return compute_sample_weights(forest_object, test_matrix, sparse_test_matrix,
                                test_matrix, sparse_test_matrix, num_threads, true);
}

// [[Rcpp::export]]
Eigen::SparseMatrix<double> compute_causal_weights(Rcpp::List forest_object,
                                            Rcpp::NumericMatrix train_matrix,
                                            Eigen::SparseMatrix<double> sparse_train_matrix,
                                            Rcpp::NumericMatrix test_matrix,
                                            Eigen::SparseMatrix<double> sparse_test_matrix,
                                            size_t treatment_index,
                                            unsigned int num_threads) {
  return compute_causal_sample_weights(forest_object, train_matrix, sparse_test_matrix,
                                test_matrix, sparse_test_matrix, treatment_index, num_threads, false);
}

// [[Rcpp::export]]
Eigen::SparseMatrix<double> compute_causal_weights_oob(Rcpp::List forest_object,
                                                Rcpp::NumericMatrix test_matrix,
                                                Eigen::SparseMatrix<double> sparse_test_matrix,
                                                size_t treatment_index,
                                                unsigned int num_threads) {
  return compute_causal_sample_weights(forest_object, test_matrix, sparse_test_matrix,
                                test_matrix, sparse_test_matrix, treatment_index, num_threads, true);
}

// [[Rcpp::export]]
std::vector<Eigen::SparseMatrix<double>> compute_causal_bootstrap_weights(Rcpp::List forest_object,
                                                   Rcpp::NumericMatrix train_matrix,
                                                   Eigen::SparseMatrix<double> sparse_train_matrix,
                                                   Rcpp::NumericMatrix test_matrix,
                                                   Eigen::SparseMatrix<double> sparse_test_matrix,
                                                   size_t treatment_index,
                                                   size_t ci_group_size,
                                                   unsigned int num_threads) {
  return compute_causal_bootstrap_sample_weights(forest_object, train_matrix, sparse_test_matrix,
                                       test_matrix, sparse_test_matrix, treatment_index, ci_group_size, num_threads, false);
}

// [[Rcpp::export]]
std::vector<Eigen::SparseMatrix<double>> compute_causal_bootstrap_weights_oob(Rcpp::List forest_object,
                                                       Rcpp::NumericMatrix test_matrix,
                                                       Eigen::SparseMatrix<double> sparse_test_matrix,
                                                       size_t treatment_index,
                                                       size_t ci_group_size,
                                                       unsigned int num_threads) {
  return compute_causal_bootstrap_sample_weights(forest_object, test_matrix, sparse_test_matrix,
                                       test_matrix, sparse_test_matrix, treatment_index, ci_group_size, num_threads, true);
}


// [[Rcpp::export]]
Rcpp::List merge(const Rcpp::List forest_objects) {
 std::vector<Forest> forests;

 for (auto& forest_obj : forest_objects) {
   Forest deserialized_forest = RcppUtilities::deserialize_forest(forest_obj);
   forests.push_back(std::move(deserialized_forest));
 }

  Forest big_forest = Forest::merge(forests);
  return RcppUtilities::serialize_forest(big_forest);
}


// [[Rcpp::export]]
Eigen::MatrixXd compute_witness(
    Rcpp::List forest_object,
    Rcpp::NumericMatrix train_matrix,
    Eigen::SparseMatrix<double> sparse_train_matrix,
    Rcpp::NumericMatrix test_matrix,
    Eigen::SparseMatrix<double> sparse_test_matrix,
    Eigen::MatrixXd Ky,
    size_t ci_group_size,
    size_t treatment_index,
    double alpha,
    unsigned int num_threads
) {
  std::unique_ptr<Data> data = RcppUtilities::convert_data(test_matrix, sparse_test_matrix);
  std::unique_ptr<Data> train_data = RcppUtilities::convert_data(train_matrix, sparse_train_matrix);
  Forest forest = RcppUtilities::deserialize_forest(forest_object);
  num_threads = ForestOptions::validate_num_threads(num_threads);

  data->set_treatment_index(treatment_index - 1);
  train_data->set_treatment_index(treatment_index - 1);

  size_t num_samples = data->get_num_rows();
  size_t num_neighbors = train_data->get_num_rows();
  size_t num_trees = forest.get_trees().size();
  size_t num_groups = (int) (num_trees / ci_group_size);

  TreeTraverser tree_traverser(num_threads);
  CausalSampleWeightComputer weight_computer;

  std::vector<std::vector<size_t>> leaf_nodes_by_tree = tree_traverser.get_leaf_nodes(forest, *data, false);
  std::vector<std::vector<bool>> trees_by_sample = tree_traverser.get_valid_trees_by_sample(forest, *data, false);

  Eigen::MatrixXd all_witness_function = Eigen::MatrixXd::Zero(3, num_neighbors);

  std::unordered_map<size_t, double> weights0;
  std::unordered_map<size_t, double> weights1;

  std::vector<Eigen::Triplet<double>> triplet_list;
  triplet_list.reserve(num_neighbors);
  Eigen::SparseMatrix<double> weights(num_samples, num_neighbors);

  for (size_t sample = 0; sample < num_samples; sample++) {
    weights0 = weight_computer.compute_weights(sample, 0, train_data, forest, 0, num_trees, leaf_nodes_by_tree, trees_by_sample);
    weights1 = weight_computer.compute_weights(sample, 1, train_data, forest, 0, num_trees, leaf_nodes_by_tree, trees_by_sample);

    for (auto it = weights0.begin(); it != weights0.end(); it++) {
      size_t neighbor = it->first;
      double weight = it->second;
      triplet_list.emplace_back(sample, neighbor, -weight);
    }

    for (auto it = weights1.begin(); it != weights1.end(); it++) {
      size_t neighbor = it->first;
      double weight = it->second;
      triplet_list.emplace_back(sample, neighbor, weight);
    }
  }
  weights.setFromTriplets(triplet_list.begin(), triplet_list.end());
  all_witness_function.row(0) = (weights * Ky).colwise().mean();

  //Eigen::MatrixXd term3 = weights * Ky * weights.transpose();

  size_t group;
  Eigen::MatrixXd group_witness_functions = Eigen::MatrixXd::Zero(num_groups, num_samples * num_samples);
  for(size_t group_index = 0; group_index < forest.get_trees().size(); group_index += ci_group_size) {
    group = group_index / ci_group_size;
    triplet_list.clear();
    triplet_list.reserve(num_neighbors);
    Eigen::SparseMatrix<double> group_weights(num_samples, num_neighbors);
    for (size_t sample = 0; sample < num_samples; sample++) {

      weights0 = weight_computer.compute_weights(sample, 0, train_data, forest, group_index, group_index + ci_group_size, leaf_nodes_by_tree, trees_by_sample);
      weights1 = weight_computer.compute_weights(sample, 1, train_data, forest, group_index, group_index + ci_group_size, leaf_nodes_by_tree, trees_by_sample);

      for (auto it = weights0.begin(); it != weights0.end(); it++) {
        size_t neighbor = it->first;
        double weight = it->second;
        triplet_list.emplace_back(sample, neighbor, -weight);
      }

      for (auto it = weights1.begin(); it != weights1.end(); it++) {
        size_t neighbor = it->first;
        double weight = it->second;
        triplet_list.emplace_back(sample, neighbor, weight);
      }
    }
    group_weights.setFromTriplets(triplet_list.begin(), triplet_list.end());

    Eigen::MatrixXd res = (group_weights - weights) * Ky * (group_weights - weights).transpose();
    Eigen::Map<Eigen::VectorXd> res_vec(res.data(), res.cols() * res.rows());
    group_witness_functions.row(group) = res_vec;
  }

  Eigen::VectorXd norms = Eigen::Map<Eigen::VectorXd>(group_witness_functions.data(), group_witness_functions.cols() * group_witness_functions.rows());

  std::sort(norms.data(), norms.data() + norms.size());
  double quantile = norms(floor((1 - alpha) * norms.size()));

  all_witness_function.row(1) = all_witness_function.row(0);
  all_witness_function.row(2) = all_witness_function.row(0);
  all_witness_function.row(1).array() -= sqrt(quantile);
  all_witness_function.row(2).array() += sqrt(quantile);

  return all_witness_function;
}
