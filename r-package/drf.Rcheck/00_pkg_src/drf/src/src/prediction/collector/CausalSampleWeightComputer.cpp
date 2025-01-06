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

#include "CausalSampleWeightComputer.h"

#include "tree/Tree.h"

namespace drf {

std::unordered_map<size_t, double> CausalSampleWeightComputer::compute_weights(size_t sample,
                                                                         size_t treatment,
                                                                         const std::unique_ptr<Data>& data,
                                                                         const Forest& forest,
                                                                         size_t tree_start_index,
                                                                         size_t tree_end_index,
                                                                         const std::vector<std::vector<size_t>>& leaf_nodes_by_tree,
                                                                         const std::vector<std::vector<bool>>& valid_trees_by_sample) const {
  std::unordered_map<size_t, double> weights_by_sample;

  // Create a list of weighted neighbors for this sample.
  for (size_t tree_index = tree_start_index; tree_index < tree_end_index; ++tree_index) {
    if (!valid_trees_by_sample[sample][tree_index]) {
      continue;
    }

    const std::vector<size_t>& leaf_nodes = leaf_nodes_by_tree.at(tree_index);
    size_t node = leaf_nodes.at(sample);

    const std::unique_ptr<Tree>& tree = forest.get_trees()[tree_index];
    const std::vector<size_t>& samples = tree->get_leaf_samples()[node];

    if (!samples.empty()) {
      size_t num_matching_treatment = 0;
      for(auto& sample : samples) {
        num_matching_treatment += (data->get_treatment(sample) == treatment ? 1 : 0);
      }

      double sample_weight = 1.0 / num_matching_treatment;
      for(auto& sample : samples) {
        if(data->get_treatment(sample) == treatment) {
          weights_by_sample[sample] += 1;
        }
      }
    }
  }

  normalize_sample_weights(weights_by_sample);
  return weights_by_sample;
}

void CausalSampleWeightComputer::add_sample_weights(std::vector<size_t> samples,
                                              std::unordered_map<size_t, double>& weights_by_sample) const {
  double sample_weight = 1.0 / samples.size();

  for (auto& sample : samples) {
    weights_by_sample[sample] += sample_weight;
  }
}

void CausalSampleWeightComputer::normalize_sample_weights(std::unordered_map<size_t, double>& weights_by_sample) const {
  double total_weight = 0.0;
  for (const auto& entry : weights_by_sample) {
    total_weight += entry.second;
  }

  for (auto& entry : weights_by_sample) {
    entry.second /= total_weight;
  }
}

} // namespace drf
