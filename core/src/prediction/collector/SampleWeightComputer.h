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

#ifndef drf_SAMPLEWEIGHTCOMPUTER_H
#define drf_SAMPLEWEIGHTCOMPUTER_H

#include "forest/Forest.h"

#include <unordered_map>
#include <vector>

namespace drf {

class SampleWeightComputer {
public:
  std::unordered_map<size_t, double> compute_weights(size_t sample,
                                                     const Forest& forest,
                                                     const std::vector<std::vector<size_t>>& leaf_nodes_by_tree,
                                                     const std::vector<std::vector<bool>>& valid_trees_by_sample) const;

private:
  void add_sample_weights(const std::vector<size_t>& samples,
                          std::unordered_map<size_t, double>& weights_by_sample) const;

  void normalize_sample_weights(std::unordered_map<size_t, double>& weights_by_sample) const;
};

} // namespace drf

#endif //drf_SAMPLEWEIGHTCOMPUTER_H
