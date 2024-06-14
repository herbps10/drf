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

#ifndef drf_CAUSALSAMPLEWEIGHTCOMPUTER_H
#define drf_CAUSALSAMPLEWEIGHTCOMPUTER_H

#include "commons/Data.h"
#include "forest/Forest.h"

#include <unordered_map>
#include <vector>

namespace drf {

class CausalSampleWeightComputer {
public:
  std::unordered_map<size_t, double> compute_weights(size_t sample,
                                                     size_t treatment,
                                                     const std::unique_ptr<Data>& data,
                                                     const Forest& forest,
                                                     size_t tree_start_index,
                                                     size_t tree_end_index,
                                                     const std::vector<std::vector<size_t>>& leaf_nodes_by_tree,
                                                     const std::vector<std::vector<bool>>& valid_trees_by_sample) const;

private:
  void add_sample_weights(std::vector<size_t> samples,
                          std::unordered_map<size_t, double>& weights_by_sample) const;

  void normalize_sample_weights(std::unordered_map<size_t, double>& weights_by_sample) const;
};

} // namespace drf

#endif //drf_CAUSALSAMPLEWEIGHTCOMPUTER_H
