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

#ifndef DRF_FORESTTRAINERS_H
#define DRF_FORESTTRAINERS_H

#include "forest/ForestTrainer.h"

namespace drf {


ForestTrainer gini_trainer(size_t dim);

ForestTrainer fourier_trainer(size_t dim);

ForestTrainer causal_fourier_trainer(size_t dim);

ForestTrainer causal_effect_fourier_trainer(size_t dim);

} // namespace drf

#endif //DRF_FORESTTRAINERS_H

