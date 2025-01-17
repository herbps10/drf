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

#ifndef drf_SPLITTINGRULEFACTORY_H
#define drf_SPLITTINGRULEFACTORY_H

#include <memory>

#include "commons/Data.h"
#include "splitting/SplittingRule.h"
#include "tree/TreeOptions.h"

namespace drf {

class SplittingRuleFactory {
public:

  virtual ~SplittingRuleFactory() = default;

  virtual std::unique_ptr<SplittingRule> create(const Data& data,
                                                const TreeOptions& options) const = 0;
};

} // namespace drf

#endif //drf_SPLITTINGRULEFACTORY_H
