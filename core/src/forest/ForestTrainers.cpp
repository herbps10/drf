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

#include "forest/ForestTrainers.h"
//#include "prediction/InstrumentalPredictionStrategy.h"
#include "prediction/RegressionPredictionStrategy.h"
//#include "relabeling/CustomRelabelingStrategy.h"
//#include "relabeling/InstrumentalRelabelingStrategy.h"
#include "relabeling/NoopRelabelingStrategy.h"
//#include "relabeling/QuantileRelabelingStrategy.h"
//#include "splitting/factory/InstrumentalSplittingRuleFactory.h"
//#include "splitting/factory/ProbabilitySplittingRuleFactory.h"
#include "splitting/factory/RegressionSplittingRuleFactory.h"
#include "splitting/factory/FourierSplittingRuleFactory.h"

namespace drf {

//ForestTrainer instrumental_trainer(double reduced_form_weight,
//                                   bool stabilize_splits) {
//
//  std::unique_ptr<RelabelingStrategy> relabeling_strategy(new InstrumentalRelabelingStrategy(reduced_form_weight));
//  std::unique_ptr<SplittingRuleFactory> splitting_rule_factory = stabilize_splits
//          ? std::unique_ptr<SplittingRuleFactory>(new InstrumentalSplittingRuleFactory())
//          : std::unique_ptr<SplittingRuleFactory>(new RegressionSplittingRuleFactory());
//  std::unique_ptr<OptimizedPredictionStrategy> prediction_strategy(new InstrumentalPredictionStrategy());
//
//  return ForestTrainer(std::move(relabeling_strategy),
//                       std::move(splitting_rule_factory),
//                       std::move(prediction_strategy));
//}
  
//ForestTrainer quantile_trainer(const std::vector<double>& quantiles) {
//    std::unique_ptr<RelabelingStrategy> relabeling_strategy(new QuantileRelabelingStrategy(quantiles));
//  std::unique_ptr<SplittingRuleFactory> splitting_rule_factory(
//      new ProbabilitySplittingRuleFactory(quantiles.size() + 1));
//
//  return ForestTrainer(std::move(relabeling_strategy),
//                       std::move(splitting_rule_factory),
//                       nullptr);
//}

ForestTrainer gini_trainer(size_t dim) {
  std::unique_ptr<RelabelingStrategy> relabeling_strategy(new NoopRelabelingStrategy());
  
  std::unique_ptr<SplittingRuleFactory> splitting_rule_factory(new RegressionSplittingRuleFactory());

  std::unique_ptr<OptimizedPredictionStrategy> prediction_strategy(new RegressionPredictionStrategy(dim));

  return ForestTrainer(std::move(relabeling_strategy),
                       std::move(splitting_rule_factory),
                       std::move(prediction_strategy));
}

ForestTrainer fourier_trainer(size_t dim) {
  std::unique_ptr<RelabelingStrategy> relabeling_strategy(new NoopRelabelingStrategy());
  
  std::unique_ptr<SplittingRuleFactory> splitting_rule_factory(new FourierSplittingRuleFactory());
  
  std::unique_ptr<OptimizedPredictionStrategy> prediction_strategy(new RegressionPredictionStrategy(dim));
  
  return ForestTrainer(std::move(relabeling_strategy),
                       std::move(splitting_rule_factory),
                       std::move(prediction_strategy));
}


//ForestTrainer custom_trainer() {
//  std::unique_ptr<RelabelingStrategy> relabeling_strategy(new CustomRelabelingStrategy());
//  std::unique_ptr<SplittingRuleFactory> splitting_rule_factory(new RegressionSplittingRuleFactory());
//
//  return ForestTrainer(std::move(relabeling_strategy),
//                       std::move(splitting_rule_factory),
//                       nullptr);
//}

} // namespace drf
