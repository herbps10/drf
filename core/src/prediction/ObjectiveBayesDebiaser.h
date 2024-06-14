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

#ifndef DRF_OBJECTBAYESDEBIASER_H
#define DRF_OBJECTBAYESDEBIASER_H

 namespace drf {

 class ObjectiveBayesDebiaser {
 public:
   double debias(double var_between,
                 double group_noise,
                 double num_good_groups) const;
 private:
   const double ONE_over_SQRT_TWO_PI = 0.3989422804;
   const double ONE_over_SQRT_TWO = 0.70710678118;
 };

 } // namespace drf

#endif //DRF_OBJECTBAYESDEBIASER_H
