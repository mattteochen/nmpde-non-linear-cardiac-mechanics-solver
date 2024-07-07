/**
 * @file NegativeFDeterminant.hpp
 * @brief Header file defining the negative deformation gradient tensor F
 * determinant exception
 */

#ifndef NEGATIVE_FDETERMINANT_HPP
#define NEGATIVE_FDETERMINANT_HPP

#include "SolverException.hpp"

/**
 * @class NegativeFDeterminant
 * @brief Utility class for the negative deformation gradient tensor F
 * determinant exception
 */
class NegativeFDeterminant : public SolverException {
public:
  /**
   * @brief Constructor.
   */
  NegativeFDeterminant()
      : SolverException(
            "Negative F tensor determinant, this might have occured due to a "
            "too far jump in Newton iteration solution causing global "
            "instability.\nTry lowering the values of <InitialReductionFactor> "
            "and <ReductionFactorIncrement> in the configuration file if the "
            "solver supports an increasing pressure technique.") {}
};

#endif // NEGATIVE_FDETERMINANT_HPP
