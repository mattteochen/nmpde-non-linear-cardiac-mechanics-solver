/**
 * @file Nan.hpp
 * @brief Header file defining the not a number runtime exception.
 */

#ifndef NAN_HPP
#define NAN_HPP

#include "SolverException.hpp"

/**
 * @class Nan
 * @brief Utility class for the not a number runtime exception.
 */
class Nan : public SolverException {
public:
  /**
   * @brief Constructor.
   */
  Nan()
      : SolverException(
            "Nan computed, this might have occured due to a too far jump in "
            "Newton iteration solution causing global instability.\nTry "
            "lowering the values of <InitialReductionFactor> and "
            "<ReductionFactorIncrement> in the configuration file") {}
};

#endif // NAN_HPP
