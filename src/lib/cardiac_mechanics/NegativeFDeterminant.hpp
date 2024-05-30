#ifndef NEGATIVE_FDETERMINANT_HPP
#define NEGATIVE_FDETERMINANT_HPP

#include "SolverException.hpp"

class NegativeFDeterminant : public SolverException {
public:
    NegativeFDeterminant() : SolverException("Negative F tensor determinant, this might have occured due to a too far jump in Newton iteration solution causing global instability.\nTry lowering the values of <InitialReductionFactor> and <ReductionFactorIncrement> in the configuration file") {}
private:
};

#endif // NEGATIVE_FDETERMINANT_HPP
