#ifndef NAN_HPP
#define NAN_HPP

#include "SolverException.hpp"

class Nan : public SolverException {
public:
    Nan() : SolverException("Nan computed, this might have occured due to a too far jump in Newton iteration solution causing global instability.\nTry lowering the values of <InitialReductionFactor> and <ReductionFactorIncrement> in the configuration file") {}
private:
};

#endif // NAN_HPP
