#ifndef SOLVER_EXCEPTION_HPP
#define SOLVER_EXCEPTION_HPP

#include <deal.II/base/exceptions.h>
#include <string>

class SolverException : public dealii::ExceptionBase {
public:
    SolverException(const std::string s) : message_(s) {}

    virtual const char* what() const noexcept override {
        return message_.c_str();
    }

private:
    std::string message_;
};

#endif // SOLVER_EXCEPTION_HPP
