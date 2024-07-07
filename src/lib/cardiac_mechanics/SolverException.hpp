/**
 * @file SolverException.hpp
 * @brief Header file defining the base class for runtime exceptions
 */

#ifndef SOLVER_EXCEPTION_HPP
#define SOLVER_EXCEPTION_HPP

#include <deal.II/base/exceptions.h>
#include <string>

/**
 * @class SolverException
 * @brief Utility class for defining the base class for runtime exceptions.
 */
class SolverException : public dealii::ExceptionBase {
public:
  /**
   * @brief Constructor.
   */
  SolverException(const std::string s) : message_(s) {}

  /**
   * @see dealii::ExceptionBase::what.
   */
  virtual const char *what() const noexcept override {
    return message_.c_str();
  }

private:
  /**
   * @brief The exception message
   */
  std::string message_;
};

#endif // SOLVER_EXCEPTION_HPP
