/**
 * @file NewtonSolverUtility.hpp
 * @brief Header file defining the NewtonSolverUtility class for configuring
 * Newton solvers.
 */

#ifndef NEWTON_SOLVER_UTILITY_HPP
#define NEWTON_SOLVER_UTILITY_HPP

#include <iostream>

/**
 * @class NewtonSolverUtility
 * @brief Utility class for configuring Newton solvers.
 * @tparam Scalar The scalar type used in the solver configuration.
 */
template <typename Scalar> class NewtonSolverUtility {
public:
  /**
   * @brief Default constructor.
   */
  NewtonSolverUtility<Scalar>(){};
  /**
   * @brief Constructor initializing the solver configuration with tolerance and
   * maximum iterations.
   * @param tolerance_ The tolerance for convergence.
   * @param max_iterations_ The maximum number of iterations.
   */
  NewtonSolverUtility<Scalar>(Scalar tolerance_, unsigned int max_iterations_)
      : tolerance(tolerance_), max_iterations(max_iterations_) {}
  /**
   * @brief Move assignment operator.
   * @param other The NewtonSolverUtility object to move from.
   */
  void operator=(NewtonSolverUtility<Scalar> &&other) {
    tolerance = other.get_tolerance();
    max_iterations = other.get_max_iterations();
  }
  /**
   * @brief Gets the tolerance for convergence.
   * @return The tolerance.
   */
  Scalar get_tolerance() const { return tolerance; }
  /**
   * @brief Gets the maximum number of iterations.
   * @return The maximum number of iterations.
   */
  unsigned int get_max_iterations() const { return max_iterations; }

protected:
  Scalar tolerance;            /**< The tolerance for convergence. */
  unsigned int max_iterations; /**< The maximum number of iterations. */
};
/**
 * @brief Stream insertion operator for NewtonSolverUtility objects.
 * @tparam Scalar The scalar type used in the NewtonSolverUtility object.
 * @param out The output stream.
 * @param nsu The NewtonSolverUtility object to output.
 */
template <typename Scalar>
void operator<<(std::ostream &out, NewtonSolverUtility<Scalar> const &nsu) {
  out << "  Tolerance: " << nsu.get_tolerance() << std::endl;
  out << "  Max iterations: " << nsu.get_max_iterations() << std::endl;
}

#endif // NEWTON_SOLVER_UTILITY_HPP
