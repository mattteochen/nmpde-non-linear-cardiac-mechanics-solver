/**
 * @file LinearSolverUtility.hpp
 * @brief Header file defining the LinearSolverUtility class for configuring
 * linear solvers.
 */

#ifndef LINEAR_SOLVER_CONFIGURATION_HPP
#define LINEAR_SOLVER_CONFIGURATION_HPP

#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>

#include <map>
#include <memory>
#include <ostream>
#include <string>

/**
 * @class LinearSolverUtility
 * @brief Utility class for configuring linear solvers.
 * @tparam Scalar The scalar type used in the linear solver configuration
 * (default is double).
 */
template <typename Scalar = double> class LinearSolverUtility {
public:
  /**
   * @brief Enumerates different types of solvers.
   */
  enum class SolverType {
    GMRES,   /**< GMRES solver */
    BiCGSTAB /**< BiCGSTAB solver */
  };
  /**
   * @brief Enumerates different types of preconditioners.
   */
  enum class Preconditioner {
    IDENTITY, /**< Identity preconditioner */
    ILU,      /**< ILU preconditioner */
    SOR,      /**< SOR preconditioner */
    SSOR,      /**< SSOR preconditioner */
    AMG      /**< AMG preconditioner */
  };
  /**
   * @brief Default constructor.
   */
  LinearSolverUtility<Scalar>() {}
  /**
   * @brief Constructor initializing the solver configuration.
   * @param solver_type_ The type of solver.
   * @param preconditioner_type_ The type of preconditioner.
   * @param tolerance_ The tolerance for convergence.
   * @param max_iterations_ The maximum number of iterations.
   */
  LinearSolverUtility<Scalar>(SolverType solver_type_,
                              Preconditioner preconditioner_type_,
                              Scalar tolerance_, Scalar max_iterations_)
      : solver_type(solver_type_), preconditioner_type(preconditioner_type_),
        tolerance(tolerance_), max_iterations(max_iterations_) {}
  /**
   * @brief Move assignment operator.
   * @param other_ The LinearSolverUtility object to move from.
   */
  void operator=(LinearSolverUtility<Scalar> &&other_) {
    solver_type = other_.solver_type;
    preconditioner_type = other_.preconditioner_type;
    tolerance = other_.tolerance;
    max_iterations = other_.max_iterations;
  }

  // Static maps for mapping string names to enumeration values and vice versa
  // (only to avoid a cascade of branches)
  static std::map<std::string, SolverType>
      solver_type_matcher; /**< Map for solver type matching */
  static std::map<SolverType, std::string>
      solver_type_matcher_rev; /**< Reverse map for solver type matching */
  static std::map<std::string, Preconditioner>
      preconditioner_type_matcher; /**< Map for preconditioner type matching */
  static std::map<Preconditioner, std::string>
      preconditioner_type_matcher_rev; /**< Reverse map for preconditioner type
                                          matching */
  /**
   * @brief Get an initialized solver control object.
   * @tparam SizeType The type of the size param
   * @tparam NormType The type of the norm param
   * @param size The size of the problem.
   * @param norm The norm of the problem.
   * @return The initialized solver control object.
   */
  template <typename SizeType, typename NormType>
  dealii::SolverControl
  get_initialized_solver_control(const SizeType size,
                                 const NormType norm) const {
    return dealii::SolverControl(static_cast<SizeType>(size * max_iterations),
                                 tolerance * norm);
  }
  /**
   * @brief Initializes the solver object.
   * @tparam Vector The vector used in the solver
   * @param solver The solver object to initialize.
   * @param solver_control The solver control object.
   */
  template <typename Vector>
  void initialize_solver(std::unique_ptr<dealii::SolverBase<Vector>> &solver,
                         dealii::SolverControl &solver_control) const {
    switch (solver_type) {
    case SolverType::GMRES: {
      solver = std::make_unique<dealii::SolverGMRES<Vector>>(solver_control);
      break;
    }
    case SolverType::BiCGSTAB: {
      solver = std::make_unique<dealii::SolverBicgstab<Vector>>(solver_control);
      break;
    }
    default: {
      solver = std::make_unique<dealii::SolverGMRES<Vector>>(solver_control);
    }
    }
  }
  /**
   * @brief Initializes the preconditioner object.
   * @tparam Matrix The matrix type
   * @param preconditioner The preconditioner object to initialize.
   * @param matrix The matrix used by the preconditioner.
   */
  template <typename Matrix>
  void initialize_preconditioner(
      std::unique_ptr<dealii::TrilinosWrappers::PreconditionBase>
          &preconditioner,
      Matrix const &matrix) const {
    switch (preconditioner_type) {
    case Preconditioner::IDENTITY: {
      preconditioner =
          std::make_unique<dealii::TrilinosWrappers::PreconditionIdentity>();
      // We have to perform a dynamic cast as the PreconditionBase class has
      // not the initialize method
      auto derived_ptr =
          dynamic_cast<dealii::TrilinosWrappers::PreconditionIdentity *>(
              preconditioner.get());
      derived_ptr->initialize(matrix);
      break;
    }
    case Preconditioner::SSOR: {
      preconditioner =
          std::make_unique<dealii::TrilinosWrappers::PreconditionSSOR>();
      // We have to perform a dynamic cast as the PreconditionBase class has
      // not the initialize method
      auto derived_ptr =
          dynamic_cast<dealii::TrilinosWrappers::PreconditionSSOR *>(
              preconditioner.get());
      derived_ptr->initialize(matrix);
      break;
    }
    case Preconditioner::SOR: {
      preconditioner =
          std::make_unique<dealii::TrilinosWrappers::PreconditionSOR>();
      // We have to perform a dynamic cast as the PreconditionBase class has
      // not the initialize method
      auto derived_ptr =
          dynamic_cast<dealii::TrilinosWrappers::PreconditionSOR *>(
              preconditioner.get());
      derived_ptr->initialize(matrix);
      break;
    }
    case Preconditioner::ILU: {
      preconditioner =
          std::make_unique<dealii::TrilinosWrappers::PreconditionILU>();
      // We have to perform a dynamic cast as the PreconditionBase class has
      // not the initialize method
      auto derived_ptr =
          dynamic_cast<dealii::TrilinosWrappers::PreconditionILU *>(
              preconditioner.get());
      derived_ptr->initialize(matrix);
      break;
    }
    case Preconditioner::AMG: {
      preconditioner =
          std::make_unique<dealii::TrilinosWrappers::PreconditionAMG>();
      // We have to perform a dynamic cast as the PreconditionBase class has
      // not the initialize method
      auto derived_ptr =
          dynamic_cast<dealii::TrilinosWrappers::PreconditionAMG *>(
              preconditioner.get());
      derived_ptr->initialize(matrix);
      break;
    }
    default: {
      preconditioner =
          std::make_unique<dealii::TrilinosWrappers::PreconditionIdentity>();
      // We have to perform a dynamic cast as the PreconditionBase class has
      // not the initialize method
      auto derived_ptr =
          dynamic_cast<dealii::TrilinosWrappers::PreconditionIdentity *>(
              preconditioner.get());
      derived_ptr->initialize(matrix);
    }
    }
  }
  /**
   * @brief Solves a linear system using the configured solver and
   * preconditioner.
   * @tparam Solver The solver type
   * @tparam Matrix The lhs matrix type
   * @tparam Solution The solution vector type
   * @tparam Vector The rhs vector type
   * @tparam Preconditioner The preconditioner type
   * @param solver The solver object.
   * @param matrix The matrix of the linear system.
   * @param solution The solution vector.
   * @param rhs The right-hand side vector.
   * @param preconditioner The preconditioner object.
   */
  template <typename Solver, typename Matrix, typename Solution,
            typename Vector, typename Preconditioner>
  void solve(std::unique_ptr<Solver> const &solver, Matrix const &matrix,
             Solution &solution, Vector const &rhs,
             std::unique_ptr<Preconditioner> const &preconditioner) {
    // We have to cast the solver pointer to the derived type as the dealii base
    // solver class has not the solve method
    switch (solver_type) {
    case SolverType::GMRES: {
      auto derived_ptr =
          dynamic_cast<dealii::SolverGMRES<Vector> *>(solver.get());
      derived_ptr->solve(matrix, solution, rhs, *preconditioner);
      break;
    }
    case SolverType::BiCGSTAB: {
      auto derived_ptr =
          dynamic_cast<dealii::SolverBicgstab<Vector> *>(solver.get());
      derived_ptr->solve(matrix, solution, rhs, *preconditioner);
      break;
    }
    default: {
      auto derived_ptr =
          dynamic_cast<dealii::SolverGMRES<Vector> *>(solver.get());
      derived_ptr->solve(matrix, solution, rhs, *preconditioner);
    }
    }
  }
  /**
   * @brief Gets the solver type.
   * @return The solver type.
   */
  SolverType get_solver_type() const { return solver_type; }
  /**
   * @brief Gets the preconditioner type.
   * @return The preconditioner type.
   */
  Preconditioner get_preconditioner_type() const { return preconditioner_type; }
  /**
   * @brief Gets the tolerance value.
   * @return The tolerance type.
   */
  Scalar get_tolerance() const { return tolerance; }
  /**
   * @brief Gets the max iteration multiplier value.
   * @return The max iteration multiplier type.
   */
  Scalar get_max_iterations() const { return max_iterations; }

private:
  SolverType solver_type;             /**< Solver type */
  Preconditioner preconditioner_type; /**< Preconditioner type */
  Scalar tolerance;                   /**< Tolerance value */
  Scalar max_iterations;              /**< Max iteration multiplier value */
};

/**
 * @brief << operator.
 * @tparam Scalar The scalar data type.
 * @param out The ostream object.
 * @param lsu The linear solver utility.
 */
template <typename Scalar>
void operator<<(std::ostream &out, LinearSolverUtility<Scalar> const &lsu) {
  out << "  Solver: "
      << LinearSolverUtility<
             Scalar>::solver_type_matcher_rev[lsu.get_solver_type()]
      << std::endl;
  out << "  Preconditioner: "
      << LinearSolverUtility<Scalar>::preconditioner_type_matcher_rev
             [lsu.get_preconditioner_type()]
      << std::endl;
  out << "  Tolerance: " << lsu.get_tolerance() << std::endl;
  out << "  Max iterations multiplier: " << lsu.get_max_iterations()
      << std::endl;
}

template <typename Scalar>
std::map<std::string, typename LinearSolverUtility<Scalar>::SolverType>
    LinearSolverUtility<Scalar>::solver_type_matcher = {
        {"BiCGSTAB", LinearSolverUtility<Scalar>::SolverType::BiCGSTAB},
        {"GMRES", LinearSolverUtility<Scalar>::SolverType::GMRES}};

template <typename Scalar>
std::map<typename LinearSolverUtility<Scalar>::SolverType, std::string>
    LinearSolverUtility<Scalar>::solver_type_matcher_rev = {
        {LinearSolverUtility<Scalar>::SolverType::BiCGSTAB, "BiCGSTAB"},
        {LinearSolverUtility<Scalar>::SolverType::GMRES, "GMRES"}};

template <typename Scalar>
std::map<std::string, typename LinearSolverUtility<Scalar>::Preconditioner>
    LinearSolverUtility<Scalar>::preconditioner_type_matcher = {
        {"IDENTITY", LinearSolverUtility<Scalar>::Preconditioner::IDENTITY},
        {"ILU", LinearSolverUtility<Scalar>::Preconditioner::ILU},
        {"SOR", LinearSolverUtility<Scalar>::Preconditioner::SOR},
        {"AMG", LinearSolverUtility<Scalar>::Preconditioner::AMG},
        {"SSOR", LinearSolverUtility<Scalar>::Preconditioner::SSOR}};

template <typename Scalar>
std::map<typename LinearSolverUtility<Scalar>::Preconditioner, std::string>
    LinearSolverUtility<Scalar>::preconditioner_type_matcher_rev = {
        {LinearSolverUtility<Scalar>::Preconditioner::IDENTITY, "IDENTITY"},
        {LinearSolverUtility<Scalar>::Preconditioner::ILU, "ILU"},
        {LinearSolverUtility<Scalar>::Preconditioner::AMG, "AMG"},
        {LinearSolverUtility<Scalar>::Preconditioner::SOR, "SOR"},
        {LinearSolverUtility<Scalar>::Preconditioner::SSOR, "SSOR"}};

#endif // LINEAR_SOLVER_CONFIGURATION_HPP
