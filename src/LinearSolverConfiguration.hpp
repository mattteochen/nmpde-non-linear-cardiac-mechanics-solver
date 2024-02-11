#ifndef LINEAR_SOLVER_CONFIGURATION_HPP
#define LINEAR_SOLVER_CONFIGURATION_HPP

#include <map>
#include <string>

template <typename Scalar = double>
class LinearSolverConfiguration {
public:
  enum class SolverType {
    GMRES,
    BiCGSTAB
  };

  enum class Preconditioner {
    IDENTITY,
    SSOR
  };

  LinearSolverConfiguration<Scalar>() {}

  LinearSolverConfiguration<Scalar>(SolverType solver_type_, Preconditioner preconditioner_, Scalar tolerance_, Scalar max_iterations_)
    :solver_type(solver_type_), preconditioner(preconditioner_), tolerance(tolerance_), max_iterations(max_iterations_) {}

  void operator=(LinearSolverConfiguration<Scalar>&& other_) {
    solver_type = other_.solver_type;
    preconditioner = other_.preconditioner;
    tolerance = other_.tolerance;
    max_iterations = other_.max_iterations;
  }

  static std::map<std::string, SolverType> solver_type_matcher;

  static std::map<std::string, Preconditioner> preconditioner_type_matcher;

  SolverType solver_type;
  Preconditioner preconditioner;
  Scalar tolerance;
  Scalar max_iterations;
};

template <typename Scalar>
std::map<std::string, typename LinearSolverConfiguration<Scalar>::SolverType>
    LinearSolverConfiguration<Scalar>::solver_type_matcher = {
        {"GMRES", LinearSolverConfiguration<Scalar>::SolverType::GMRES},
        {"BiCGSTAB", LinearSolverConfiguration<Scalar>::SolverType::BiCGSTAB},
};

template <typename Scalar>
std::map<std::string,
         typename LinearSolverConfiguration<Scalar>::Preconditioner>
    LinearSolverConfiguration<Scalar>::preconditioner_type_matcher = {
        {"IDENTITY",
         LinearSolverConfiguration<Scalar>::Preconditioner::IDENTITY},
        {"SSOR", LinearSolverConfiguration<Scalar>::Preconditioner::SSOR}};

#endif //LINEAR_SOLVER_CONFIGURATION_HPP
