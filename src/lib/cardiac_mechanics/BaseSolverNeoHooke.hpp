/**
 * @file BaseSolverNeoHooke.hpp
 * @brief Header file defining the abstract base solver class for materials
 * having the neo Hooke constructive law.
 */

#ifndef BASESOLVERNEOHOOKE_HPP
#define BASESOLVERNEOHOOKE_HPP

#include <Reporter.hpp>
#include <cardiac_mechanics/BoundariesUtility.hpp>
#include <cardiac_mechanics/LinearSolverUtility.hpp>
#include <cardiac_mechanics/NewtonSolverUtility.hpp>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#if DEAL_II_VERSION_MAJOR >= 9 && defined(DEAL_II_WITH_TRILINOS)
#include <deal.II/differentiation/ad.h>
#define ENABLE_SACADO_FORMULATION
#endif
// These must be included below the AD headers so that
// their math functions are available for use in the
// definition of tensors and kinematic quantities
#include <Sacado.hpp>
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#ifdef BUILD_TYPE_DEBUG
#warning "Building in DEBUG mode"
#include <string>
#endif

using namespace dealii;

/**
 * @class BaseSolverNeoHooke
 * @brief Abstract class representing a base solver for non linear cardiac
 * mechanics by using the neo Hooke constitutive law.
 * @tparam dim The problem dimension
 * @tparam Scalar The scalar type for the problem, by default a double
 */
template <int dim, typename Scalar = double> class BaseSolverNeoHooke {
  /**
   * Alias for the base preconditioner pointer
   */
  using Preconditioner = std::unique_ptr<TrilinosWrappers::PreconditionBase>;
  /**
   * Alias for the base linear solver pointer
   */
  using LinearSolver =
      std::unique_ptr<SolverBase<TrilinosWrappers::MPI::Vector>>;
  /**
   * Sacado automatic differentiation type code from
   */
  static constexpr Differentiation::AD::NumberTypes ADTypeCode =
      Differentiation::AD::NumberTypes::sacado_dfad;
  /**
   * Alias for the AD helper
   */
  using ADHelper =
      Differentiation::AD::ResidualLinearization<ADTypeCode, Scalar>;
  /**
   * Alias for the AD number type
   */
  using ADNumberType = typename ADHelper::ad_type;

public:
  /**
   * @brief: Triangulation geometry
   */
  enum class TriangulationType { T, Q };

  /**
   * @brief Material related parameters.
   */
  struct Material {
    /**
     * mu coefficient
     */
    static Scalar mu;
    /**
     * lambda coefficient
     */
    static Scalar lambda;

    /**
     * Retrive a std::string representing the current configuration
     * @return The configuration string
     */
    static std::string show() {
      return "  mu: " + std::to_string(mu) + "\n" +
             "  lambda: " + std::to_string(lambda) + "\n";
    }
  };
  /**
   * @brief Class representing the applied pressure component
   */
  class ConstantPressureFunction : Function<dim, Scalar> {
  public:
    /**
     * @brief Default constructor
     */
    ConstantPressureFunction() {}
    /**
     * @brief Parametrized constructor
     * @param pressure_ The pressure value;
     * @param reduction_factor_ The pressure reduction factor to accomodate the
     * modified Newton iteration
     * @param reduction_factor_inc_ The incremental value for the above
     */
    ConstantPressureFunction(const Scalar pressure_,
                             const Scalar reduction_factor_,
                             const Scalar reduction_factor_inc_)
        : pressure(pressure_), reduction_factor(reduction_factor_),
          reduction_increment_strat(reduction_factor_inc_) {}
    /**
     * @brief Retrieve the pressure value
     * @return The configured pressure value
     */
    Scalar value() const {
      return static_cast<Scalar>(static_cast<double>(pressure) *
                                 reduction_factor);
    }
    /**
     * @brief Evaluate the pressure at a given point
     * @param p The evaluation point
     * @param component The component id
     */
    virtual Scalar value(const Point<dim> & /*p*/,
                         const unsigned int /*component*/ = 0) const override {
      return static_cast<Scalar>(static_cast<double>(pressure) *
                                 reduction_factor);
    }
    /**
     * @brief Increment the reduction factor
     */
    void increment_reduction_factor() {
      reduction_factor += reduction_increment_strat;
      reduction_factor = std::min(reduction_factor, 1.0);
    }
    /**
     * @brief Get the reduction factor
     * @return The current reduction factor
     */
    double get_reduction_factor() const { return reduction_factor; }

  protected:
    /**
     * @brief Pressure value in Pa
     */
    Scalar pressure;
    /**
     * @brief A reduction factor to be applied to the set pressure value
     */
    double reduction_factor;
    /**
     * The pressure reduction factor increment value. Refer to the reference
     * paper for the employed modfied Newton iteration
     */
    double reduction_increment_strat;
  };
  /**
   * @brief Determine if a given faces is at a Neumann boundary
   * @param face The face values
   * @return The requested query
   */
  bool is_face_at_neumann_boundary(const unsigned face) const {
    return neumann_boundary_faces.find(face) != neumann_boundary_faces.end();
  }
  /**
   * @brief Constructor
   * @param problem_name_ The problem name
   */
  BaseSolverNeoHooke(const std::string &problem_name_)
      : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
        mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)),
        pcout(std::cout, mpi_rank == 0), mesh(MPI_COMM_WORLD),
        problem_name(problem_name_) {
    // Trillinos library supports only doubles for now.
    static_assert(std::is_same_v<Scalar, double>, "Type not supported");
  }
  /**
   * @brief Virtual destructor for abstract class
   */
  virtual ~BaseSolverNeoHooke() {}

  virtual void setup();

  virtual void solve_newton();

  virtual void output() const;

protected:
  /**
   * @brief Initialise boundaries tag. This must be implemented by the derived
   * class
   */
  virtual void initialise_boundaries_tag() = 0;
  /**
   * @brief Initialise parameters handler. This must be implemented by the
   * derived class
   * @param file_ The input parameters file
   */
  virtual void initialize_param_handler(const std::string &file_) = 0;

  virtual void compute_piola_kirchhoff(
      Tensor<2, dim, ADNumberType> &out_tensor,
      const Tensor<2, dim, ADNumberType> &solution_gradient_quadrature,
      const unsigned cell_index);

  virtual void assemble_system();

  virtual unsigned solve_system();

  virtual void declare_parameters();

  virtual void parse_parameters(const std::string &parameters_file_name_);
  /**
   * MPI size
   */
  const unsigned int mpi_size;
  /**
   * MPI rank
   */
  const unsigned int mpi_rank;
  /**
   * MPI conditional ostream
   */
  ConditionalOStream pcout;
  /**
   * The function pressure
   */
  ConstantPressureFunction pressure;
  /**
   * A set of Neumann boundary faces
   */
  std::set<unsigned int> neumann_boundary_faces;
  /**
   * A map of Dirichlet boundary faces
   */
  std::map<types::boundary_id, const Function<dim, Scalar> *>
      dirichlet_boundary_functions;
  /**
   * The input mesh file name
   */
  std::string mesh_file_name;
  /**
   * The polynomial degree
   */
  unsigned int r;
  /**
   * The distributed triangulation
   */
  parallel::fullydistributed::Triangulation<dim> mesh;
  /**
   * The finite element object representation
   */
  std::unique_ptr<FiniteElement<dim>> fe;
  /**
   * The quadrature object representation
   */
  std::unique_ptr<Quadrature<dim>> quadrature;
  /**
   * The neumann boundary quadrature object representation
   */
  std::unique_ptr<Quadrature<dim - 1>> quadrature_face;
  /**
   * The degree of freedom handler
   */
  DoFHandler<dim> dof_handler;
  /**
   * The rank locally owned dofs
   */
  IndexSet locally_owned_dofs;
  /**
   * The rank locally relevant dofs
   */
  IndexSet locally_relevant_dofs;
  /**
   * The Jacobian matrix for the Newton iteration
   */
  TrilinosWrappers::SparseMatrix jacobian_matrix;
  /**
   * The residual vector
   */
  TrilinosWrappers::MPI::Vector residual_vector;
  /**
   * The solution increment with no ghost elements
   */
  TrilinosWrappers::MPI::Vector delta_owned;
  /**
   * The system solution with no ghost elements
   */
  TrilinosWrappers::MPI::Vector solution_owned;
  /**
   * The system solution with ghost elements
   */
  TrilinosWrappers::MPI::Vector solution;
  /**
   * The problem name
   */
  const std::string &problem_name;
  /**
   * The problem parameter handler
   */
  ParameterHandler prm;
  /**
   * The linear solver utiility
   */
  LinearSolverUtility<Scalar> linear_solver_utility;
  /**
   * The Newton method solver utiility
   */
  NewtonSolverUtility<Scalar> newton_solver_utility;
  /**
   * The boundaries tag utiility
   */
  BoundariesUtility boundaries_utility;
  /**
   * The mesh triangulation type
   */
  TriangulationType triangulation_type;
};

/**
 * @brief Define static member of BaseSolverNeoHooke::Material::mu.
 * Needed for linking.
 */
template <int dim, typename Scalar>
Scalar BaseSolverNeoHooke<dim, Scalar>::Material::mu;
/**
 * @brief Define static member of BaseSolverNeoHooke::Material::lambda Needed for
 * linking.
 */
template <int dim, typename Scalar>
Scalar BaseSolverNeoHooke<dim, Scalar>::Material::lambda;

#endif // BASESOLVERNEOHOOKE_HPP
