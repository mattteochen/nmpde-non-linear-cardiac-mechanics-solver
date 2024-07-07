/** @file BaseSolverGuccione.hpp @brief Header file defining the abstract base
 * solver class for materials following the Guccione et al (Guccione JM, Costa
 * KD, McCulloch AD. 1995 Finite element stress analysis of left ventricular
 * mechanics in the beating dog heart) constructive law.
 */

#ifndef BASESOLVERGUCCIONE_HPP
#define BASESOLVERGUCCIONE_HPP

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
#include <type_traits>
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
 * @class BaseSolverGuccione
 * @brief Abstract class representing a base solver for non linear cardiac
 * mechanics by using the isotropic or transversely isotropic constitutive law
 * by Guccione et al. (https://pubmed.ncbi.nlm.nih.gov/8550635/).
 * @tparam dim The problem dimension
 * @tparam Scalar The scalar type for the problem, by default a double
 */
template <int dim, typename Scalar = double> class BaseSolverGuccione {
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
   * @brief Material related parameters. Reference papaer:
   * https://pubmed.ncbi.nlm.nih.gov/26807042/
   */
  struct Material {
    /**
     * b_f coefficient
     */
    static Scalar b_f;
    /**
     * b_t coefficient
     */
    static Scalar b_t;
    /**
     * b_fs coefficient
     */
    static Scalar b_fs;
    /**
     * C coefficient
     */
    static Scalar C;

    /**
     * Retrive a std::string representing the current configuration
     * @return The configuration string
     */
    static std::string show() {
      return "  b_f: " + std::to_string(Material::b_f) + "\n" +
             "  b_t : " + std::to_string(b_t) + "\n" +
             "  b_fs: " + std::to_string(b_fs) + "\n" +
             "  C: " + std::to_string(C) + "\n";
    }
  };
  /**
   * @brief Class representing the applied pressure component
   */
  class ConstantPressureFunction : Function<dim> {
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
                             const double reduction_factor_,
                             const double reduction_factor_inc_)
        : pressure(pressure_), reduction_factor(reduction_factor_),
          reduction_increment_strat(reduction_factor_inc_) {}
    /**
     * @brief Copy operator
     * @param other The input ConstantPressureFunction object
     */
    void operator=(ConstantPressureFunction &&other) {
      pressure = other.pressure;
      reduction_factor = other.reduction_factor;
      reduction_increment_strat = other.reduction_increment_strat;
    }
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
   * @brief Class representing the exponent Q.
   * This parameter is used by the construction of the strain energy function:
   * https://pubmed.ncbi.nlm.nih.gov/26807042/
   * @tparam NumberType The exponent scalar type
   */
  template <typename NumberType> class ExponentQ : Function<dim> {
  protected:
    /**
     * The values of the exponent
     */
    NumberType q = NumberType(0.0);
    /**
     * NumberType representation of Material::b_f coefficient
     */
    const NumberType b_f = NumberType(Material::b_f);
    /**
     * NumberType representation of Material::b_t coefficient
     */
    const NumberType b_t = NumberType(Material::b_t);
    /**
     * NumberType representation of Material::b_fs coefficient
     */
    const NumberType b_fs = NumberType(Material::b_fs);

  public:
    /**
     * @brief Retrieve the Q value
     * @return The exponent q
     */
    NumberType get_q() const { return q; }
    /**
     * @brief Evaluate the Q exponent at a given green Lagrange strain tensor
     * @tparam TensorType The specialised dealii:Tensor type representing the
     * green Lagrange strain tensor
     * @param gst A green Lagrange strain tensor
     * @return The exponent q
     */
    template <typename TensorType> NumberType compute(TensorType const &gst) {
      return q = b_f * gst[0][0] * gst[0][0] +
                 b_t * (gst[1][1] * gst[1][1] + gst[2][2] * gst[2][2] +
                        gst[1][2] * gst[1][2] + gst[2][1] * gst[2][1]) +
                 b_fs * (gst[0][1] * gst[0][1] + gst[1][0] * gst[1][0] +
                         gst[0][2] * gst[0][2] + gst[2][0] * gst[2][0]);
    }
  };
  /**
   * @brief Determine if a given faces is at a Newmann boundary
   * @param face The face values
   * @return The requested query
   */
  bool is_face_at_newmann_boundary(const unsigned face) {
    return newmann_boundary_faces.find(face) != newmann_boundary_faces.end();
  }
  /**
   * @brief Constructor
   * @param problem_name_ The problem name
   */
  BaseSolverGuccione(const std::string &problem_name_)
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
  virtual ~BaseSolverGuccione() {}

  virtual void setup();

  virtual void solve_newton();

  virtual void output() const;

protected:
  /**
   * @brief Initialise boundaries tag (pure virtual). This must be implemented
   * by the derived class
   */
  virtual void initialise_boundaries_tag() = 0;
  /**
   * @brief Initialise parameters handler (pure virtual). This must be
   * implemented by the derived class
   * @param file_ The input parameters file
   */
  virtual void initialize_param_handler(const std::string &file_) = 0;
  /**
   * @brief Initialise Piola Kirchhoff weights. There represent the material
   * coefficient that have to be multiplied to the tensor pk[i][j]. This must be
   * implemented by the derived class
   */
  virtual void initialize_pk_weights() = 0;

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
   * A set of Newmann boundary faces
   */
  std::set<unsigned int> newmann_boundary_faces;
  /**
   * A map of Dirichlet boundary faces
   */
  std::map<types::boundary_id, const Function<dim> *>
      dirichlet_boundary_functions;
  /**
   * The b_ij coefficients for the Piola Kiochhoff tensor
   */
  std::map<std::pair<int, int>, double> piola_kirchhoff_b_weights;
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
   * The newmann boundary quadrature object representation
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
 * @brief Define static member of BaseSolverGuccione::Material::b_f. Needed for
 * linking.
 */
template <int dim, typename Scalar>
Scalar BaseSolverGuccione<dim, Scalar>::Material::b_f;
/**
 * @brief Define static member of BaseSolverGuccione<dim, Scalar>::Material::b_t
 * Needed for linking.
 */
template <int dim, typename Scalar>
Scalar BaseSolverGuccione<dim, Scalar>::Material::b_t;
/**
 * @brief Define static member of BaseSolverGuccione::Material::b_fs Needed for
 * linking.
 */
template <int dim, typename Scalar>
Scalar BaseSolverGuccione<dim, Scalar>::Material::b_fs;
/**
 * @brief Define static member of BaseSolverGuccione::Material::C
 * Needed for linking.
 */
template <int dim, typename Scalar>
Scalar BaseSolverGuccione<dim, Scalar>::Material::C;

#endif // BASESOLVERGUCCIONE_HPP
