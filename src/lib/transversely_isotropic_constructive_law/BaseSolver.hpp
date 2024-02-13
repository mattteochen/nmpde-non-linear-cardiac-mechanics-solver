/**
 * @file BaseSolver.hpp
 * @brief Header file defining the abstract base solver class for materials
 * having the transversely isotropic constructive law.
 */

#ifndef BASESOLVER_HPP
#define BASESOLVER_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
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
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#include <Reporter.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <transversely_isotropic_constructive_law/BoundariesUtility.hpp>
#include <transversely_isotropic_constructive_law/LinearSolverUtility.hpp>
#include <transversely_isotropic_constructive_law/NewtonSolverUtility.hpp>

using namespace dealii;

/**
 * @class BaseSolver
 * @brief Abstract class representing a base solver for non linear cardiac
 * mechanics by using the transversely isotropic constitutive law by Guccione et
 * al. (https://pubmed.ncbi.nlm.nih.gov/8550635/).
 * @tparam dim The problem dimension
 * @tparam Scalar The scalar type for the problem, by default a double
 */
template <int dim, typename Scalar = double> class BaseSolver {
  /**
   * Alias for the base preconditioner pointer
   */
  using Preconditioner = std::unique_ptr<TrilinosWrappers::PreconditionBase>;
  /**
   * Alias for the base linear solver pointer
   */
  using LinearSolver =
      std::unique_ptr<SolverBase<TrilinosWrappers::MPI::Vector>>;

public:
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
     */
    ConstantPressureFunction(const double pressure_) : pressure(pressure_) {}
    /**
     * @brief Copy operator
     * @param other The input ConstantPressureFunction object
     */
    void operator=(ConstantPressureFunction &&other) {
      pressure = other.pressure;
    }
    /**
     * @brief Evaluate the pressure at a given point
     * @param p The evaluation point
     * @param component The component id
     */
    virtual double value(const Point<dim> & /*p*/,
                         const unsigned int /*component*/ = 0) const override {
      return pressure;
    }

  protected:
    /**
     * @brief Pressure value in kPa
     */
    double pressure = 0.0;
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
    NumberType q = static_cast<Scalar>(0);

  public:
    /**
     * @brief Retrieve the Q value
     * @return The exponent q
     */
    NumberType get_q() { return q; }
    /**
     * @brief Evaluate the Q exponent at a given green Lagrange strain tensor
     * @tparam TensorType The specialised dealii:Tensor type representing the
     * green Lagrange strain tensor
     * @param gst A green Lagrange strain tensor
     * @return The exponent q
     */
    template <typename TensorType> NumberType compute(TensorType const &gst) {
      return q = Material::b_f * gst[0][0] * gst[0][0] +
                 Material::b_t *
                     (gst[1][1] * gst[1][1] + gst[2][2] * gst[2][2] +
                      gst[1][2] * gst[1][2] + gst[2][1] * gst[2][1]) +
                 Material::b_fs *
                     (gst[0][1] * gst[0][1] + gst[1][0] * gst[1][0] +
                      gst[0][2] * gst[0][2] + gst[2][0] * gst[2][0]);
    }
  };
  /**
   * @brief Initialise boundaries tag. This must be implemented by the derived
   * class
   */
  virtual void initialise_boundaries_tag() = 0;
  /**
   * @brief Determine if a given faces is at a Newmann boundary
   * @param face The face values
   * @return The requested query
   */
  bool is_face_at_newmann_boundary(const int face) {
    return newmann_boundary_faces.find(face) != newmann_boundary_faces.end();
  }
  /**
   * @brief Constructor
   * @param parameters_file_name_ The parameters file name
   * @param mesh_file_name_ The mesh file name
   * @param problem_name_ The problem name
   */
  BaseSolver(const std::string &parameters_file_name_,
             const std::string &mesh_file_name_,
             const std::string &problem_name_)
      : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
        mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)),
        pcout(std::cout, mpi_rank == 0), mesh_file_name(mesh_file_name_),
        mesh(MPI_COMM_WORLD), problem_name(problem_name_) {
    // Initalized the parameter handler
    parse_parameters(parameters_file_name_);

    // Set the Piola Kirchhoff b values
    piola_kirchhoff_b_weights[{0, 0}] = Material::b_f;
    piola_kirchhoff_b_weights[{1, 1}] = Material::b_t;
    piola_kirchhoff_b_weights[{2, 2}] = Material::b_t;
    piola_kirchhoff_b_weights[{1, 2}] = Material::b_t;
    piola_kirchhoff_b_weights[{2, 1}] = Material::b_t;
    piola_kirchhoff_b_weights[{0, 2}] = Material::b_fs;
    piola_kirchhoff_b_weights[{2, 0}] = Material::b_fs;
    piola_kirchhoff_b_weights[{0, 1}] = Material::b_fs;
    piola_kirchhoff_b_weights[{1, 0}] = Material::b_fs;
  }
  /**
   * @brief Virtual destructor for abstract class
   */
  virtual ~BaseSolver() {}

  void setup();

  void solve_newton();

  void output() const;

protected:
  virtual void assemble_system();

  virtual void solve_system();

  void parse_parameters(const std::string &parameters_file_name_);
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
  const std::string &mesh_file_name;
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
};

/**
 * @brief Define static member of BaseSolver<dim, Scalar>::Material::b_f.
 * Needed for linking.
 */
template <int dim, typename Scalar>
Scalar BaseSolver<dim, Scalar>::Material::b_f;
/**
 * @brief Define static member of BaseSolver<dim, Scalar>::Material::b_t
 * Needed for linking.
 */
template <int dim, typename Scalar>
Scalar BaseSolver<dim, Scalar>::Material::b_t;
/**
 * @brief Define static member of BaseSolver<dim, Scalar>::Material::b_fs
 * Needed for linking.
 */
template <int dim, typename Scalar>
Scalar BaseSolver<dim, Scalar>::Material::b_fs;
/**
 * @brief Define static member of BaseSolver<dim, Scalar>::Material::C
 * Needed for linking.
 */
template <int dim, typename Scalar> Scalar BaseSolver<dim, Scalar>::Material::C;

#endif // BASESOLVER_HPP
