#ifndef IDEALIZEDLV_HPP
#define IDEALIZEDLV_HPP

#include <deal.II/base/conditional_ostream.h>
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

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>

#include "../../utils/Assert.hpp"
#include "../../utils/Utils.hpp"

using namespace dealii;

/**
 * @class Class representing the Slab Cubic solver
 */
class IdealizedLV {
 public:
  /**
   * Const physical dimension
   */
  static constexpr unsigned int dim = 3;
  /**
   * Const boundaries faces
   */
  static constexpr unsigned int boundary_faces_num = 6;
  /**
   * Material related parameters
   */
  template <typename Scalar = double>
  struct Material {
    static constexpr Scalar b_f = static_cast<Scalar>(1);
    static constexpr Scalar b_t = static_cast<Scalar>(1);
    static constexpr Scalar b_fs = static_cast<Scalar>(1);
    static constexpr Scalar default_C = static_cast<Scalar>(10000);
  };
  /**
   * @class Class representing the pressure component
   */
  class FunctionPressure : Function<dim> {
   public:
    /**
     * @brief Evaluate the pressure at a given point
     * @param p The evaluation point
     * @param component The component id
     */
    virtual double value(const Point<dim> & /*p*/,
                         const unsigned int /*component*/ = 0) const override {
      return 4.0;
    }
  };
  /**
   * @class Class representing W's exponent Q
   */
  template <typename Scalar>
  class ExponentQ : Function<dim> {
   protected:
    /**
     * The values of the exponent
     */
    Scalar q = static_cast<Scalar>(0);
    /**
     * An initialisation guard
     */
    uint8_t initialised = 0;

   public:
    /**
     * @brief Retrieve the Q value
     * @return The exponent q
     */
    Scalar get_q() {
      ASSERT(initialised, "The exponent Q has not been initialised");
      return q;
    }
    /**
     * @brief Evaluate the Q exponent at a given green Lagrange strain tensor
     * @tparam TensorType The specialised dealii:Tensor type representing the
     * green Lagrange strain tensor
     * @param gst A green Lagrange strain tensor
     * @return The exponent q
     */
    template <typename TensorType>
    void compute(const TensorType &gst) {
      q = Material<double>::b_f * gst[0][0] * gst[0][0] +
          Material<double>::b_t *
              (gst[1][1] * gst[1][1] + gst[2][2] * gst[2][2] +
               gst[1][2] * gst[1][2] + gst[2][1] * gst[2][1]) +
          Material<double>::b_fs *
              (gst[0][1] * gst[0][1] + gst[1][0] * gst[1][0] +
               gst[0][2] * gst[0][2] + gst[2][0] * gst[2][0]);
      initialised = 1;
    }
  };
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
   * @param mesh_file_name_ The mesh file name
   * @param r_ The polynomial degree
   */
  IdealizedLV(const std::string &mesh_file_name_, const unsigned int &r_)
      : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
        mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)),
        pcout(std::cout, mpi_rank == 0),
        mesh_file_name(mesh_file_name_),
        r(r_),
        mesh(MPI_COMM_WORLD) {
    // Set the Piola Kirchhoff b values
    piola_kirchhoff_b_weights[{0, 0}] = Material<double>::b_f;
    piola_kirchhoff_b_weights[{1, 1}] = Material<double>::b_t;
    piola_kirchhoff_b_weights[{2, 2}] = Material<double>::b_t;
    piola_kirchhoff_b_weights[{1, 2}] = Material<double>::b_t;
    piola_kirchhoff_b_weights[{2, 1}] = Material<double>::b_t;
    piola_kirchhoff_b_weights[{0, 2}] = Material<double>::b_fs;
    piola_kirchhoff_b_weights[{2, 0}] = Material<double>::b_fs;
    piola_kirchhoff_b_weights[{0, 1}] = Material<double>::b_fs;
    piola_kirchhoff_b_weights[{1, 0}] = Material<double>::b_fs;

    // Set Newmann boundary faces
    newmann_boundary_faces.insert(20);

    // Set Dirichlet boundary faces
    dirichlet_boundary_functions[50] = &zero_function;
  }
  // Initialization.
  void setup();
  // Solve the problem using Newton's method.
  void solve_newton();
  // Output.
  void output() const;

 protected:
  // Assemble the tangent problem.
  void assemble_system();
  // Solve the tangent problem.
  void solve_system();

  // MPI parallel. /////////////////////////////////////////////////////////////

  // Number of MPI processes.
  const unsigned int mpi_size;
  // This MPI process.
  const unsigned int mpi_rank;
  // Parallel output stream.
  ConditionalOStream pcout;

  // Problem definition. ///////////////////////////////////////////////////////
  /**
   * Utility zero function for Dirichilet boundary
   */
  Functions::ZeroFunction<dim> zero_function;
  /**
   * The function pressure
   */
  FunctionPressure pressure;
  /**
   * A set of Newmann boundary faces
   */
  std::set<int> newmann_boundary_faces;
  /**
   * A map of Dirichlet boundary faces
   */
  std::map<types::boundary_id, const Function<dim> *>
      dirichlet_boundary_functions;
  /**
   * The b_ij coefficients for the Piola Kiochhoff tensor
   */
  std::map<std::pair<int, int>, double> piola_kirchhoff_b_weights;

  // Discretization. ///////////////////////////////////////////////////////////

  // Mesh file name.
  const std::string &mesh_file_name;
  // Polynomial degree.
  const unsigned int r;
  // Mesh.
  parallel::fullydistributed::Triangulation<dim> mesh;
  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;
  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;
  // Quadrature formula for face integrals.
  std::unique_ptr<Quadrature<dim - 1>> quadrature_face;
  // DoF handler.
  DoFHandler<dim> dof_handler;
  // DoFs owned by current process.
  IndexSet locally_owned_dofs;
  // DoFs relevant to the current process (including ghost DoFs).
  IndexSet locally_relevant_dofs;
  // Jacobian matrix.
  TrilinosWrappers::SparseMatrix jacobian_matrix;
  // Residual vector.
  TrilinosWrappers::MPI::Vector residual_vector;
  // Solution increment (without ghost elements).
  TrilinosWrappers::MPI::Vector delta_owned;
  // System solution (without ghost elements).
  TrilinosWrappers::MPI::Vector solution_owned;
  // System solution (including ghost elements).
  TrilinosWrappers::MPI::Vector solution;
};

#endif
