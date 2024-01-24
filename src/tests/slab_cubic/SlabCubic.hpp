#ifndef SLAB_CUBIC_HPP
#define SLAB_CUBIC_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>

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

#include "../../utils/Utils.hpp"
#include "../../utils/Assert.hpp"

using namespace dealii;
using ui = uint32_t;

/**
 * Class representing the Slab Cubic solver
 */
class SlabCubic
{
  /**
   * Const physical dimension
   */
  static constexpr unsigned int dim = 3;

  using tensor = Tensor<2, dim>;
public:

  /**
   * Const boundaries faces
   */
  static constexpr unsigned int boundary_faces_num = 6;

  template<typename Scalar = double>
  struct Material
  {
    static constexpr Scalar b_f = static_cast<Scalar>(8);
    static constexpr Scalar b_t = static_cast<Scalar>(2);
    static constexpr Scalar b_fs = static_cast<Scalar>(4);
    static constexpr Scalar default_C = static_cast<Scalar>(2000);
  };

  /**
   * A class representing the pressure component
   */
  class FunctionPressure : Function<dim>
  {
  public:
    virtual double value(const Point<dim> & /*p*/,
                         const unsigned int /*component*/ = 0) const override {
      return 4.0;
    }
  };

  /**
   * A class representing W's exponent Q
   */
  template<typename Scalar>
  class ExponentQ : Function<dim>
  {
  protected:
    Scalar q = static_cast<Scalar>(0);
    uint8_t initialised = 0;
  public:
    /**
     * Retrieve the Q value
     * @return The exponent Q
     */
    Scalar get_q() {
      ASSERT(initialised, "The exponent Q has not been initialised");
      return q;
    }

    /**
     * Evaluate the Q exponent at a given green Lagrange strain tensor
     * @param gst A DeformationGradientTensor
     * @return The exponent Q
     */
    void compute(const tensor& gst) {
      q = Material<double>::b_f * gst[0][0] * gst[0][0] +
             Material<double>::b_t * (gst[1][1] * gst[1][1] + gst[2][2] * gst[2][2] +
                    gst[1][2] * gst[1][2] + gst[2][1] * gst[2][1]) +
             Material<double>::b_fs * (gst[0][1] * gst[0][1] + gst[1][0] * gst[1][0] +
                     gst[0][2] * gst[0][2] + gst[2][0] * gst[2][0]);
      initialised = 1;
    }
  };

  /**
   * Constructor
   * @param mesh_file_name_ The mesh file name
   * @param r_ The polynomial degree
   */
  SlabCubic(const std::string &mesh_file_name_, const unsigned int &r_)
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , mesh_file_name(mesh_file_name_)
    , r(r_)
    , mesh(MPI_COMM_WORLD)
  {
    PK_b_weights[{0,0}] = Material<double>::b_f; 
    PK_b_weights[{1,1}] = Material<double>::b_t; 
    PK_b_weights[{2,2}] = Material<double>::b_t; 
    PK_b_weights[{1,2}] = Material<double>::b_t; 
    PK_b_weights[{2,1}] = Material<double>::b_t; 
    PK_b_weights[{0,2}] = Material<double>::b_fs; 
    PK_b_weights[{2,0}] = Material<double>::b_fs; 
    PK_b_weights[{0,1}] = Material<double>::b_fs; 
    PK_b_weights[{1,0}] = Material<double>::b_fs; 
  }

  // Initialization.
  void
  setup();

  // Solve the problem using Newton's method.
  void
  solve_newton();

  // Output.
  void
  output() const;

protected:
  // Assemble the tangent problem.
  void
  assemble_system();

  // Solve the tangent problem.
  void
  solve_system();

  // MPI parallel. /////////////////////////////////////////////////////////////

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // Parallel output stream.
  ConditionalOStream pcout;

  // Problem definition. ///////////////////////////////////////////////////////

  /**
   * The exponent Q
   */
  ExponentQ<double> exponent_Q;

  /**
   * The deformation gradient tensor
   */
  tensor F;

  /**
   * The green Lagrange stress tensor
   */
  tensor E;

  /**
   * The Piola Kirchhoff tensor
   */
  tensor PK;

  /**
   * The function pressure
   */
  FunctionPressure pressure;

  /**
   * The b_ij coefficients for the Piola Kiochhoff tensor
   */
  std::map<std::pair<int, int>, double> PK_b_weights;

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
