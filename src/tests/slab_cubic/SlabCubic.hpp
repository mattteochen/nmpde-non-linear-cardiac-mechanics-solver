#ifndef SLAB_CUBIC_HPP
#define SLAB_CUBIC_HPP

#include <cmath>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_in.h>

#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

#include "../../utils/Utils.hpp"
#include "../../utils/Assert.hpp"

using namespace dealii;
using ui = uint32_t;

/**
 * Class representing the Slab Cubic solver
 */
class SlabCubic
{
public:
  /**
   * Const physical dimension
   */
  static constexpr unsigned int dim = 3;

  /**
   * A class representing the deformation gradient tensor
   */
  class DeformationGradientTensor : public Function<dim>
  {
    using T = Tensor<2, dim>;
  protected:
    /**
     * A dealii Tensor<2, dim> representing the deformation gradient tensor
     */
    T deformation_gradient_tensor;
  public:
    /**
     * Default constructor
     */
    DeformationGradientTensor() = default;

    /**
     * Retrieve the deformation gradient tensor, const version
     * @return A dealii Tensor<2, dim> representing the deformation gradient tensor
     */
    const T& get_tensor() const {
      return deformation_gradient_tensor;
    }

    /**
     * Retrieve the deformation gradient tensor
     * @return A dealii Tensor<2, dim> representing the deformation gradient tensor
     */
    T& get_tensor() {
      return deformation_gradient_tensor;
    }
  
    /**
     * Evaluate the deformation gradient tensor at a specific point x;
     * @param x A dealii Tensor
     * @return The deformation gradient tensor
     */
    T& value(const T& x, const unsigned int /*component*/ = 0) {
      deformation_gradient_tensor.clear();
      for (ui i=0; i<dim; ++i) {
        for (ui j=0; j<dim; ++j) {
          deformation_gradient_tensor[i][j] = -x[i][j];
        }
      }
      return deformation_gradient_tensor;
    }
  };

  /**
   * A class representing the green Lagrange strain tensor
   */
  class GreenStrainTensor : public Function<dim>
  {
    using T = Tensor<2, dim>;
  protected:
    /**
     * A dealii Tensor<2, dim> representing the green Lagrange strain tensor
     */
    T green_lagrange_strain_tensor;
  public:
    /**
     * Default constructor
     */
    GreenStrainTensor() = default;

    /**
     * Retrieve the green Lagrange strain tensor, const version
     * @return A dealii Tensor<2, dim> representing the green Lagrange strain tensor
     */
    const T& get_tensor() const {
      return green_lagrange_strain_tensor;
    }

    /**
     * Retrieve the green Lagrange strain tensor
     * @return A dealii Tensor<2, dim> representing the green Lagrange strain tensor
     */
    T& get_tensor() {
      return green_lagrange_strain_tensor;
    }
  
    /**
     * Evaluate the green Lagrange strain tensor at a specific deformation gradient tensor;
     * @param x A DeformationGradientTensor
     * @return The green Lagrange strain tensor
     */
     T& value(const DeformationGradientTensor& dgt, const unsigned int /*component*/ = 0) {
      return green_lagrange_strain_tensor =
                 0.5 * (Utils::dealii::Tensor::get_transpose<2, dim>(dgt.get_tensor()) * dgt.get_tensor() -
                        Utils::dealii::Tensor::get_identity<2, dim>());
    }
  };

  /**
   * A class representing W's exponent Q
   */
  template<typename Scalar>
  class ExponentQ : Function<dim>
  {
  protected:
    static constexpr Scalar b_f = static_cast<Scalar>(8);
    static constexpr Scalar b_t = static_cast<Scalar>(2);
    static constexpr Scalar b_fs = static_cast<Scalar>(4);

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
    void compute(const GreenStrainTensor& gst) const {
      auto& gst_tensor = gst.get_tensor();
      q = b_f * gst_tensor[0][0] * gst_tensor[0][0] +
             b_t * (gst_tensor[1][1] * gst_tensor[1][1] + gst_tensor[2][2] * gst_tensor[2][2] +
                    gst_tensor[1][2] * gst_tensor[1][2] + gst_tensor[2][1] * gst_tensor[2][1]) +
             b_fs * (gst_tensor[0][1] * gst_tensor[0][1] + gst_tensor[1][0] * gst_tensor[1][0] +
                     gst_tensor[0][2] * gst_tensor[0][2] + gst_tensor[2][0] * gst_tensor[2][0]);
      initialised = 1;
    }
  };

  /**
   * A class representing the parameter W of Piola Kirchhof 
   */
  template<typename Scalar>
  class PKW : Function<dim>
  {
  protected:
    static constexpr Scalar default_C = static_cast<Scalar>(2000);
    Scalar C = default_C;
    Scalar w = static_cast<Scalar>(0);
  public:
    /**
     * Retrieve the W value
     * @return The W value
     */
    Scalar get_w() {
      return w;
    }

    void set_C(Scalar value) {
      C = value;
    }

    /**
     * Evaluate the W value with given Q exponent
     * @param q An exponent q evaluated at a GreenStrainTensor
     * @return The parameter W
     */
    Scalar value(const ExponentQ<Scalar>& q) {
      return w = C * 0.5 * (std::exp(q.get_q() - static_cast<Scalar>(1)));
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
  {}

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
   * The deformation gradient tensor object
   */
  DeformationGradientTensor dgt;

  /**
   * The green Lagrange strain tensor
   */
  GreenStrainTensor gst;

  /**
   * The exponent Q
   */
  ExponentQ<double> exponent_Q;

  /**
   * The W param of Piola Kirchhof stress tensor
   */
  PKW<double> w;

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
