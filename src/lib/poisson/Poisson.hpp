/**
 * @file Poisson.hpp
 * @brief Header file defining the Poisson problem solver.
 */

#ifndef POISSON_HPP
#define POISSON_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>

using namespace dealii;

/**
 * @class Poisson
 * @brief Class defining a solver for the Poisson problem. Note this problem has
 * only the diffusion part active with const value of 1.0.
 * @tparam dim The problem dimension
 * @tparam Scalar The scalar type for the problem, by default a double
 */
template <int dim, typename Scalar = double> class Poisson {
public:
  /**
   * @brief Class representing the forcing term component
   */
  class ForcingTerm : public Function<dim> {
  public:
    /**
     * @brief Default constructor
     */
    ForcingTerm() {}
    /**
     * @brief Evaluate the forcing term at a given point
     * @param p The evaluation point
     * @param component The component id
     */
    virtual Scalar value(const Point<dim> & /*p*/,
                         const unsigned int /*component*/ = 0) const override {
      return static_cast<Scalar>(0);
    }
  };
  /**
   * @brief Constructor
   * @param mesh_file_name_ The mesh file name
   * @param r_ The polynomial degree
   */
  Poisson(const std::string &mesh_file_name_, const unsigned int &r_)
      : mesh_file_name(mesh_file_name_), r(r_),
        mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
        mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)),
        endocardium_function(0), epicardium_function(1), mesh(MPI_COMM_WORLD),
        pcout(std::cout, mpi_rank == 0) {
    dirichlet_boundary_functions[20] = &endocardium_function;
    dirichlet_boundary_functions[10] = &epicardium_function;
  }

  void setup();

  void assemble();

  void solve();

  const TrilinosWrappers::MPI::Vector &get_solution() const;

protected:
  /**
   * The input mesh file name
   */
  const std::string mesh_file_name;
  /**
   * Polynomial degree
   */
  const unsigned int r;
  /**
   * MPI size
   */
  const unsigned int mpi_size;
  /**
   * MPI rank
   */
  const unsigned int mpi_rank;
  /**
   * Forcing term object
   */
  ForcingTerm forcing_term;
  /**
   * Endocardium constant Dirichlet boundary function
   */
  Functions::ConstantFunction<dim> endocardium_function;
  /**
   * Epicardium constant Dirichlet boundary function
   */
  Functions::ConstantFunction<dim> epicardium_function;
  /**
   * A map of Dirichlet boundary faces
   */
  std::map<types::boundary_id, const Function<dim> *>
      dirichlet_boundary_functions;
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
   * The degree of freedom handler
   */
  DoFHandler<dim> dof_handler;
  /**
   * System matrix
   */
  TrilinosWrappers::SparseMatrix system_matrix;
  /**
   * System rhs vector
   */
  TrilinosWrappers::MPI::Vector system_rhs;
  /**
   * System solution vector
   */
  TrilinosWrappers::MPI::Vector solution;
  /**
   * System solution ghost vector
   */
  TrilinosWrappers::MPI::Vector solution_ghost;
  /**
   * MPI conditional ostream
   */
  ConditionalOStream pcout;
  /**
   * DoFs owned by current process
   */
  IndexSet locally_owned_dofs;
};

#endif
