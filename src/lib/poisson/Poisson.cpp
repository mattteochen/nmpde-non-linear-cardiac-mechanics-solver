/**
 * @file Poisson.cpp
 * @brief Implementation file for the Poisson class.
 */

#include <fstream>
#include <poisson/Poisson.hpp>

/**
 * @brief Setup the problem by loading the mesh, creating the finite element
 * and initialising the linear system.
 * @tparam dim The problem dimension space
 * @tparam Scalar The scalar type being used
 */
template <int dim, typename Scalar> void Poisson<dim, Scalar>::setup() {
  // Initialize the mesh
  {
    Triangulation<dim> mesh_serial;

    {
      GridIn<dim> grid_in;
      grid_in.attach_triangulation(mesh_serial);

      std::ifstream grid_in_file(mesh_file_name);
      grid_in.read_msh(grid_in_file);
    }

    // Then, we copy the triangulation into the parallel one.
    {
      GridTools::partition_triangulation(mpi_size, mesh_serial);
      const auto construction_data = TriangulationDescription::Utilities::
          create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
      mesh.create_triangulation(construction_data);
    }
  }

  // Initialize the finite element space. This is the same as in serial codes.
  {
    fe = std::make_unique<FE_SimplexP<dim>>(r);
    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);
  }

  // Initialize the DoF handler.
  {
    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    // We retrieve the set of locally owned DoFs, which will be useful when
    // initializing linear algebra classes.
    locally_owned_dofs = dof_handler.locally_owned_dofs();
  }

  pcout << "Poisson solver" << std::endl;
  pcout << "  Degree                     = " << fe->degree << std::endl;
  pcout << "  DoFs per cell              = " << fe->dofs_per_cell << std::endl;
  pcout << "  Quadrature points per cell = " << quadrature->size() << std::endl;
  pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  pcout << "===============================================" << std::endl;

  // Initialize the linear system.
  {
    // To initialize the sparsity pattern, we use Trilinos' class, that manages
    // some of the inter-process communication.
    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);

    // After initialization, we need to call compress, so that all process
    // retrieve the information they need for the rows they own (i.e. the rows
    // corresponding to locally owned DoFs).
    sparsity.compress();

    // Then, we use the sparsity pattern to initialize the system matrix. Since
    // the sparsity pattern is partitioned by row, so will the matrix.
    system_matrix.reinit(sparsity);

    // Finally, we initialize the right-hand side and solution vectors.
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  }
}

/**
 * @brief Assemble the linear system
 * @tparam dim The problem dimension space
 * @tparam Scalar The scalar type being used
 */
template <int dim, typename Scalar> void Poisson<dim, Scalar>::assemble() {
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q = quadrature->size();

  FEValues<dim> fe_values(*fe, *quadrature,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  system_matrix = 0.0;
  system_rhs = 0.0;

  unsigned cell_counter = 0;
  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (!cell->is_locally_owned())
      continue;
    cell_counter++;

    fe_values.reinit(cell);

    cell_matrix = 0.0;
    cell_rhs = 0.0;

    for (unsigned int q = 0; q < n_q; ++q) {
      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        for (unsigned int j = 0; j < dofs_per_cell; ++j) {
          cell_matrix(i, j) += fe_values.shape_grad(i, q) *
                               fe_values.shape_grad(j, q) * fe_values.JxW(q);
        }
        cell_rhs(i) += forcing_term.value(fe_values.quadrature_point(q)) *
                       fe_values.shape_value(i, q) * fe_values.JxW(q);
      }
    }

    cell->get_dof_indices(dof_indices);
    // as this is an utility solver, save the global dof indices reference to be
    // communicated to the caller solver
    aggregate_dof_indices.push_back(dof_indices);

    system_matrix.add(dof_indices, cell_matrix);
    system_rhs.add(dof_indices, cell_rhs);
  }
  pcout << "Poisson assemble" << std::endl;
  std::cout << "  Poisson aggregate dof indices cache size (processor: "
            << mpi_rank << ") = " << aggregate_dof_indices.size() << std::endl;
  std::cout << "  Poisson cell_counter (processor: " << mpi_rank
            << ") = " << cell_counter << std::endl;

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

  // Dirichlet conditions.
  {
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(
        dof_handler, dirichlet_boundary_functions, boundary_values);
    // setting the flag to false as for
    // https://www.dealii.org/current/doxygen/deal.II/namespaceMatrixTools.html#a967ecdb0d0efe1549be8e3f6b9bbf123
    MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution,
                                       system_rhs, false);
  }
}

/**
 * @brief Solve the linear system
 * @tparam dim The problem dimension space
 * @tparam Scalar The scalar type being used
 */
template <int dim, typename Scalar> void Poisson<dim, Scalar>::solve() {
  {
    SolverControl solver_control(1000, 1e-6 * system_rhs.l2_norm());

    SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

    TrilinosWrappers::PreconditionSSOR preconditioner;
    preconditioner.initialize(
        system_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

    pcout << "-----------------------------------------------" << std::endl;
    pcout << "Solving the Poisson problem" << std::endl;
    solver.solve(system_matrix, solution, system_rhs, preconditioner);
    pcout << "  " << solver_control.last_step() << " CG iterations"
          << std::endl;
  }
  {
    // To correctly export the solution, each process needs to know the solution
    // DoFs it owns, and the ones corresponding to elements adjacent to the ones
    // it owns (the locally relevant DoFs, or ghosts). We create a vector to
    // store them.
    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    solution_ghost.reinit(locally_owned_dofs, locally_relevant_dofs,
                          MPI_COMM_WORLD);

    // This performs the necessary communication so that the locally relevant
    // DoFs are received from other processes and stored inside solution_ghost.
    solution_ghost = solution;
    pcout << "  Solution size: " << solution_ghost.size() << std::endl;
  }
}

/**
 * @brief Retrieve a reference to the solution
 * @tparam dim The problem dimension space
 * @tparam Scalar The scalar type being used
 */
template <int dim, typename Scalar>
const TrilinosWrappers::MPI::Vector &
Poisson<dim, Scalar>::get_solution() const {
  return solution_ghost;
}

/**
 * @brief Retrieve a reference to the cached dof indices
 * @tparam dim The problem dimension space
 * @tparam Scalar The scalar type being used
 */
template <int dim, typename Scalar>
const std::vector<std::vector<types::global_dof_index>> &
Poisson<dim, Scalar>::get_aggregate_dof_indices() const {
  return aggregate_dof_indices;
}

// Explicit template initializations
template class Poisson<3, double>;
