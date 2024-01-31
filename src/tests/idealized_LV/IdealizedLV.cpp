#include "IdealizedLV.hpp"

/**
 * @brief Setup the problem by loading the mesh, creating the finite element and
 * initialising the linear system.
 */
void IdealizedLV::setup() {
  // Create the mesh.
  {
    pcout << "Initializing the mesh" << std::endl;

    // First we read the mesh from file into a serial (i.e. not parallel)
    // triangulation.
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

    // Notice that we write here the number of *global* active cells (across all
    // processes).
    pcout << "  Number of elements = " << mesh.n_global_active_cells()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space. This is the same as in serial codes.
  {
    pcout << "Initializing the finite element space" << std::endl;

    // To construct a vector-valued finite element space, we use the FESystem
    // class. It is still derived from FiniteElement.
    FE_SimplexP<dim> fe_scalar(r);
    fe = std::make_unique<FESystem<dim>>(fe_scalar, dim);

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;

    quadrature_face = std::make_unique<QGaussSimplex<dim - 1>>(r + 1);

    pcout << "  Quadrature points per face = " << quadrature_face->size()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    // We retrieve the set of locally owned DoFs, which will be useful when
    // initializing linear algebra classes.
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    pcout << "  Initializing the sparsity pattern" << std::endl;

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
    pcout << "  Initializing the system matrix" << std::endl;
    jacobian_matrix.reinit(sparsity);

    // Finally, we initialize the right-hand side and solution vectors.
    pcout << "  Initializing the system right-hand side" << std::endl;
    residual_vector.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    pcout << "  Initializing the solution vector" << std::endl;
    solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
    delta_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  }
}

/**
 * @brief Assemble the system for a Newton iteration. The residual and Jacobian
 * matrix are evaluated leveraging automatic differentiation by Sacado:
 * https://www.dealii.org/current/doxygen/deal.II/classDifferentiation_1_1AD_1_1ResidualLinearization.html
 */
void IdealizedLV::assemble_system() {
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q = quadrature->size();

  FEValues<dim> fe_values(*fe, *quadrature,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  FEFaceValues<dim> fe_values_boundary(
      *fe, *quadrature_face,
      update_values | update_quadrature_points | update_JxW_values |
          update_normal_vectors);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  jacobian_matrix = 0.0;
  residual_vector = 0.0;

  FEValuesExtractors::Vector displacement(0);

  constexpr Differentiation::AD::NumberTypes ADTypeCode =
      Differentiation::AD::NumberTypes::sacado_dfad_dfad;

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (!cell->is_locally_owned()) continue;

    // Retrive the indipendent and dependent variables num
    // Setting them to dofs_per_cell as
    // https://www.dealii.org/current/doxygen/deal.II/classDifferentiation_1_1AD_1_1ResidualLinearization.html
    const uint32_t n_independent_vars = dofs_per_cell;
    const uint32_t n_dependent_vars = dofs_per_cell;

    // Retirive the dof indices
    cell->get_dof_indices(dof_indices);

    // Create some aliases for the AD helper.
    using ADHelper =
        Differentiation::AD::ResidualLinearization<ADTypeCode, double>;
    using ADNumberType = typename ADHelper::ad_type;

    // Create and initialize an instance of the helper class.
    ADHelper ad_helper(n_independent_vars, n_dependent_vars);

    // Reinit finite element
    fe_values.reinit(cell);
    // Reinit local data structures
    cell_matrix = 0.0;
    cell_rhs = 0.0;

    // ==================================================
    // =               AD Recording Phase               =
    // ==================================================
    {
      // First, we set the values for all DoFs.
      ad_helper.register_dof_values(solution, dof_indices);

      // Then we get the complete set of degree of freedom values as
      // represented by auto-differentiable numbers.
      const std::vector<ADNumberType> dof_values_ad =
          ad_helper.get_sensitive_dof_values();

      // Problem specific task, compute values and gradients
      std::vector<Tensor<2, dim, ADNumberType>> solution_gradient_loc(
          n_q, Tensor<2, dim, ADNumberType>());
      std::vector<Tensor<1, dim, ADNumberType>> solution_loc(
          n_q, Tensor<1, dim, ADNumberType>());
      fe_values[displacement].get_function_gradients_from_local_dof_values(
          dof_values_ad, solution_gradient_loc);
      fe_values[displacement].get_function_values_from_local_dof_values(
          dof_values_ad, solution_loc);

      // This variable stores the cell residual vector contributions.
      // Good practise is to initialise it to zero.
      std::vector<ADNumberType> residual_ad(n_dependent_vars,
                                            ADNumberType(0.0));

      // Loop over quadrature points
      for (unsigned int q = 0; q < n_q; ++q) {
        // Compute deformation gradient tensor
        auto F = Physics::Elasticity::Kinematics::F(solution_gradient_loc[q]);
        // Compute green Lagrange tensor
        auto E = Physics::Elasticity::Kinematics::E(F);
        // Compute exponent Q
        ExponentQ<ADNumberType> exponent_q;
        exponent_q.compute(E);
        // Compute Piola Kirchhoff tensor
        Tensor<2, dim, ADNumberType> piola_kirchhoff;
        for (uint32_t i = 0; i < dim; ++i) {
          for (uint32_t j = 0; j < dim; ++j) {
            piola_kirchhoff[i][j] = Material<double>::default_C *
                                    piola_kirchhoff_b_weights[{i, j}] *
                                    E[i][j] * std::exp(exponent_q.get_q());
          }
        }
        // Compute the integration weight
        const auto quadrature_integration_w = fe_values.JxW(q);

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          // Compose -F:
          // It is give by (L(q) + B_N(q)). This piece compose L(q).
          residual_ad[i] +=
              scalar_product(piola_kirchhoff,
                             fe_values[displacement].gradient(i, q)) *
              quadrature_integration_w;  // L(d)
        }
      }
      // Loop over quadrature points for Neumann boundaries conditions
      if (cell->at_boundary()) {
        for (uint32_t face_number = 0; face_number < cell->n_faces();
             ++face_number) {
          if (cell->face(face_number)->at_boundary() &&
              is_face_at_newmann_boundary(
                  cell->face(face_number)->boundary_id())) {
            fe_values_boundary.reinit(cell, face_number);

            for (unsigned int q = 0; q < n_q; ++q) {
              // Compute deformation gradient tensor
              // TODO: maybe cache this (is 3x3 for now)
              auto F =
                  Physics::Elasticity::Kinematics::F(solution_gradient_loc[q]);
              // Compute determinant of F
              auto det_F = determinant(F);
              // Compute F^T
              auto F_T = transpose(F);
              // Compute (F^T)^{-1}
              auto F_T_inverse = invert(F_T);
              // Compute H_h (tensor)
              auto H_h = det_F * F_T_inverse;

              for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                // Compose B_N(q)
                residual_ad[i] +=
                    pressure.value(fe_values_boundary.quadrature_point(q)) *
                    scalar_product(
                        H_h * fe_values_boundary.normal_vector(q),
                        fe_values_boundary[displacement].value(i, q)) *
                    fe_values_boundary.JxW(q);
              }
            }
          }
        }
      }
      // Register the residual AD
      ad_helper.register_residual_vector(residual_ad);
      // Compute the residual
      ad_helper.compute_residual(cell_rhs);
      // We need -R(displacement)(test_function) at Newton rhs
      cell_rhs *= -1.0;
      // Compute the local Jacobian
      ad_helper.compute_linearization(cell_matrix);
      // Add to global matrix and vector
      jacobian_matrix.add(dof_indices, cell_matrix);
      residual_vector.add(dof_indices, cell_rhs);
    }
  }
  // Share between MPI processes
  jacobian_matrix.compress(VectorOperation::add);
  residual_vector.compress(VectorOperation::add);

  // Boundary conditions.
  {
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(
        dof_handler, dirichlet_boundary_functions, boundary_values);
    MatrixTools::apply_boundary_values(boundary_values, jacobian_matrix,
                                       delta_owned, residual_vector, true);
  }
}

/**
 * @brief Solve the linear system using GMRES
 */
void IdealizedLV::solve_system() {
  SolverControl solver_control(1000, 1e-6 * residual_vector.l2_norm());

  SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control);
  TrilinosWrappers::PreconditionSSOR preconditioner;
  preconditioner.initialize(
      jacobian_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

  solver.solve(jacobian_matrix, delta_owned, residual_vector, preconditioner);
  pcout << "   " << solver_control.last_step() << " GMRES iterations"
        << std::endl;
}

/**
 * @brief Solve the non linear problem using the Newton method
 */
void IdealizedLV::solve_newton() {
  pcout << "===============================================" << std::endl;

  const unsigned int n_max_iters = 1000;
  const double residual_tolerance = 1e-4;

  unsigned int n_iter = 0;
  double residual_norm = residual_tolerance + 1;

  while (n_iter < n_max_iters && residual_norm > residual_tolerance) {
    assemble_system();
    residual_norm = residual_vector.l2_norm();

    pcout << "Newton iteration " << n_iter << "/" << n_max_iters
          << " - ||r|| = " << std::scientific << std::setprecision(6)
          << residual_norm << std::flush;

    // We actually solve the system only if the residual is larger than the
    // tolerance.
    if (residual_norm > residual_tolerance) {
      solve_system();

      solution_owned += delta_owned;
      solution = solution_owned;
    } else {
      pcout << " < tolerance" << std::endl;
    }

    ++n_iter;
  }

  pcout << "===============================================" << std::endl;
}

/**
 * @brief Write the output
 */
void IdealizedLV::output() const {
  DataOut<dim> data_out;

  // By passing these two additional arguments to add_data_vector, we specify
  // that the three components of the solution are actually the three components
  // of a vector, so that the visualization program can take that into account.
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
          dim, DataComponentInterpretation::component_is_part_of_vector);
  std::vector<std::string> solution_names(dim, "u");

  data_out.add_data_vector(dof_handler, solution, solution_names,
                           data_component_interpretation);

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  const std::string output_file_name = "slab-cubic";
  data_out.write_vtu_with_pvtu_record("./", output_file_name, 0,
                                      MPI_COMM_WORLD);

  pcout << "Output written to " << output_file_name << "." << std::endl;

  pcout << "===============================================" << std::endl;
}
