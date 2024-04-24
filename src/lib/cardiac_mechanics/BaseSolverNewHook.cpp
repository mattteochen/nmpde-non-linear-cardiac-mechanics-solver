/**
 * @file BaseSolverNewHook.cpp
 * @brief Implementation file for the base solver class.
 */

#include "Assert.hpp"
#include <cardiac_mechanics/BaseSolverNewHook.hpp>
#include <cmath>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <fstream>

/**
 * @class BaseSolverNewHook
 * @brief Setup the problem by loading the mesh, creating the finite element
 * and initialising the linear system.
 * @tparam dim The problem dimension space
 * @tparam Scalar The scalar type being used
 */
template <int dim, typename Scalar>
void BaseSolverNewHook<dim, Scalar>::setup() {
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
    switch (triangulation_type) {
    case TriangulationType::T: {
      pcout << "  Using triangulation: T" << std::endl;
      fe = std::make_unique<FESystem<dim>>(FE_SimplexP<dim>(r), dim);
      quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);
      quadrature_face = std::make_unique<QGaussSimplex<dim - 1>>(r + 1);
      break;
    };
    case TriangulationType::Q: {
      pcout << "  Using triangulation: Q" << std::endl;
      fe = std::make_unique<FESystem<dim>>(FE_Q<dim>(r), dim);
      quadrature = std::make_unique<QGauss<dim>>(r + 1);
      quadrature_face = std::make_unique<QGauss<dim - 1>>(r + 1);
      break;
    };
    };

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;
    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;
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
 * @brief Assemble the Piola Kirchhoff tensor
 * @tparam dim The problem dimension space
 * @tparam Scalar The scalar type being used
 * @param out_tensor A reference to the output tensor
 * @param solution_gradient_quadrature The current solution gradient at given
 * quadrature node
 * @param cell_index The index of the current dealii cell
 */
template <int dim, typename Scalar>
void BaseSolverNewHook<dim, Scalar>::compute_piola_kirchhoff(
    Tensor<2, dim, ADNumberType> &out_tensor,
    const Tensor<2, dim, ADNumberType> &solution_gradient_quadrature,
    const unsigned /*cell_index*/) {
  // Compute deformation gradient tensor
  const Tensor<2, dim, ADNumberType> F =
      Physics::Elasticity::Kinematics::F(solution_gradient_quadrature);
  const Tensor<2, dim, ADNumberType> F_inverse = dealii::invert(F);
  const ADNumberType F_det = dealii::determinant(F);
  // #ifdef BUILD_TYPE_DEBUG
  ASSERT(F_det > ADNumberType(0.0), "Negative F determinant");
  // #endif
  Tensor<2, dim, ADNumberType> I;
  for (uint32_t i = 0; i < dim; ++i) {
    I[i][i] = ADNumberType(1.0);
  }

  for (uint32_t i = 0; i < dim; ++i) {
    for (uint32_t j = 0; j < dim; ++j) {
      out_tensor[i][j] =
          ((ADNumberType(Material::mu) * (I[i][j] - F_inverse[i][j]))) +
          (ADNumberType(Material::lambda) * Sacado::Fad::log(F_det) *
           F_inverse[i][j]);
    }
  }
#ifdef BUILD_TYPE_DEBUG
  for (unsigned row = 0; row < dim; row++) {
    const auto &PK_i = out_tensor[row];
    for (unsigned col = 0; col < dim; col++) {
      const double scalar = PK_i[col].val();
      ASSERT(dealii::numbers::is_finite(scalar),
             "rank = " << mpi_rank << " PK NaN: " << scalar << std::endl);
    }
  }
#endif
}

/**
 * @brief Assemble the system for a Newton iteration. The residual vector and
 * Jacobian matrix are evaluated leveraging automatic differentiation by Sacado:
 * https://www.dealii.org/current/doxygen/deal.II/classDifferentiation_1_1AD_1_1ResidualLinearization.html
 * @tparam dim The problem dimension space
 * @tparam Scalar The scalar type being used
 */
template <int dim, typename Scalar>
void BaseSolverNewHook<dim, Scalar>::assemble_system() {
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q = quadrature->size();
  const unsigned int n_face_q = quadrature_face->size();

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
  unsigned int cell_index = 0;

  FEValuesExtractors::Vector displacement(0);

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (!cell->is_locally_owned()) {
      continue;
    }

    // Retrive the indipendent and dependent variables num
    // Setting them to dofs_per_cell as
    // https://www.dealii.org/current/doxygen/deal.II/classDifferentiation_1_1AD_1_1ResidualLinearization.html
    const uint32_t n_independent_vars = dofs_per_cell;
    const uint32_t n_dependent_vars = dofs_per_cell;

    // Retirive the dof indices
    cell->get_dof_indices(dof_indices);

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
        // compute the piola kirchhoff tensor
        Tensor<2, dim, ADNumberType> piola_kirchhoff;
        compute_piola_kirchhoff(piola_kirchhoff, solution_gradient_loc[q],
                                cell_index);

        // Compute the integration weight
        const auto quadrature_integration_w = fe_values.JxW(q);

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          // Compose -F:
          // It is give by (L(q) + B_N(q)). This piece compose L(q).
          residual_ad[i] +=
              scalar_product(piola_kirchhoff,
                             fe_values[displacement].gradient(i, q)) *
              quadrature_integration_w; // L(q)
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

            // Loop over face quadrature points
            for (unsigned int q = 0; q < n_face_q; ++q) {
              // Compute deformation gradient tensor
              // TODO: maybe cache this (is 3x3 for now)
              const auto F =
                  Physics::Elasticity::Kinematics::F(solution_gradient_loc[q]);
              // Compute determinant of F
              const auto det_F = determinant(F);
              // Compute (F^T)^{-1}
              const auto F_T_inverse = invert(transpose(F));
              // Compute H_h (tensor)
              const auto H_h = det_F * F_T_inverse;

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
    // we only count local owned cells
    cell_index++;
  }
  // Share between MPI processes
  jacobian_matrix.compress(VectorOperation::add);
  residual_vector.compress(VectorOperation::add);

  // Boundary conditions.
  {
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(
        dof_handler, dirichlet_boundary_functions, boundary_values);
    // setting the flag to false as for
    // https://www.dealii.org/current/doxygen/deal.II/namespaceMatrixTools.html#a967ecdb0d0efe1549be8e3f6b9bbf123
    MatrixTools::apply_boundary_values(boundary_values, jacobian_matrix,
                                       delta_owned, residual_vector, false);
  }
}

/**
 * @brief Solve the linear system
 * @tparam dim The problem dimension space
 * @tparam Scalar The scalar type being used
 */
template <int dim, typename Scalar>
void BaseSolverNewHook<dim, Scalar>::solve_system() {
  auto solver_control = linear_solver_utility.get_initialized_solver_control(
      jacobian_matrix.m(), residual_vector.l2_norm());

  Preconditioner preconditioner;
  LinearSolver solver;

  linear_solver_utility.initialize_solver(solver, solver_control);
  linear_solver_utility.initialize_preconditioner(preconditioner,
                                                  jacobian_matrix);
  linear_solver_utility.solve(solver, jacobian_matrix, delta_owned,
                              residual_vector, preconditioner);
  pcout << "   " << solver_control.last_step() << " "
        << LinearSolverUtility<Scalar>::solver_type_matcher_rev
               [linear_solver_utility.get_solver_type()]
        << " iterations" << std::endl;
}

/**
 * @brief Solve the non linear problem using the Newton method
 * @tparam dim The problem dimension space
 * @tparam Scalar The scalar type being used
 */
template <int dim, typename Scalar>
void BaseSolverNewHook<dim, Scalar>::solve_newton() {
  pcout << "===============================================" << std::endl;

  unsigned int n_iter = 0;
  double residual_norm = newton_solver_utility.get_tolerance() + 1;

  {
    const std::chrono::high_resolution_clock::time_point begin_time =
        std::chrono::high_resolution_clock::now();
    while (n_iter < newton_solver_utility.get_max_iterations() &&
           residual_norm > newton_solver_utility.get_tolerance()) {
      assemble_system();

      pcout << "Newton iteration " << n_iter << "/"
            << newton_solver_utility.get_max_iterations()
            << " - ||r|| = " << std::scientific << std::setprecision(6)
            << residual_norm << std::flush;

      // We actually solve the system only if the residual is larger than the
      // tolerance.
      if (residual_norm > newton_solver_utility.get_tolerance()) {
        solve_system();

        residual_norm = delta_owned.l2_norm();

        solution_owned += delta_owned;
        solution = solution_owned;
      } else {
        pcout << " < tolerance" << std::endl;
      }

      ++n_iter;
    }
    const std::chrono::high_resolution_clock::time_point end_time =
        std::chrono::high_resolution_clock::now();
    const long long diff =
        std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                              begin_time)
            .count();
    pcout << std::endl
          << "Newton metod ended in " << diff << " us" << std::endl;

    if (mpi_rank == 0) {
      const std::string report_name =
          LinearSolverUtility<Scalar>::solver_type_matcher_rev
              [linear_solver_utility.get_solver_type()] +
          "_" +
          LinearSolverUtility<Scalar>::preconditioner_type_matcher_rev
              [linear_solver_utility.get_preconditioner_type()];
      Reporter reporter(report_name + ".log",
                        "TYPE,TIME(us),NEWTON_ITERATIONS,NEWTON_RESIDUAL_NORM",
                        report_name);
      reporter.write(diff, ',', n_iter, ',', residual_norm);
    }
  }

  pcout << "===============================================" << std::endl;
}

/**
 * @brief Write the output
 * @tparam dim The problem dimension space
 * @tparam Scalar The scalar type being used
 */
template <int dim, typename Scalar>
void BaseSolverNewHook<dim, Scalar>::output() const {
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

  const std::string output_file_name = problem_name;
  data_out.write_vtu_with_pvtu_record("./", output_file_name, 0,
                                      MPI_COMM_WORLD);

  pcout << "Output written to " << output_file_name << "." << std::endl;

  pcout << "===============================================" << std::endl;
}

/**
 * @brief Parse the parameter input
 * @tparam dim The problem dimension space
 * @tparam Scalar The scalar type being used
 * @param parameters_file_name_ The input configuration parameter file name
 */
template <int dim, typename Scalar>
void BaseSolverNewHook<dim, Scalar>::parse_parameters(
    const std::string &parameters_file_name_) {
  // Add the linear solver subsection
  prm.enter_subsection("TriangulationType");
  {
    prm.declare_entry("Type", "T", Patterns::Selection("T|Q"),
                      "Triangulation cell type");
  }
  prm.leave_subsection();

  prm.enter_subsection("LinearSolver");
  {
    prm.declare_entry("SolverType", "GMRES",
                      Patterns::Selection("GMRES|BiCGSTAB"),
                      "Type of solver used to solve the linear system");

    prm.declare_entry("Residual", "1e-6", Patterns::Double(1e-10, 1e-3),
                      "Linear solver tolerance");

    prm.declare_entry("MaxIteration", "1.0", Patterns::Double(1.0),
                      "Linear solver max iterations multiplier");

    prm.declare_entry("PreconditionerType", "ILU",
                      Patterns::Selection("IDENTITY|SSOR|ILU|SOR"),
                      "Type of preconditioner");
  }
  prm.leave_subsection();

  // Add the polynomial degree subsection
  prm.enter_subsection("PolynomialDegree");
  {
    prm.declare_entry("Degree", "1", Patterns::Integer(1, 2),
                      "Degree of the polynomial for finite elements");
  }
  prm.leave_subsection();

  // Add the Newton method solver subsection
  prm.enter_subsection("NewtonMethod");
  {
    prm.declare_entry("Residual", "1e-6", Patterns::Double(1e-10, 1e-3),
                      "Newton solver tolerance");

    prm.declare_entry("MaxIterations", "100", Patterns::Integer(10),
                      "Newton solver max iterations");
  }
  prm.leave_subsection();

  // Add the material properties subsection
  prm.enter_subsection("Material");
  {
    prm.declare_entry("mu", "1.0", Patterns::Double(1.0),
                      "mu coefficient of the New Hook material");

    prm.declare_entry("lambda", "1.0", Patterns::Double(1.0),
                      "lambda coefficient of the New Hook material");
  }
  prm.leave_subsection();

  // Add the boundaries subsection
  prm.enter_subsection("Boundaries");
  {
    prm.declare_entry("Dirichlet", "", Patterns::Anything(),
                      "Dirichlet boundaries tags");

    prm.declare_entry("Newmann", "", Patterns::Anything(),
                      "Newmann boundaries tags");
  }
  prm.leave_subsection();

  // Add the pressure subsection
  prm.enter_subsection("Pressure");
  {
    prm.declare_entry("Value", "0.0", Patterns::Double(0.0),
                      "Boundary pressure value (Pa)");
    prm.declare_entry("FiberValue", "0.0", Patterns::Double(0.0),
                      "Fiber pressure value (Pa)");
  }
  prm.leave_subsection();

  // Read input file
  prm.parse_input(parameters_file_name_);
  // prm.print_parameters(std::cout, ParameterHandler::OutputStyle::JSON);

  // Parse the triangulation type
  prm.enter_subsection("TriangulationType");
  {
    triangulation_type =
        prm.get("Type") == "T" ? TriangulationType::T : TriangulationType::Q;
  }
  prm.leave_subsection();

  // Parse linear solver subsection to the configuration object
  prm.enter_subsection("LinearSolver");
  {
    linear_solver_utility = LinearSolverUtility<Scalar>(
        LinearSolverUtility<Scalar>::solver_type_matcher[prm.get("SolverType")],
        LinearSolverUtility<Scalar>::preconditioner_type_matcher[prm.get(
            "PreconditionerType")],
        prm.get_double("Residual"), prm.get_double("MaxIteration"));
  }
  prm.leave_subsection();
  pcout << "===============================================" << std::endl;
  pcout << "Linear solver configuration:" << std::endl;
  pcout << linear_solver_utility;

  // Parse polynomial degree to class member
  prm.enter_subsection("PolynomialDegree");
  { r = prm.get_integer("Degree"); }
  prm.leave_subsection();
  pcout << "-----------------------------------------------" << std::endl;
  pcout << "Polynomial configuration:" << std::endl;
  pcout << "  Degree: " << r << std::endl;
  pcout << "-----------------------------------------------" << std::endl;

  // Parse Newton method subsection to the configuration object
  prm.enter_subsection("NewtonMethod");
  {
    newton_solver_utility = NewtonSolverUtility<Scalar>(
        prm.get_double("Residual"), prm.get_integer("MaxIterations"));
  }
  prm.leave_subsection();
  pcout << "Newton method configuration:" << std::endl;
  pcout << newton_solver_utility;
  pcout << "-----------------------------------------------" << std::endl;

  // Parse material subsection to the configuration struct
  prm.enter_subsection("Material");
  Material::mu = prm.get_double("mu");
  Material::lambda = prm.get_double("lambda");
  prm.leave_subsection();
  pcout << "Material configuration:" << std::endl;
  pcout << Material::show();
  pcout << "-----------------------------------------------" << std::endl;

  // Parse the boundary tags in the utility object
  prm.enter_subsection("Boundaries");
  {
    boundaries_utility =
        BoundariesUtility(prm.get("Dirichlet"), prm.get("Newmann"));
  }
  prm.leave_subsection();
  pcout << "Boundaries configuration:" << std::endl;
  pcout << boundaries_utility;
  pcout << "===============================================" << std::endl;

  // Parse the pressure in the pressure function object
  prm.enter_subsection("Pressure");
  { pressure = ConstantPressureFunction(prm.get_double("Value")); }
  prm.leave_subsection();
}

// Explicit template initializations
template class BaseSolverNewHook<3, double>;
