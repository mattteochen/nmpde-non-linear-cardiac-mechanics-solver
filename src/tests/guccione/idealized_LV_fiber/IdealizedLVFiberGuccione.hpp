/**
 * @file IdealizedLVFiberGuccione.hpp
 * @brief Header file defining the idealized lv solver class.
 */

#ifndef IDEALIZED_LV_FIBER_GUCCIONE_HPP
#define IDEALIZED_LV_FIBER_GUCCIONE_HPP

#include <cardiac_mechanics/BaseSolverGuccione.hpp>
#include <poisson/Poisson.hpp>

#include <deal.II/base/numbers.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/lac/trilinos_solver.h>

#include <cmath>
#include <memory>
#ifdef BUILD_TYPE_DEBUG
#include <fstream>
#include <string>
#endif

/**
 * @class IdealizedLVFiberGuccione
 * @brief Class representing the an Idealized LV solver with fiber contraction
 * (https://pubmed.ncbi.nlm.nih.gov/26807042/)
 */
template <int dim, typename Scalar>
class IdealizedLVFiberGuccione : public BaseSolverGuccione<dim, Scalar> {
  /**
   * Alias for base class
   */
  using Base = BaseSolverGuccione<dim, Scalar>;
  /**
   * Sacado automatic differentiation type code from
   */
  static constexpr Differentiation::AD::NumberTypes ADTypeCode =
      Differentiation::AD::NumberTypes::sacado_dfad;
  /**
   * Alias for the AD helper
   */
  using ADHelper =
      Differentiation::AD::ResidualLinearization<ADTypeCode, double>;
  /**
   * Alias for the AD number type
   */
  using ADNumberType = typename ADHelper::ad_type;

public:
  /**
   * @brief Constructor
   * @param parameters_file_name_ The parameters file name
   * @param problem_name_ The problem name
   */
  IdealizedLVFiberGuccione(const std::string &parameters_file_name_,
                           const std::string &problem_name_)
      : Base(problem_name_),
        zero_function(dealii::Functions::ZeroFunction<dim>(dim)) {
    initialize_param_handler(parameters_file_name_);
    initialise_boundaries_tag();
    initialize_pk_weights();
  }
  /**
   * @see Base::solve_newton
   */
  void solve_newton() override {
    Base::pcout << "==============================================="
                << std::endl;
    poisson_solver->assemble();
    poisson_solver->solve();

    Base::pcout << "==============================================="
                << std::endl;

    auto log_pressure = [&]() {
      Base::pcout << "Current pressure = " << std::fixed << std::setprecision(6)
                  << Base::pressure.value() << " Pa" << std::endl;
      Base::pcout << "Current fiber pressure = " << std::fixed
                  << std::setprecision(6) << fiber_pressure.value() << " Pa"
                  << std::endl;
    };

    unsigned int n_iter = 0;
    double residual_norm = Base::newton_solver_utility.get_tolerance() + 1;
    {
      const std::chrono::high_resolution_clock::time_point begin_time =
          std::chrono::high_resolution_clock::now();

      log_pressure();

      while (n_iter < Base::newton_solver_utility.get_max_iterations()) {

        unsigned solver_steps = 0;
        Base::assemble_system();
        solver_steps = Base::solve_system();

        // update our solution
        residual_norm = Base::delta_owned.l2_norm();
        Base::solution_owned += Base::delta_owned;
        Base::solution = Base::solution_owned;

        Base::pcout << "Newton iteration " << n_iter << "/"
                    << Base::newton_solver_utility.get_max_iterations()
                    << " - ||r|| = " << std::scientific << std::setprecision(6)
                    << residual_norm << "   " << solver_steps << " "
                    << LinearSolverUtility<Scalar>::solver_type_matcher_rev
                           [Base::linear_solver_utility.get_solver_type()]
                    << " iterations" << std::endl
                    << std::flush;

        ++n_iter;

        // Exit condition: we have reached the treashold residual value and the
        // applied pressure is the whole
        if (residual_norm < Base::newton_solver_utility.get_tolerance() &&
            static_cast<double>(Base::pressure.get_reduction_factor()) >= 1.0) {
          n_iter = Base::newton_solver_utility.get_max_iterations();
        }

        // Solved Newton iteration with lower pressure that requested. Enhance
        // the applied pressure value and solve again
        if (residual_norm < Base::newton_solver_utility.get_tolerance() &&
            static_cast<double>(Base::pressure.get_reduction_factor()) < 1.0) {
          Base::pressure.increment_reduction_factor();
          fiber_pressure.increment_reduction_factor();

          log_pressure();
        }
      }
      const std::chrono::high_resolution_clock::time_point end_time =
          std::chrono::high_resolution_clock::now();
      const long long diff =
          std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                                begin_time)
              .count();
      Base::pcout << std::endl
                  << "Newton metod ended in " << diff << " us" << std::endl;

      if (Base::mpi_rank == 0) {
        const std::string report_name =
            LinearSolverUtility<Scalar>::solver_type_matcher_rev
                [Base::linear_solver_utility.get_solver_type()] +
            "_" +
            LinearSolverUtility<Scalar>::preconditioner_type_matcher_rev
                [Base::linear_solver_utility.get_preconditioner_type()];
        Reporter reporter(
            report_name + ".log",
            "TYPE,TIME(us),NEWTON_ITERATIONS,NEWTON_RESIDUAL_NORM",
            report_name);
        reporter.write(diff, ',', n_iter, ',', residual_norm);
      }
    }

    Base::pcout << "==============================================="
                << std::endl;
  }
  /**
   * @see Base::setup
   */
  void setup() override {
    poisson_solver =
        std::make_unique<Poisson<dim, Scalar>>(Base::mesh_file_name, Base::r);
    poisson_solver->setup();
    Base::setup();
    switch (Base::triangulation_type) {
    case Base::TriangulationType::T: {
      FE_SimplexP<dim> fe_linear(Base::r);
      MappingFE mapping(fe_linear);
      DoFTools::map_dofs_to_support_points(mapping, poisson_solver->dof_handler,
                                           dofs_support_points);
      break;
    };
    case Base::TriangulationType::Q: {
      FE_Q<dim> fe_linear(Base::r);
      MappingFE mapping(fe_linear);
      DoFTools::map_dofs_to_support_points(mapping, poisson_solver->dof_handler,
                                           dofs_support_points);
      break;
    };
    }
#ifdef BUILD_TYPE_DEBUG
    std::cout << "  rank = " << Base::mpi_rank
              << " dof support points count (based on poisson dofs) = "
              << dofs_support_points.size() << std::endl;
    std::ofstream out_f("dofs_support_points" + std::to_string(Base::mpi_rank) +
                        ".log");
    for (auto &[k, v] : dofs_support_points) {
      out_f << v[0] << " " << v[1] << " " << v[2] << std::endl;
    }
    out_f.close();
#endif
  }

protected:
  /**
   * @see Base::initialise_boundaries_tag
   */
  void initialise_boundaries_tag() override {
    //  Set Newmann boundary faces
    for (auto &t : Base::boundaries_utility.get_newmann_boundaries_tags()) {
      Base::newmann_boundary_faces.insert(t);
    }

    // Set Dirichlet boundary faces
    for (auto &t : Base::boundaries_utility.get_dirichlet_boundaries_tags()) {
      Base::dirichlet_boundary_functions[t] = &zero_function;
    }
  };
  /**
   * @see Base::initialize_param_handler
   */
  void initialize_param_handler(const std::string &file_) override {
    declare_parameters();
    parse_parameters(file_);
  }
  /**
   * @see Base::declare_parameters
   */
  void declare_parameters() override {
    Base::declare_parameters();
    Base::prm.enter_subsection("MeshGeometry");
    {
      Base::prm.declare_entry("EpiRl", "0.0", Patterns::Double(0.0),
                              "Epicardium major axe length (mm)");
      Base::prm.declare_entry("EpiRs", "0.0", Patterns::Double(0.0),
                              "Epicardium minor axe length (mm)");
      Base::prm.declare_entry("EndoRl", "0.0", Patterns::Double(0.0),
                              "Endocardium major axe length (mm)");
      Base::prm.declare_entry("EndoRs", "0.0", Patterns::Double(0.0),
                              "Endocardium minor axe length (mm)");
    }
    Base::prm.leave_subsection();
  }
  /**
   * @see Base::parse_parameters
   */
  void parse_parameters(const std::string &file_) override {
    Base::parse_parameters(file_);
    Base::prm.enter_subsection("MeshGeometry");
    {
      Ellipsoid::endo_r_l = Base::prm.get_double("EndoRl");
      Ellipsoid::endo_r_s = Base::prm.get_double("EndoRs");
      Ellipsoid::epi_r_l = Base::prm.get_double("EpiRl");
      Ellipsoid::epi_r_s = Base::prm.get_double("EpiRs");
    }
    Base::prm.leave_subsection();
    Base::prm.enter_subsection("Pressure");
    fiber_pressure = typename Base::ConstantPressureFunction(
        Base::prm.get_double("FiberValue"),
        Base::prm.get_double("InitialReductionFactor"),
        Base::prm.get_double("ReductionFactorIncrement"));
    Base::prm.leave_subsection();

    Base::pcout << "Problem pressure configuration" << std::endl;
    Base::pcout << "  Boundary pressure value: "
                << Base::pressure.value() /
                       Base::pressure.get_reduction_factor()
                << " Pa" << std::endl;
    Base::pcout << "  Fiber pressure value: "
                << fiber_pressure.value() /
                       fiber_pressure.get_reduction_factor()
                << " Pa" << std::endl;
    Base::pcout << "-----------------------------------------------"
                << std::endl;
    Base::pcout << "Problem geometry configuration" << std::endl;
    Base::pcout << "  Endocardium major axe: " << Ellipsoid::endo_r_l << "mm"
                << std::endl;
    Base::pcout << "  Endocardium minor axe: " << Ellipsoid::endo_r_s << "mm"
                << std::endl;
    Base::pcout << "  Epicardium major axe: " << Ellipsoid::epi_r_l << "mm"
                << std::endl;
    Base::pcout << "  Epicardium minor axe: " << Ellipsoid::epi_r_s << "mm"
                << std::endl;
    Base::pcout << "==============================================="
                << std::endl;
  }
  /**
   * @see Base::initialize_pk_weights
   */
  void initialize_pk_weights() override {
    // Set the Piola Kirchhoff b_x values
    Base::piola_kirchhoff_b_weights[{0, 0}] = Base::Material::b_f;
    Base::piola_kirchhoff_b_weights[{1, 1}] = Base::Material::b_t;
    Base::piola_kirchhoff_b_weights[{2, 2}] = Base::Material::b_t;
    Base::piola_kirchhoff_b_weights[{1, 2}] = Base::Material::b_t;
    Base::piola_kirchhoff_b_weights[{2, 1}] = Base::Material::b_t;
    Base::piola_kirchhoff_b_weights[{0, 2}] = Base::Material::b_fs;
    Base::piola_kirchhoff_b_weights[{2, 0}] = Base::Material::b_fs;
    Base::piola_kirchhoff_b_weights[{0, 1}] = Base::Material::b_fs;
    Base::piola_kirchhoff_b_weights[{1, 0}] = Base::Material::b_fs;
  }
  /**
   * @see Base::compute_piola_kirchhoff
   */
  void compute_piola_kirchhoff(
      Tensor<2, dim, ADNumberType> &out_tensor,
      const Tensor<2, dim, ADNumberType> &solution_gradient_quadrature,
      const unsigned cell_index) override {

    auto norm = [](const auto &v) {
      auto sum = 0.0;
      for (auto &n : v) {
        sum += n * n;
      }
      return static_cast<decltype(v[0])>(std::sqrt(sum));
    };

    auto normalize = [](auto &v, const auto norm) {
      for (auto &n : v) {
        n /= norm;
      }
    };

    const auto &poisson_solution = poisson_solver->get_solution();
    const auto &poisson_dof_indices =
        poisson_solver->get_aggregate_dof_indices();
    Tensor<1, dim, ADNumberType> f;

    for (unsigned i = 0; i < poisson_solver->fe->dofs_per_cell; ++i) {
      const auto global_index = poisson_dof_indices[cell_index][i];
      const auto &support_point = dofs_support_points[global_index];

#ifdef BUILD_TYPE_DEBUG
      Assert(global_index >= 0 && global_index < poisson_solution.size(),
             ExcMessage("global_index out of bounds, global_index = " +
                        std::to_string(global_index) +
                        " poisson_solution size = " +
                        std::to_string(poisson_solution.size()) + "\n"));
#endif
      // retrive the t value
      const Scalar t = poisson_solution[global_index];
      // compute fiber parameters
      const Scalar endo_r_2 = Ellipsoid::endo_r_l;
      const Scalar endo_r_1 = Ellipsoid::endo_r_s;
      const Scalar epi_r_2 = Ellipsoid::epi_r_l;
      const Scalar epi_r_1 = Ellipsoid::epi_r_s;
      const Scalar endo_epi_r_1_delta = std::abs(endo_r_1 - epi_r_1);
      const Scalar endo_epi_r_2_delta = std::abs(endo_r_2 - epi_r_2);
      const Scalar r_s = endo_r_1 + endo_epi_r_1_delta * t;
      const Scalar r_e = endo_r_2 + endo_epi_r_2_delta * t;
      const Scalar u_rad = std::acos(support_point[2] / r_e);
      const Scalar v_rad1 =
          std::asin(support_point[1] / (r_s * std::sin(u_rad)));
      const Scalar v_rad2 =
          std::acos(support_point[0] / (r_s * std::sin(u_rad)));
      const Scalar alpha_deg = 90.0 - 180.0 * t;
      const Scalar alpha_rad = alpha_deg * (M_PI / 180.0);

      // sometimes when the argument of the std::asin or std::acos is ~1/~-1 we
      // have detected numerical imprecisions that will lead to NaN. We are
      // skipping those Dofs contributions (from tests we have detected only one
      // point over ~74k total Dofs (with r = 1)).
      if ((std::isnan(v_rad1) && std::isnan(v_rad2)) || std::isnan(u_rad)) {
#ifdef BUILD_TYPE_DEBUG
        std::cout << "ERR: rank: " << Base::mpi_rank
                  << " failed point with global index = " << global_index
                  << ": " << support_point[0] << " " << support_point[1] << " "
                  << support_point[2] << std::endl;
#endif
        continue;
      }
      // angle v can be recovered by using 2 equations
      // (https://pubmed.ncbi.nlm.nih.gov/26807042/)
      const Scalar v_rad = std::isnan(v_rad1) ? v_rad2 : v_rad1;

      // declare partial derivative vectors (handmade derivative)
      std::vector<Scalar> dx_du = {r_s * std::cos(u_rad) * std::cos(v_rad),
                                   r_s * std::cos(u_rad) * std::sin(v_rad),
                                   -r_e * std::sin(u_rad)};
      std::vector<Scalar> dx_dv = {
          static_cast<Scalar>(-1) * r_s * std::sin(u_rad) * std::sin(v_rad),
          r_s * std::sin(u_rad) * std::cos(v_rad), static_cast<Scalar>(0)};
      normalize(dx_du, norm(dx_du));
      normalize(dx_dv, norm(dx_dv));
      // compute vector f
      for (unsigned i = 0; i < dim; ++i) {
        f[i] +=
            ADNumberType(dx_du[i]) * Sacado::Fad::sin(ADNumberType(alpha_rad)) +
            ADNumberType(dx_dv[i]) * Sacado::Fad::cos(ADNumberType(alpha_rad));
      }
    }
    out_tensor +=
        (ADNumberType(fiber_pressure.value()) * dealii::outer_product(f, f));
    Base::compute_piola_kirchhoff(out_tensor, solution_gradient_quadrature,
                                  cell_index);
  }
  /**
   * @brief Ellipsoidal geometry axes size
   */
  struct Ellipsoid {
  public:
    /**
     * Endocardium major axe
     */
    static Scalar endo_r_l;
    /**
     * Endocardium minor axe
     */
    static Scalar endo_r_s;
    /**
     * Epicardium major axe
     */
    static Scalar epi_r_l;
    /**
     * Epicardium minor axe
     */
    static Scalar epi_r_s;
  };
  /**
   * Utility zero function for Dirichilet boundary
   */
  const dealii::Functions::ZeroFunction<dim> zero_function;
  /**
   * Dofs support points
   */
  std::map<dealii::types::global_dof_index, Point<dim>> dofs_support_points;
  /**
   * Utility object to solve Poisson problems
   */
  std::unique_ptr<Poisson<dim, Scalar>> poisson_solver;
  /**
   * The fiber pressure T_a value (https://pubmed.ncbi.nlm.nih.gov/26807042/)
   */
  typename Base::ConstantPressureFunction fiber_pressure;
};

template <int dim, typename Scalar>
Scalar IdealizedLVFiberGuccione<dim, Scalar>::Ellipsoid::epi_r_l;

template <int dim, typename Scalar>
Scalar IdealizedLVFiberGuccione<dim, Scalar>::Ellipsoid::epi_r_s;

template <int dim, typename Scalar>
Scalar IdealizedLVFiberGuccione<dim, Scalar>::Ellipsoid::endo_r_l;

template <int dim, typename Scalar>
Scalar IdealizedLVFiberGuccione<dim, Scalar>::Ellipsoid::endo_r_s;

#endif // IDEALIZED_LV_FIBER_GUCCIONE_HP
