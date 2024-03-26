/**
 * @file IdealizedLVfiber.hpp
 * @brief Header file defining the idealized lv solver class.
 */

#ifndef IDEALIZED_LV_HPP
#define IDEALIZED_LV_HPP

#include <transversely_isotropic_constructive_law/BaseSolver.hpp>
#include <Assert.hpp>

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/fe/mapping_fe.h>

#include <cmath>
#include <fstream>
#include <poisson/Poisson.hpp>
#include <string>

/**
 * @class IdealizedLVfiber
 * @brief Class representing the an Idealized LV solver
 * (https://pubmed.ncbi.nlm.nih.gov/26807042/)
 */
template <int dim, typename Scalar>
class IdealizedLVfiber : public BaseSolver<dim, Scalar> {
  /**
   * Alias for base class
   */
  using Base = BaseSolver<dim, Scalar>;
  /**
   * Sacado automatic differentiation type code from
   */
  static constexpr Differentiation::AD::NumberTypes ADTypeCode =
      Differentiation::AD::NumberTypes::sacado_dfad_dfad;
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
   * @param mesh_file_name_ The mesh file name
   * @param problem_name_ The problem name
   */
  IdealizedLVfiber(const std::string &parameters_file_name_,
                   const std::string &mesh_file_name_,
                   const std::string &problem_name_)
      : Base(parameters_file_name_, mesh_file_name_, problem_name_),
        zero_function(dealii::Functions::ZeroFunction<dim>(dim)),
        poisson_solver(mesh_file_name_, 1) {}
  /**
   * @brief Initialise boundaries tag. Boundaries are problem specific hence we
   * override the base virtual implementation.
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
   * @brief Overridden function.
   * 
   * This implementation takes count for the active fiber contraction.
   * 
   * @see Base::baseFunction()
   */
  void compute_piola_kirchhoff(
      Tensor<2, dim, ADNumberType> &out_tensor,
      Tensor<2, dim, ADNumberType> &solution_gradient_quadrature,
      const unsigned cell_index) override {
    Base::compute_piola_kirchhoff(out_tensor, solution_gradient_quadrature, cell_index);

    auto norm = [](const auto& v) {
      auto sum = 0.0;
      for (auto& n : v) {
        sum += n*n;
      }
      return static_cast<decltype(v[0])>(std::sqrt(sum));
    };

    auto normalize = [](auto& v, const auto norm) {
      for (auto& n : v) {
        n /= norm;
      } 
    };

    const auto& poisson_solution = poisson_solver.get_solution();
    const auto& poisson_dof_indices = poisson_solver.get_aggregate_dof_indices();
    //TODO: move const parameter insdie the config file
    const double T_a_pressure = 6000.0;
    Tensor<1, dim> f;
    for (unsigned i=0; i<poisson_solver.fe->dofs_per_cell; ++i) {
      const auto global_index = poisson_dof_indices[cell_index][i];
      const auto& support_point = dofs_support_points[global_index];

      //retrive the t value
      ASSERT(global_index >= 0 && global_index < poisson_solution.size(),
             "global_index out of bounds, global_index = "
                 << global_index << " poisson_solution size = "
                 << poisson_solution.size() << std::endl);
      const double t = poisson_solution[global_index];
      // Base::pcout << "t: " << t << std::endl;
      //compute fiber parameters
      const double d_focal = 45.0;
      const double nu_endo = 0.6;
      const double nu_epi = 0.8;
      const double endo_r_1 = d_focal * std::sinh(nu_endo);
      const double endo_r_2 = d_focal * std::cosh(nu_endo);
      const double epi_r_1 = d_focal * std::sinh(nu_epi);
      const double epi_r_2 = d_focal * std::cosh(nu_epi);
      const double endo_epi_r_1_delta = std::abs(endo_r_1 - epi_r_1);
      const double endo_epi_r_2_delta = std::abs(endo_r_2 - epi_r_2);
      const double r_s = endo_r_1 + endo_epi_r_1_delta * t;
      const double r_e = endo_r_2 + endo_epi_r_2_delta * t;
      const double u_deg = std::acos(support_point[2] / r_e) * (180.0 / M_PI);
      const double u_rad = u_deg * (M_PI / 180.0);
      const double v_deg1 = std::asin(support_point[1] / (r_s * std::sin(u_rad))) * (180.0 / M_PI);
      const double v_deg2 = std::acos(support_point[0] / (r_s * std::sin(u_rad))) * (180.0 / M_PI);
      const double v_rad1 = v_deg1 * (M_PI / 180.0);
      const double v_rad2 = v_deg2 * (M_PI / 180.0);
      const double alpha_deg = 90.0 - 180.0 * t;
      const double alpha_rad = alpha_deg * (M_PI / 180.0);

      if ((std::isnan(v_rad1) && std::isnan(v_rad2)) || std::isnan(u_rad)) {
        std::cout << "ERR: rank: " << Base::mpi_rank
                  << " failed point with global index = " << global_index << ": "
                  << support_point[0] << " " << support_point[1] << " "
                  << support_point[2] << std::endl;
        continue;
      }
      //angle v can be recovered by using 2 equations (https://pubmed.ncbi.nlm.nih.gov/26807042/)
      const double v_rad = std::isnan(v_rad1) ? v_rad2 : v_rad1;

      //declare partial derivative vectors (handmade derivative)
      std::vector<double> dx_du = {
        r_s * std::cos(u_rad) * std::cos(v_rad),
        r_s * std::cos(u_rad) * std::sin(v_rad),
        -r_e * std::sin(u_rad)
      };
      std::vector<double> dx_dv = {
        -1.0 * r_s * std::sin(u_rad) * std::sin(v_rad),
        r_s * std::sin(u_rad) * std::cos(v_rad),
        0
      };
      normalize(dx_du, norm(dx_du));
      normalize(dx_dv, norm(dx_dv));
      //compute vector f
      for (unsigned i=0; i<dim; ++i) {
        f[i] += dx_du[i] * std::sin(alpha_rad) + dx_dv[i] * std::cos(alpha_rad);
      } 
    }
    //TODO: debug the outer product
    const auto ff = T_a_pressure * dealii::outer_product(f, f);
    out_tensor += ff;
  }
  /**
   * @brief Solve the non linear problem using the Newton method
   */
  void solve_newton() override {
    Base::pcout << "==============================================="
                << std::endl;
    poisson_solver.assemble();
    poisson_solver.solve();
    const auto& sol = poisson_solver.get_solution();
    std::ofstream f("t_distances_" + std::to_string(Base::mpi_rank) + ".log");
    std::set<decltype(sol[0])> s;
    for (auto n : sol) {
      s.insert(n);
    }
    for (auto n : s) {
      f << n << std::endl;
    }
    f.close();
    Base::solve_newton();
  }

  void setup() override {
    poisson_solver.setup();
    Base::setup();
    switch (Base::triangulation_type) {
      case Base::TriangulationType::T: {
        FE_SimplexP<dim> fe_linear(Base::r);
        MappingFE mapping(fe_linear);
        DoFTools::map_dofs_to_support_points(mapping, poisson_solver.dof_handler, dofs_support_points);
        break;
      };
      case Base::TriangulationType::Q: {
        FE_Q<dim> fe_linear(Base::r);
        MappingFE mapping(fe_linear);
        DoFTools::map_dofs_to_support_points(mapping, poisson_solver.dof_handler, dofs_support_points);
        break;
      };
    }
    std::cout << "  rank = " << Base::mpi_rank << " dof support points size (based on poisson dofs) = " << dofs_support_points.size() << std::endl;
    std::ofstream f("dofs_support_points" + std::to_string(Base::mpi_rank) + ".log");
    for (auto& [k, v] : dofs_support_points) {
      f << v[0] << " " << v[1] << " " << v[2] << std::endl;
    }
    f.close();
  }

protected:
  /**
   * Utility zero function for Dirichilet boundary
   */
  dealii::Functions::ZeroFunction<dim> zero_function;
  /**
   * Dofs support points
   */
  std::map<dealii::types::global_dof_index, Point<dim>> dofs_support_points;
  /**
   * Utility object to solve Poisson problems
   */
  Poisson<dim, Scalar> poisson_solver;
};

#endif // IDEALIZED_LV_HPP
