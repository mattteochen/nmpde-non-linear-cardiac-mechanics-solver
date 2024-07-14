/**
 * @file SlabCubicGuccione.hpp
 * @brief Header file defining the slab cubic solver class.
 */

#ifndef SLAB_CUBIC_GUCCIONE_HPP
#define SLAB_CUBIC_GUCCIONE_HPP

#include <cardiac_mechanics/BaseSolverGuccione.hpp>

/**
 * @class SlabCubicGuccione
 * @brief Class representing the Slab Cubic solver
 * (https://pubmed.ncbi.nlm.nih.gov/26807042/)
 */
template <int dim, typename Scalar>
class SlabCubicGuccione : public BaseSolverGuccione<dim, Scalar> {
  /**
   * Alias for base class
   */
  using Base = BaseSolverGuccione<dim, Scalar>;

public:
  /**
   * @brief Constructor
   * @param parameters_file_name_ The parameters file name
   * @param problem_name_ The problem name
   */
  SlabCubicGuccione(const std::string &parameters_file_name_,
                    const std::string &problem_name_)
      : Base(problem_name_),
        zero_function(dealii::Functions::ZeroFunction<dim>(dim)) {
    initialize_param_handler(parameters_file_name_);
    initialise_boundaries_tag();
    initialize_pk_weights();
  }

protected:
  /**
   * @see Base::initialise_boundaries_tag
   */
  void initialise_boundaries_tag() override {
    //  Set Neumann boundary faces
    for (auto &t : Base::boundaries_utility.get_neumann_boundaries_tags()) {
      Base::neumann_boundary_faces.insert(t);
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
    Base::declare_parameters();
    Base::parse_parameters(file_);

    Base::pcout << "Problem boundary pressure configuration" << std::endl;
    Base::pcout << "  Value: "
                << Base::pressure.value() /
                       Base::pressure.get_reduction_factor()
                << " Pa" << std::endl;
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
   * Utility zero function for Dirichilet boundary
   */
  dealii::Functions::ZeroFunction<dim> zero_function;
};

#endif // SLAB_CUBIC_GUCCIONE_HPP
