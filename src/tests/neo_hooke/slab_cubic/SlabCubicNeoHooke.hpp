/**
 * @file SlabCubicNeoHooke.hpp
 * @brief Header file defining the slab cubic solver class.
 */

#ifndef SLAB_CUBIC_NEO_HOOKE_HPP
#define SLAB_CUBIC_NEO_HOOKE_HPP

#include <cardiac_mechanics/BaseSolverNeoHooke.hpp>

/**
 * @class SlabCubicNeoHooke
 * @brief Class representing the Slab Cubic solver
 * (https://pubmed.ncbi.nlm.nih.gov/26807042/)
 */
template <int dim, typename Scalar>
class SlabCubicNeoHooke : public BaseSolverNeoHooke<dim, Scalar> {
  /**
   * Alias for base class
   */
  using Base = BaseSolverNeoHooke<dim, Scalar>;

public:
  /**
   * @brief Constructor
   * @param parameters_file_name_ The parameters file name
   * @param problem_name_ The problem name
   */
  SlabCubicNeoHooke(const std::string &parameters_file_name_,
                   const std::string &problem_name_)
      : Base(problem_name_),
        zero_function(dealii::Functions::ZeroFunction<dim>(dim)) {
    initialize_param_handler(parameters_file_name_);
    initialise_boundaries_tag();
  }
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

protected:
  /**
   * Utility zero function for Dirichilet boundary
   */
  dealii::Functions::ZeroFunction<dim> zero_function;
};

#endif // SLAB_CUBIC_NEO_HOOKE_HPP
