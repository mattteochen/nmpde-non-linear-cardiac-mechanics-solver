/**
 * @file IdealizedLVNewHook.hpp
 * @brief Header file defining the idealized lv solver class.
 */

#ifndef IDEALIZED_LV_NEW_HOOK_HPP
#define IDEALIZED_LV_NEW_HOOK_HPP

#include <cardiac_mechanics/BaseSolverNewHook.hpp>

/**
 * @class IdealizedLVNewHook
 * @brief Class representing the an Idealized LV solver
 * (https://pubmed.ncbi.nlm.nih.gov/26807042/)
 */
template <int dim, typename Scalar>
class IdealizedLVNewHook : public BaseSolverNewHook<dim, Scalar> {
  /**
   * Alias for base class
   */
  using Base = BaseSolverNewHook<dim, Scalar>;

public:
  /**
   * @brief Constructor
   * @param parameters_file_name_ The parameters file name
   * @param problem_name_ The problem name
   */
  IdealizedLVNewHook(const std::string &parameters_file_name_,
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
   * @see Base::initialize_param_handler()
   */
  void initialize_param_handler(const std::string &file_) override {
    Base::declare_parameters();
    Base::parse_parameters(file_);

    Base::pcout << "Problem boundary pressure configuration" << std::endl;
    Base::pcout << "  Value: " << Base::pressure.value() << " Pa" << std::endl;
    Base::pcout << "==============================================="
                << std::endl;
  }

protected:
  /**
   * Utility zero function for Dirichilet boundary
   */
  dealii::Functions::ZeroFunction<dim> zero_function;
};

#endif // IDEALIZED_LV_NEW_HOOK_HPP
