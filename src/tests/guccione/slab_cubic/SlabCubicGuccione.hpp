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
   * @param mesh_file_name_ The mesh file name
   * @param problem_name_ The problem name
   */
  SlabCubicGuccione(const std::string &parameters_file_name_,
                    const std::string &mesh_file_name_,
                    const std::string &problem_name_)
      : Base(parameters_file_name_, mesh_file_name_, problem_name_) {
    Base::pcout << "Problem boundary pressure configuration" << std::endl;
    Base::pcout << "  Value: " << Base::pressure.value() << " Pa" << std::endl;
    Base::pcout << "==============================================="
                << std::endl;
  }
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

protected:
  /**
   * Utility zero function for Dirichilet boundary
   */
  dealii::Functions::ZeroFunction<dim> zero_function;
};

#endif // SLAB_CUBIC_GUCCIONE_HPP
