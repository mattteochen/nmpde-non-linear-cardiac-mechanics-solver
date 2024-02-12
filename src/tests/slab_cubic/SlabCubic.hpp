/**
 * @file SlabCubic.hpp
 * @brief Header file defining the slab cubic solver class.
 */

#ifndef SLAB_CUBIC_HPP
#define SLAB_CUBIC_HPP

#include <BaseSolver.hpp>

/**
 * @class Class representing the Slab Cubic solver
 */
template <int dim, typename Scalar>
class SlabCubic : public BaseSolver<dim, Scalar> {
  /**
   * Alias for base class
   */
  using Base = BaseSolver<dim, Scalar>;

 public:
  /**
   * @brief Constructor
   * @param parameters_file_name_ The parameters file name
   * @param mesh_file_name_ The mesh file name
   * @param problem_name_ The problem name
   */
  SlabCubic(const std::string &parameters_file_name_,
            const std::string &mesh_file_name_,
            const std::string &problem_name_)
      : Base(parameters_file_name_, mesh_file_name_, problem_name_) {}
  /**
   * @brief Initialise boundaries tag
   */
  void initialise_boundaries_tag() override {
    // TODO: use parameter handler
    //  Set Newmann boundary faces
    Base::newmann_boundary_faces.insert(60);

    // Set Dirichlet boundary faces
    Base::dirichlet_boundary_functions[40] = &zero_function;
  };

 protected:
  /**
   * Utility zero function for Dirichilet boundary
   */
  dealii::Functions::ZeroFunction<dim> zero_function;
};

#endif  // SLAB_CUBIC_HPP
