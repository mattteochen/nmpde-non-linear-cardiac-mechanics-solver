/**
 * @file BoundariesUtility.hpp
 * @brief Header file defining the boundary utility class.
 */

#ifndef BOUNDARIES_UTILITY_HPP
#define BOUNDARIES_UTILITY_HPP

#include <deal.II/base/types.h>

#include <iterator>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

/**
 * @brief Utility class for managing boundary conditions.
 */
class BoundariesUtility {
public:
  /**
   * @brief Default constructor.
   */
  BoundariesUtility() {}
  /**
   * @brief Parameterized constructor.
   * @param dirichlet_boundaries_ A string containing Dirichlet boundary tags
   * separated by commas.
   * @param neumann_boundaries_ A string containing Neumann boundary tags
   * separated by commas.
   */
  BoundariesUtility(const std::string &dirichlet_boundaries_,
                    const std::string &neumann_boundaries_) {
    std::istringstream dirichlet_iss(dirichlet_boundaries_);
    std::istringstream neumann_iss(neumann_boundaries_);
    for (std::string token; std::getline(dirichlet_iss, token, ',');) {
      dirichlet_boundaries.push_back(std::stoi(token));
    }
    for (std::string token; std::getline(neumann_iss, token, ',');) {
      neumann_boundaries.push_back(std::stoi(token));
    }
  }
  /**
   * @brief Move assignment operator.
   * @param other Another BoundariesUtility object to be moved from.
   */
  void operator=(BoundariesUtility &&other) {
    dirichlet_boundaries = std::move(other.dirichlet_boundaries);
    neumann_boundaries = std::move(other.neumann_boundaries);
  }
  /**
   * @brief Get the Dirichlet boundary tags.
   * @return A constant reference to a vector containing Dirichlet boundary
   * tags.
   */
  const std::vector<unsigned int> &get_dirichlet_boundaries_tags() const {
    return dirichlet_boundaries;
  }
  /**
   * @brief Get the Neumann boundary tags.
   * @return A constant reference to a vector containing Neumann boundary tags.
   */
  const std::vector<unsigned int> &get_neumann_boundaries_tags() const {
    return neumann_boundaries;
  }

protected:
  std::vector<dealii::types::boundary_id>
      dirichlet_boundaries; /**< Vector to store Dirichlet boundary tags. */
  std::vector<dealii::types::boundary_id>
      neumann_boundaries; /**< Vector to store Neumann boundary tags. */
};

/**
 * @brief Overloaded stream insertion operator to output BoundaryUtility object
 * information.
 * @param out The output stream.
 * @param bu The BoundaryUtility object.
 */
inline void operator<<(std::ostream &out, BoundariesUtility const &bu) {
  out << "  Dirichlet tags: ";
  for (auto &t : bu.get_dirichlet_boundaries_tags()) {
    out << t << " ";
  }
  out << std::endl;
  out << "  Neumann tags: ";
  for (auto &t : bu.get_neumann_boundaries_tags()) {
    out << t << " ";
  }
  out << std::endl;
}

#endif // BOUNDARIES_UTILITY_HPP
