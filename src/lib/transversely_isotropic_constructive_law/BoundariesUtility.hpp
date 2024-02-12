/**
 * @file BoundariesUtility.hpp
 * @brief Header file defining the boundary utility class.
 */

#ifndef BOUNDARIES_UTILITY_HPP
#define BOUNDARIES_UTILITY_HPP

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
   * @param newmann_boundaries_ A string containing Newmann boundary tags
   * separated by commas.
   */
  BoundariesUtility(const std::string &dirichlet_boundaries_,
                    const std::string &newmann_boundaries_) {
    std::istringstream dirichlet_iss(dirichlet_boundaries_);
    std::istringstream newmann_iss(newmann_boundaries_);
    for (std::string token; std::getline(dirichlet_iss, token, ',');) {
      dirichlet_boundaries.push_back(std::stoi(token));
    }
    for (std::string token; std::getline(newmann_iss, token, ',');) {
      newmann_boundaries.push_back(std::stoi(token));
    }
  }
  /**
   * @brief Move assignment operator.
   * @param other Another BoundariesUtility object to be moved from.
   */
  void operator=(BoundariesUtility &&other) {
    dirichlet_boundaries = std::move(other.dirichlet_boundaries);
    newmann_boundaries = std::move(other.newmann_boundaries);
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
   * @brief Get the Newmann boundary tags.
   * @return A constant reference to a vector containing Newmann boundary tags.
   */
  const std::vector<unsigned int> &get_newmann_boundaries_tags() const {
    return newmann_boundaries;
  }

protected:
  std::vector<unsigned int>
      dirichlet_boundaries; /**< Vector to store Dirichlet boundary tags. */
  std::vector<unsigned int>
      newmann_boundaries; /**< Vector to store Newmann boundary tags. */
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
  out << "  Newmann tags: ";
  for (auto &t : bu.get_newmann_boundaries_tags()) {
    out << t << " ";
  }
  out << std::endl;
}

#endif // BOUNDARIES_UTILITY_HPP
