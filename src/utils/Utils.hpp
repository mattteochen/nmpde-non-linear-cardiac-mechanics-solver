#ifndef UTILS_HPP
#define UTILS_HPP

#include <deal.II/base/tensor.h>

namespace Utils {
namespace dealii {
namespace Tensor {
/**
 * Retrieve a dealii Tensor<2, dim> transpose
 * @tparam rank The dealii Tensor rank
 * @tparam dim The tensor dimension
 * @param t A ::dealii::Tensor<2, dim> tensor
 * @return The input tensor transpose
 */
template <int rank = 2, int dim>
::dealii::Tensor<rank, dim> get_transpose(
    const ::dealii::Tensor<rank, dim>& t) {
  ::dealii::Tensor<rank, dim> transpose;
  for (uint32_t i = 0; i < dim; i++) {
    for (uint32_t j = 0; j < dim; j++) {
      transpose[j][i] = t[i][j];
    }
  }
  return transpose;
}

/**
 * Retrieve an identity dealii Tensor<2, dim>
 * @tparam rank The dealii Tensor rank
 * @tparam dim The tensor dimension
 * @return The identity tensor
 */
template <int rank = 2, int dim>
const ::dealii::Tensor<rank, dim> get_identity() {
  ::dealii::Tensor<rank, dim> identity;
  for (uint32_t i = 0; i < dim; i++) {
    identity[i][i] = 1.0;
  }
  return identity;
}

}  // namespace Tensor
}  // namespace dealii
}  // namespace Utils

#endif
