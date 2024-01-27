#include <deal.II/base/tensor.h> // Needed to include dealii
#include <deal.II/lac/vector.h>

#if DEAL_II_VERSION_MAJOR >= 9 && defined(DEAL_II_WITH_TRILINOS)
#include <deal.II/differentiation/ad.h>
#define ENABLE_SACADO_FORMULATION
#endif

#include <iostream>
#include <vector>

template <typename NumberType>
NumberType f(const NumberType &x)
{
  return std::cos(x);
}

template <typename Scalar>
void run_ad(const Scalar x) {
  constexpr unsigned int dim = 1;
  constexpr unsigned int n_independent_variables = 1;
  constexpr dealii::Differentiation::AD::NumberTypes ADTypeCode =
    dealii::Differentiation::AD::NumberTypes::sacado_dfad_dfad;

  using ADHelper =
    dealii::Differentiation::AD::ScalarFunction<dim, ADTypeCode, double>;

  ADHelper ad_helper(n_independent_variables);
  using ADNumberType = typename ADHelper::ad_type;

  ad_helper.register_independent_variables({x});
  const std::vector<ADNumberType> independent_variables_ad = ad_helper.get_sensitive_variables();
  const ADNumberType &x_ad = independent_variables_ad[0];

  const ADNumberType f_ad = f(x_ad);

  ad_helper.register_dependent_variable(f_ad);

  dealii::Vector<double> Df(ad_helper.n_dependent_variables());
  ad_helper.compute_gradient(Df);
  std::cout << "Df:" << std::endl << Df << std::endl;
}

int main(int argc, char* argv[]) {
  (void)argv;
  (void)argc;
  std::cout << "Testing Sacado AD" << std::endl;
  run_ad(2.0);
}
