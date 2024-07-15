#include <IdealizedLVFiberGuccione.hpp>

int main(int argc, char *argv[]) {
  dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
  std::string parameter_file_name = "../parameters.prm";
  const std::string problem_name = "idealized_LV_fiber";

  if (argc > 1) {
    parameter_file_name = argv[1];
  }

  IdealizedLVFiberGuccione<3, double> problem(parameter_file_name,
                                              problem_name);
  problem.setup();
  problem.solve_newton();
  problem.output();
  return 0;
}
