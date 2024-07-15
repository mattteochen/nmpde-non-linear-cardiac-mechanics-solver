#include <IdealizedLVNeoHooke.hpp>

int main(int argc, char *argv[]) {
  dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
  std::string parameter_file_name = "../parameters.prm";
  const std::string problem_name = "idealized_LV";

  if (argc > 1) {
    parameter_file_name = argv[1];
  }

  IdealizedLVNeoHooke<3, double> problem(parameter_file_name, problem_name);
  problem.setup();
  problem.solve_newton();
  problem.output();
  return 0;
}
