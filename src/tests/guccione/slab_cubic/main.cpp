#include <SlabCubicGuccione.hpp>

int main(int argc, char *argv[]) {
  dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const std::string problem_name = "slab_cubic";
  std::string parameter_file_name = "../parameters.prm";
  if (argc > 1) {
    parameter_file_name = argv[1];
  }

  SlabCubicGuccione<3, double> problem(parameter_file_name, problem_name);
  problem.setup();
  problem.solve_newton();
  problem.output();

  return 0;
}
