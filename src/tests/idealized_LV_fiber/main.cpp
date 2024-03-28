#include <IdealizedLVFiber.hpp>

int main(int argc, char *argv[]) {
  dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
  std::string mesh_file_name =
      "../../../../lifex_fiber_generation_examples/mesh/idealized_LV.msh";
  std::string parameter_file_name = "../parameters.prm";
  const std::string problem_name = "idealized_lv";

  if (argc > 1) {
    mesh_file_name = argv[1];
  }
  if (argc > 2) {
    parameter_file_name = argv[2];
  }

  IdealizedLVFiber<3, double> problem(parameter_file_name, mesh_file_name,
                                 problem_name);
  problem.initialise_boundaries_tag();
  problem.setup();
  problem.solve_newton();
  problem.output();
  return 0;
}
