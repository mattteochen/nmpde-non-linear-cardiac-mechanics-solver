#include <IdealizedLV.hpp>

int main(int argc, char* argv[]) {
  dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
  const std::string mesh_file_name =
      "../../../../lifex_fiber_generation_examples/mesh/idealized_LV.msh";
  const std::string parameter_file_name = "../parameters.prm";
  const std::string problem_name = "idealized_lv";
  IdealizedLV<3, double> s(parameter_file_name, mesh_file_name, problem_name);
  s.initialise_boundaries_tag();
  s.setup();
  s.solve_newton();
  s.output();
  return 0;
}
