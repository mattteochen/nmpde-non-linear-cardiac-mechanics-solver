#include "SlabCubic.hpp"

int main(int argc, char* argv[]) {
  dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
  const std::string mesh_file_name =
      "../../../../lifex_fiber_generation_examples/mesh/slab_cubic.msh";
  const std::string parameter_file_name =
      "../parameters.prm";
  const std::string problem_name = "slab_cubic";
  const unsigned int degree = 1;
  SlabCubic<3, double> s(parameter_file_name, mesh_file_name, degree, problem_name);
  s.initialise_boundaries_tag();
  s.setup();
  s.solve_newton();
  s.output();
  return 0;
}
