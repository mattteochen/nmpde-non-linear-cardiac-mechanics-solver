#include "SlabCubic.hpp"

int main(int argc, char* argv[]) {
  dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
  const std::string mesh_file_name =
      "../../../../lifex_fiber_generation_examples/mesh/slab_cubic.msh";
  const unsigned int degree = 1;
  SlabCubic s(mesh_file_name, degree);
  s.setup();
  s.solve_newton();
  s.output();
  return 0;
}
