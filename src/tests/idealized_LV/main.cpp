#include "IdealizedLV.hpp"

int main(int argc, char* argv[]) {
  dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
  const std::string mesh_file_name =
      "../../../../lifex_fiber_generation_examples/mesh/idealized_LV.msh";
  const unsigned int degree = 1;
  IdealizedLV lv(mesh_file_name, degree);
  lv.setup();
  lv.solve_newton();
  lv.output();
  return 0;
}
