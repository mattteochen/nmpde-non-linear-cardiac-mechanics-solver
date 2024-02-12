# Non-linear cardiac mechanics

Add into blah blah....

## Constructive law: transversely isotropic constitutive law
This section covers the finite element solver for materials that follow the [transversely isotropic constitutive law](https://pubmed.ncbi.nlm.nih.gov/8550635/).

An abstract base solver class `BaseSolver` has been introduced as a base line starting point for each mesh test that we have run.
This is a complete solver except for the boundaries initialization which must occur in any derived class as every problem is different.

The base class takes in account a pressure component that pushes on ***x*** boundaries, referred as Newmann boundaries.
If your problem does not follow this pattern, please ***override*** all the needed implementations (this might the `assemble` method).

Dirichlet boundaries condition values are assigned in the derived class.

`BaseSolver`'s extension is a trivial task in order to create more customizations or just to test different meshes/boundaries conditions:
```
#ifndef DEMO_HPP
#define DEMO_HPP

#include <BaseSolver.hpp>

template <int dim, typename Scalar>
class Demo : public BaseSolver<dim, Scalar> {
  using Base = BaseSolver<dim, Scalar>;
public:
  Demo(const std::string &parameters_file_name_,
            const std::string &mesh_file_name_,
            const std::string &problem_name_)
      : Base(parameters_file_name_, mesh_file_name_, problem_name_) {}

  void initialise_boundaries_tag() override {
    //  Set Newmann boundary faces
    for (auto &t : Base::boundaries_utility.get_newmann_boundaries_tags()) {
      Base::newmann_boundary_faces.insert(t);
    }

    // Set Dirichlet boundary faces
    for (auto &t : Base::boundaries_utility.get_dirichlet_boundaries_tags()) {
      Base::dirichlet_boundary_functions[t] = &zero_function;
    }
  };

 protected:
  dealii::Functions::ZeroFunction<dim> zero_function;
};

#endif  // DEMO_HPP
```

### Non Linear Solver
As the cardiac problem is non linear, we have employed the [Newton method](https://en.wikipedia.org/wiki/Newton%27s_method).
The Jacobian matrix has been created by using `dealii` built-in [AD](https://www.dealii.org/current/doxygen/deal.II/group__auto__symb__diff.html) packages. 

### Linear Solver
Currently the following ***iterative solvers*** are supported in order to solve a single Newton iteration:
- [x] GMRES
- [x] BiCGSTAB
- [] CG

***Preconditioners*** available:
- [x] SSOR

### Configuration file
The single entry point for all run time and application specific parameters can be specified inside a configuration file called `your_config_name.prm`.

Changes can be tested without recompilation if they can be modified through the configuration file.

A mock configuration file:
```
subsection LinearSolver
  # Type of solver used to solve the linear system
  set SolverType                = BiCGSTAB

  # Linear solver residual (scaled by residual norm)
  set Residual                  = 1e-6

  # Linear solver iterations multiplier (to be multiplied to matrix size)
  set MaxIteration              = 1.0

  # Preconditioner type
  set PreconditionerType        = SSOR
end

subsection PolynomialDegree
  # Degree of the polynomial for finite elements
  set Degree                    = 1
end

subsection NewtonMethod
  # Newton method residual
  set Residual                  = 1e-6

  # Newton method max iterations
  set MaxIterations             = 100
end

subsection Material
  # b_f value of the strain energy tensor
  set b_f                       = 8.0

  # b_t value of the strain energy tensor
  set b_t                       = 2.0

  # b_fs value of the strain energy tensor
  set b_fs                       = 4.0

  # C value of the strain energy tensor
  set C                          = 2000.0
end

subsection Boundaries
  # Dirichlet boundaries tags
  set Dirichlet                  = 40

  # Newmann boundaries tags
  set Newmann                    = 60
end

subsection Pressure
  # External pressure value
  set Value                      = 4.0
end
```
**Please be sure to add every subsection shown above as for missing sections, the program sets a default value that may lead to undefined results**.

## Enviroenment
This project is dependent on several c++ packages as:
- [dealii](https://www.dealii.org/current/doxygen/deal.II/) >= v9.0
- [MPI](https://www.open-mpi.org/) >= v3.1
- [BOOST](https://www.boost.org/) >= v1.76.0

To ease the configuration, a scientific docker enviroenment can be found [here](https://hub.docker.com/r/pcafrica/mk).

## Compiling
The following commands are dedicated for enviroenment with `mk` modules installed.

To build the executable, make sure you have loaded the needed modules with
```bash
$ module load gcc-glibc dealii
```
Then run the following commands:
```bash
$ mkdir build
$ cd build
$ cmake ..
$ make
```
The executable will be created into `build`, and can be executed through
```bash
$ mpirun -n x executable_name
```

## Formatting
All the source files are formatted by using `clang-format` with `LLVM` coding style.
```
clang-format -i -style=LLVM --sort-includes your_file
```
