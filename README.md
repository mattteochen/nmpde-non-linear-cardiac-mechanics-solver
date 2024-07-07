# Non-linear cardiac mechanics
The problem entails solving the partial differential equation concerning the contraction of
the left ventricle when a force acts on the internal surface of the ventricle, known as the
Endocardium. We model the acting forces as pressure without considering
how the force is generated. Our modeling strategy follows a bottom-up approach: initially,
we employ a simplified geometry and basic mathematical model, making assumptions about
material properties. Subsequently, we refine the model by incorporating a more intricate
formulation that accounts for the material properties. This testing methodology is inspired
by the approach outlined [here](https://pubmed.ncbi.nlm.nih.gov/26807042/).

## Constructive law: Guccione
This section covers the finite element solver for materials that follow the [Guccione’s Law](https://pubmed.ncbi.nlm.nih.gov/8550635/).

An abstract base solver class `BaseSolverGuccione` has been introduced as a base line starting point for each mesh test that we have run.
This is a complete solver except for the boundaries initialization which must occur in any derived class as every problem is different.

The base class takes in account a pressure component that pushes on ***x*** boundaries, referred as Newmann boundaries.
If your problem does not follow this pattern, please ***override*** all the needed implementations (this might the `assemble` method).

Dirichlet boundaries condition values are assigned in the derived class.

`BaseSolverGuccione`'s extension is a trivial task in order to create custom tests (e.g. `SlabCubicGuccione`).

## Constructive law: New Hook
This section covers the finite element solver for materials that follow the New Hook law.

An abstract base solver class `BaseSolverNewHook` has been introduced as a base line starting point for each mesh test that we have run.
This is a complete solver except for the boundaries initialization which must occur in any derived class as every problem is different.

The base class takes in account a pressure component that pushes on ***x*** boundaries, referred as Newmann boundaries.
If your problem does not follow this pattern, please ***override*** all the needed implementations (this might the `assemble` method).

Dirichlet boundaries condition values are assigned in the derived class.

`BaseSolverNewHook`'s extension is a trivial task in order to create custom tests (e.g. `SlabCubicNewHook`).

## Non Linear Solver
As the cardiac problem is non linear, we have employed the [Newton method](https://en.wikipedia.org/wiki/Newton%27s_method).
The Jacobian matrix has been created by using `dealii` built-in [AD](https://www.dealii.org/current/doxygen/deal.II/group__auto__symb__diff.html) packages. 

## Linear Solver
Currently the following ***iterative solvers*** are supported in order to solve a single Newton iteration:
- [x] GMRES
- [x] BiCGSTAB

***Preconditioners*** available:
- [x] ILU
- [x] SOR
- [x] SSOR
- [x] AMG

## Configuration file
The single entry point for all run time and application specific parameters can be specified inside a configuration file called `your_config_name.prm`.

Changes can be tested without recompilation if they can be modified through the configuration file.

A mock configuration file for a Guccione constructive law problem:
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
  # b_f value of the material
  set b_f                       = 8.0

  # b_t value of the material
  set b_t                       = 2.0

  # b_fs value of the material
  set b_fs                       = 4.0

  # C value of the material
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

subsection Mesh
  # Mesh file name
  set File                      = /path/to/mesh
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
