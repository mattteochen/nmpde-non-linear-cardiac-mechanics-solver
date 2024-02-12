# Non-linear cardiac mechanics

Add into blah blah....

## Codebase
An abstract base solver class `BaseSolver` has been introduced as a base line starting point for each mesh test that we have run.
This is a complete solver except for the boundaries initialization which must occur in any derived class as every problem is different.

The base class takes in account a pressure component that pushes on `x` boundaries, referred as Newmann boundaries.
If your problem does not follow this pattern, please ***override*** all the needed implementations (this might the `assemble` method).

Dirichlet boundaries condition values are assigned in the derived class.

The application control point lives in the [`parameters.prm`](https://www.dealii.org/current/doxygen/deal.II/classParameterHandler.html) configuration file.

`BaseSolver`'s extension is a trivial task in order to create more customizations or just to test different meshes/boundaries conditions (see `SlabCubic` or `IdealizedLV`).

## Non Linear Solver
As the cardiac problem is non linear, we have employed the [Newton method](https://en.wikipedia.org/wiki/Newton%27s_method).
The Jacobian matrix has been created by using `dealii` built-in [AD](https://www.dealii.org/current/doxygen/deal.II/group__auto__symb__diff.html) packages. 

## Linear Solver
Currently the following ***iterative solvers*** are supported in order to solve a single Newton iteration:
[x] GMRES
[x] BiCGSTAB
[] CG

***Preconditioners*** available:
[x] SSOR

## Configuration file
The single entry point for all run time and application specific parameters can be specified inside a configuration file called `your_config_name.prm`.

## Compiling
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
