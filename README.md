# Fokker–Planck Finite Element Solver

This repository accompanies my MSc dissertation **“A Structure-preserving Finite Element for the FENE Dumbbell Fokker–Planck Equation.”**  
It contains the source code used to implement and run all numerical experiments reported in Chapter 5 of the thesis.

## Contents

- `domain/` – mesh generation scripts for physical space (\Omega) and configuration space (D).
- `fem_spaces/` – finite element space construction.
- `utils/` – auxiliary modules, including convex regularisations \(\mathcal{F}_\delta\), the discrete chain-rule operators \(\tilde{\Lambda}\), interpolators \(\pi\), and parameters.
- `experiments/` – scripts to reproduce all numerical experiments from Chapter 5, organised by section number.
- `solvers/` – individual solvers in \(x-\) and \(q\)-space, Strang algorithm, and full solver algorithm.
- `variational_formulations/` – definitions of the numerical schemes (4.1) and (4.2) from the thesis.
- `viz/` – XDMF results visualisation to be displayed in ParaView.

## Requirements

The code is written in Python 3 and was developed in a conda environment.  
Core requirements are **FEniCSx**, **PETSc**, and supporting libraries (`numpy`, `matplotlib`, `gmsh`).

## Usage

To reproduce a given experiment (e.g. §5.1 baseline model):

```bash
cd experiments/5_1_baseline
python -m run
python -m post_processing
```

## Contact

For questions about the content of this repository, please contact:

thomas.hall@maths.ox.ac.uk
