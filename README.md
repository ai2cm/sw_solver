# SWES - Shallow Water Equations on Sphere

[![.github/workflows/ci.yml](https://github.com/ai2cm/sw_solver/actions/workflows/ci.yml/badge.svg)](https://github.com/ai2cm/sw_solver/actions/workflows/ci.yml)
[![sloc](https://img.shields.io/tokei/lines/github/ai2cm/sw_solver)](https://github.com/ai2cm/sw_solver)
[![license](https://img.shields.io/github/license/ai2cm/sw_solver)](https://github.com/ai2cm/sw_solver/blob/main/LICENSE)

Simple approximate solver for the shallow water equations discretized on a sphere using a cartesian grid.

There are shared utilities using numpy, as well as a few implementations of the solver, including a base version, and one using [gt4py](https://github.com/GridTools/gt4py).

## Getting Started

### Dependencies

- Python 3.8+ and setuptools for installation
- `pre-commit` for development
- `tox` and/or direct `pytest` for testing

### Installation

`sw_solver` uses the standard python packaging guidelines with `setuptools`,
as such, it is installable via `pip` or plain `setup.py`.

### Examples

Scripts are located in the `examples/` directory.

## Help

Currently there is no documentation hub, but each public function and method has documentation enforced by `pre-commit`.
