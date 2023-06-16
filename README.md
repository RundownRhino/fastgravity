# fastgravity
A simple [Barnes-Hut simulation](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation) of n-body gravity, written in Rust and exposed as a Python library via `pyo3`. Uses a multipole expansion up to the quadrupole moment. Supports evaluation of both potentials and accelerations.

## Installation
Download the right wheel from Releases for your platform (for example, if you have Python 3.11 on 64-bit Windows, you want the cp**310**-none-**win**_amd**64** one) and install it with `pip install path_to_the_wheel.whl`. Alternatively, see [Building](#building) on building a wheel from source. 

## Building
Compile with `maturin build --release`, then install the resulting wheel via `pip install`.

## Usage
```py
from fastgravity import GravitySystem
import numpy as np

# generate a random arrangement of masses:
n = 20
pts = np.random.random((n, 2))
pts[:,0]/=10
masses = np.random.normal(1, 0.2, n)

# create the simulation object:
tree = GravitySystem(pts, masses)

# grid to evaluate things at:
from_x, to_x, nx = -1.1, 1.1, 300
from_y, to_y, ny = -1.1, 1.1, 300
x = np.linspace(from_x, to_x, nx)
y = np.linspace(from_y, to_y, ny)
grid = np.stack(np.meshgrid(x,y, indexing="ij"), axis=2).reshape(-1, 2)

# evaluate potentials:
potentials = np.array(tree.evaluate_potential(grid)).reshape((len(x),) * 2)
# evaluate accelerations:
fields = np.stack(tree.evaluate_gravity(grid)).reshape((nx, ny, 2))
```

## TODO
- [ ] Stub file for types
- [ ] Parallelization?
- [ ] Higher moments?