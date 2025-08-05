# LCHSimulation

A quantum computing framework for simulating partial differential equations (PDEs) using the Ladder-Controlled Hamiltonian Simulation (LCHS) method. This project provides tools for constructing quantum circuits that simulate the evolution of classical PDEs, with particular focus on acoustic wave equations and heat equations.

## Overview

LCHSimulation implements a novel approach to PDE simulation by decomposing differential operators into tensor products of ladder operators and projectors, then constructing quantum circuits that simulate the time evolution. The framework supports both 1D and 2D problems with various boundary conditions.

## Features

- **Operator Construction**: Build differential operators (Laplacian, first-order derivatives) using tensor products
- **Quantum Circuit Generation**: Automatically generate quantum circuits for Hamiltonian simulation
- **State Evolution**: Simulate both state vector and matrix-based evolution
- **Spatial Varying Parameters**: Support for spatially varying coefficients using Quine-McCluskey algorithm
- **Boundary Conditions**: Support for Dirichlet, Neumann, and periodic boundary conditions
- **Visualization**: Built-in plotting capabilities for simulation results

## Installation

### Prerequisites

- Python 3.12
- Conda (recommended)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd LCHSimulation
```

2. Create and activate the conda environment:
```bash
conda env create -f LCHS.yml
conda activate LCHS_env
```

Alternatively, install dependencies directly:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
import numpy as np
from diff_op import LaplacianOperator
from lchsimulation import LCHSimulation

# Create a 2D Laplacian operator
nx, ny = 5, 5
H = LaplacianOperator(num_bits_x=nx, num_bits_y=ny, h=1.0)

# Initialize simulation
lchs = LCHSimulation(op=H)

# Generate quantum circuit
dt = 0.001
circuit = lchs.simulation_circuit(dt=dt)

# Set initial state
initial_state = np.zeros(2**(nx+ny), dtype=np.complex128)
initial_state[0] = 1.0  # Example: ground state

# Run simulation
T = 10.0
state_evolution = lchs.matrix_simulation_evolve(T=T, state=initial_state, sample_rate=10)
```

### Acoustic Wave Simulation

```python

# Create acoustic wave Hamiltonian
H = acoustic_wave_hamiltonian(nx, ny, h, bc_dict, c_dict, c1)
lchs = LCHSimulation(op=H)

# Run simulation
T = 20.0
dt = 0.001
w_q_list = lchs.matrix_simulation_evolve(T=T, state=initial_state, sample_rate=10)
```

## Core Components

### 1. Operator Classes (`diff_op.py`)

- **`Operator`**: Base class for all differential operators
- **`DifferentialOperator`**: First-order differential operators
- **`LaplacianOperator`**: Second-order differential operators (Laplacian)
- **`IdentityOperator`**: Identity operator

### 2. Simulation Engine (`lchsimulation.py`)

- **`LCHSimulation`**: Main simulation class
  - `simulation_circuit()`: Generate quantum circuit for evolution
  - `state_simulation_evolve()`: State vector evolution (large time steps)
  - `matrix_simulation_evolve()`: Matrix-based evolution (small time steps)
  - `hermitian_check()`: Verify operator Hermiticity

### 3. Spatial Varying Parameters (`svp_op.py`)

- **`binary_coord()`**: Convert coordinates to binary strings
- **`qma_grouping()`**: Quine-McCluskey algorithm for term minimization
- **`bit2op()`**: Convert binary patterns to operator dictionaries

## Operator Notation

The framework uses a symbolic notation for operators:

- **`I`**: Identity operator
- **`R`**: Ladder operator (raising) - σ₀₁
- **`L`**: Ladder operator (lowering) - σ₁₀  
- **`U`**: Projector operator - σ₀₀
- **`D`**: Projector operator - σ₁₁

For example, `'IRLDU': 2.0` represents:
```
2.0 × I ⊗ R ⊗ L ⊗ D ⊗ U
```

## Examples

### 1D Heat Equation

```python
# 1D Laplacian with periodic boundary conditions
H = LaplacianOperator1D(num_bits=4, h=1.0, bc={'0': ('P',)})
lchs = LCHSimulation(op=H)

# Initial condition: Gaussian pulse
initial_state = create_gaussian_pulse(4, center=8, width=2)
evolution = lchs.matrix_simulation_evolve(T=10.0, state=initial_state)
```

### 2D Acoustic Wave

```python
# 2D acoustic wave with spatial varying parameters
H = acoustic_wave_hamiltonian(nx=5, ny=5, h=1.0, c_dict=spatial_params)
lchs = LCHSimulation(op=H)

# Point source initial condition
initial_state = create_point_source(nx=5, ny=5, position=(2, 2))
evolution = lchs.matrix_simulation_evolve(T=20.0, state=initial_state)
```

## Visualization

The framework includes plotting utilities for simulation results:

```python
import matplotlib.pyplot as plt

# Plot 2D snapshots at different times
idx = [0, 10, 20]
zs = [evolution[int(t/dt), :2**(n-2)].real.reshape(2**ny, 2**nx) for t in idx]

fig, axes = plt.subplots(1, len(idx), figsize=(16, 4.5))
for ax, z, t in zip(axes, zs, idx):
    im = ax.imshow(z, origin='lower', cmap='bwr')
    ax.set_title(f't = {t}')
plt.show()
```

## Dependencies

- **Qiskit**: Quantum circuit construction and simulation
- **NumPy**: Numerical computations
- **SymPy**: Symbolic mathematics
- **Matplotlib**: Visualization
- **SciPy**: Scientific computing utilities
