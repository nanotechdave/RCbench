# RCbench - Reservoir Computing Benchmark Toolkit

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-blue)
![Version](https://img.shields.io/badge/version-0.1.50-green)

**RCbench (Reservoir Computing Benchmark Toolkit)** is a comprehensive Python package for evaluating and benchmarking reservoir computing systems. It provides standardized tasks, flexible visualization tools, and efficient evaluation methods for both physical and simulated reservoirs.

## Features

RCbench provides a complete suite of benchmark tasks and evaluation tools:

### **Benchmark Tasks**

- **NLT (Nonlinear Transformation):** Evaluate reservoir performance on standard nonlinear transformations (square wave, phase shift, double frequency, triangular wave)
- **NARMA (Nonlinear Auto-Regressive Moving Average):** Test with NARMA models of different orders (NARMA-2, NARMA-10, etc.)
- **Memory Capacity:** Measure short and long-term memory capabilities with linear memory capacity evaluation
- **Nonlinear Memory:** Map the memory-nonlinearity trade-off using `y(t) = sin(ν * s(t-τ))` benchmark
- **Information Processing Capacity (IPC):** Comprehensive capacity framework from Dambre et al. (2012) measuring both linear memory and nonlinear computational capacity using Legendre polynomials
- **Sin(x) Approximation:** Assess reservoir ability to transform a random signal into sin(x)
- **Kernel Rank:** Evaluate the effective dimensionality and nonlinearity of the reservoir (based on Wringe et al. 2025)
- **Generalization Rank:** Assess the generalization capabilities across different datasets

### **Framework Integration**

- **ReservoirPy Integration:** Examples showing how to benchmark Echo State Networks created with ReservoirPy
- **PRCpy Integration:** Examples demonstrating integration with the Physical Reservoir Computing Python package
- **Performance Benchmarking:** Tools for measuring computational time of each evaluator

### **Advanced Visualization**

- Task-specific plotters with customizable configurations
- General reservoir property visualization (input signals, output responses, nonlinearity)
- Frequency domain analysis of reservoir behavior
- Target vs. prediction comparison with proper time alignment
- Heatmaps for capacity matrices and parameter sweeps
- IPC capacity decomposition plots (linear memory vs nonlinear capacity)
  
### **Efficient Data Handling**

- Automatic measurement loading and parsing with `ElecResDataset` and `ReservoirDataset`
- Support for various experimental data formats (CSV, whitespace-separated)
- Automatic node classification (input, ground, computation nodes)
- Feature selection and dimensionality reduction tools (PCA, k-best)

### **Flexible Evaluation Framework**

- Base evaluator class with common functionality
- Support for Ridge and Linear regression models
- Multiple metrics (NMSE, RNMSE, MSE, Capacity)
- Configurable train/test splits
- Reproducible results with random state control

---

## Project Structure

```plaintext
RCbench/
├── rcbench/
│   ├── __init__.py
│   ├── examples/                    # Example scripts
│   │   ├── example_nlt.py           # NLT with real data
│   │   ├── example_nlt_matrix.py    # NLT with synthetic data
│   │   ├── example_NARMA.py         # NARMA with real data
│   │   ├── example_NARMA_matrix.py  # NARMA with synthetic data
│   │   ├── example_sinx.py          # Sin(x) with real data
│   │   ├── example_sinx_matrix.py   # Sin(x) with synthetic data
│   │   ├── example_MC.py            # Memory Capacity with real data
│   │   ├── example_MC_matrix.py     # Memory Capacity with synthetic data
│   │   ├── example_nonlinearmemory.py        # Nonlinear Memory with real data
│   │   ├── example_nonlinearmemory_matrix.py # Nonlinear Memory with synthetic data
│   │   ├── example_ipc.py           # Information Processing Capacity
│   │   ├── example_KR.py            # Kernel Rank
│   │   ├── example_KR_matrix.py     # Kernel Rank with synthetic data
│   │   ├── example_reservoirpy.py   # ReservoirPy ESN integration
│   │   ├── example_prcpy.py         # PRCpy integration
│   │   ├── benchmark_evaluators.py  # Performance benchmarking
│   │   └── examplePCA.py            # Feature selection example
│   ├── measurements/                # Data handling
│   │   ├── __init__.py
│   │   ├── dataset.py               # ReservoirDataset and ElecResDataset classes
│   │   ├── loader.py                # MeasurementLoader for data loading
│   │   └── parser.py                # MeasurementParser for node identification
│   ├── tasks/                       # Benchmark tasks
│   │   ├── __init__.py
│   │   ├── baseevaluator.py         # Base evaluation methods
│   │   ├── featureselector.py       # Feature selection utilities
│   │   ├── nlt.py                   # Nonlinear Transformation task
│   │   ├── narma.py                 # NARMA task
│   │   ├── memorycapacity.py        # Memory Capacity task
│   │   ├── nonlinearmemory.py       # Nonlinear Memory benchmark
│   │   ├── ipc.py                   # Information Processing Capacity
│   │   ├── sinx.py                  # Sin(x) approximation task
│   │   ├── kernelrank.py            # Kernel Rank evaluation
│   │   ├── generalizationrank.py    # Generalization Rank evaluation
│   │   └── taskManuals/             # Task documentation
│   │       ├── NONLINEARMEMORY_README.md
│   │       └── IPC_README.md
│   ├── visualization/               # Plotting utilities
│   │   ├── __init__.py
│   │   ├── base_plotter.py          # Base plotting functionality
│   │   ├── plot_config.py           # Configuration classes for all plotters
│   │   ├── nlt_plotter.py           # NLT visualization
│   │   ├── narma_plotter.py         # NARMA visualization
│   │   ├── sinx_plotter.py          # Sin(x) visualization
│   │   └── mc_plotter.py            # Memory Capacity visualization
│   ├── classes/                     # Core classes
│   │   ├── __init__.py
│   │   ├── Measurement.py           # Measurement data structures
│   │   └── sample.py                # Sample handling
│   ├── utils/                       # Utility functions
│   │   ├── __init__.py
│   │   └── utils.py
│   └── logger.py                    # Logging utilities
├── tests/                           # Test suite
│   ├── test_nlt_dataset.py          # NLT evaluation tests
│   ├── test_reservoir_dataset_consistency.py
│   ├── test_electrode_selection.py
│   └── test_files/                  # Test data files
├── setup.py                         # Package setup
├── pytest.ini                       # Pytest configuration
└── README.md                        # This file
```

---

## Installation

### Install from PyPI:

```bash
pip install rcbench
```

### Install directly from GitHub:

```bash
pip install git+https://github.com/nanotechdave/RCbench.git
```

### Install locally (development mode):

```bash
git clone https://github.com/nanotechdave/RCbench.git
cd RCbench
pip install -e .
```

### Dependencies

RCbench requires:
- Python >= 3.9
- numpy
- scipy
- matplotlib
- scikit-learn
- pandas

Optional dependencies for integration examples:
- reservoirpy (for ReservoirPy integration)
- prcpy (for PRCpy integration)

---

## Quick Start

### Example 1: NLT Evaluation with Real Data

```python
from rcbench import ElecResDataset, NltEvaluator

# Load measurement data
dataset = ElecResDataset("your_measurement_file.txt")

# Get input signal and node outputs
input_signal = dataset.get_input_voltages()[dataset.input_nodes[0]]
nodes_output = dataset.get_node_voltages()

# Create evaluator
evaluator = NltEvaluator(
    input_signal=input_signal,
    nodes_output=nodes_output,
    time_array=dataset.time
)

# Run evaluation
result = evaluator.run_evaluation(target_name='square_wave')

print(f"NMSE: {result['accuracy']:.6f}")
```

### Example 2: Memory Capacity Evaluation

```python
from rcbench import ElecResDataset, MemoryCapacityEvaluator, MCPlotConfig

# Load data
dataset = ElecResDataset("your_measurement_file.txt")
input_voltages = dataset.get_input_voltages()
nodes_output = dataset.get_node_voltages()

input_signal = input_voltages[dataset.input_nodes[0]]
node_names = dataset.nodes

# Create evaluator
plot_config = MCPlotConfig(
    plot_mc_curve=True,
    plot_predictions=True,
    plot_total_mc=True
)

evaluator = MemoryCapacityEvaluator(
    input_signal=input_signal,
    nodes_output=nodes_output,
    max_delay=30,
    node_names=node_names,
    plot_config=plot_config
)

# Calculate total memory capacity
results = evaluator.calculate_total_memory_capacity(
    feature_selection_method='pca',
    num_features='all',
    modeltype='Ridge',
    regression_alpha=0.1,
    train_ratio=0.8
)

print(f"Total Memory Capacity: {results['total_memory_capacity']:.4f}")

# Generate plots
evaluator.plot_results()
```

### Example 3: Information Processing Capacity (IPC)

```python
import numpy as np
from rcbench import IPCEvaluator, IPCPlotConfig

# Generate random input (must be uniform in [-1, 1])
np.random.seed(42)
input_signal = np.random.uniform(-1, 1, size=5000)

# Your reservoir states (shape: [time_steps, n_nodes])
reservoir_states = ...  # Your reservoir output

# Create plot configuration
plot_config = IPCPlotConfig(
    save_dir='./results',
    show_plot=True,
    plot_capacity_by_degree=True,
    plot_tradeoff=True,
    plot_summary=True
)

# Create evaluator
evaluator = IPCEvaluator(
    input_signal=input_signal,
    nodes_output=reservoir_states,
    max_delay=10,           # Maximum delay to consider
    max_degree=3,           # Maximum polynomial degree
    random_state=42,
    plot_config=plot_config
)

# Calculate total information processing capacity
results = evaluator.calculate_ipc(
    feature_selection_method='pca',
    num_features='all',
    modeltype='Ridge',
    regression_alpha=0.1,
    train_ratio=0.8
)

print(f"Total Capacity: {results['total_capacity']:.4f}")
print(f"Linear Memory: {results['linear_memory_capacity']:.4f}")
print(f"Nonlinear Capacity: {results['nonlinear_capacity']:.4f}")

# Generate plots
evaluator.plot_results()
```

### Example 4: ReservoirPy Integration

```python
import numpy as np
from reservoirpy.nodes import Reservoir
from rcbench import MemoryCapacityEvaluator, SinxEvaluator, MCPlotConfig

# Create reservoir using ReservoirPy
reservoir = Reservoir(
    units=100,
    lr=0.3,           # Leak rate
    sr=0.9,           # Spectral radius
    input_scaling=0.1
)

# Generate input signal
np.random.seed(42)
n_samples = 5000
input_signal = np.random.uniform(-1, 1, n_samples)

# Run reservoir
states = reservoir.run(input_signal.reshape(-1, 1))
nodes_output = np.array(states)

# Benchmark with Memory Capacity
mc_evaluator = MemoryCapacityEvaluator(
    input_signal=input_signal,
    nodes_output=nodes_output,
    max_delay=30,
    plot_config=MCPlotConfig(show_plot=True)
)

mc_results = mc_evaluator.calculate_total_memory_capacity(
    feature_selection_method='pca',
    modeltype='Ridge',
    train_ratio=0.8
)

print(f"Total Memory Capacity: {mc_results['total_memory_capacity']:.4f}")
```

### Example 5: Kernel Rank Evaluation

```python
from rcbench.tasks.kernelrank import KernelRankEvaluator
from rcbench.tasks.generalizationrank import GeneralizationRankEvaluator

# Using combined mode (recommended per Wringe et al. 2025)
kr_evaluator = KernelRankEvaluator(
    nodes_output=nodes_output,
    input_signal=input_signal,  # Include input for combined dynamics
    kernel='linear',
    threshold=1e-6
)

kr_results = kr_evaluator.run_evaluation()
print(f"Kernel Rank: {kr_results['kernel_rank']}")
print(f"Features: {kr_results['n_features']}")

# Generalization Rank
gr_evaluator = GeneralizationRankEvaluator(nodes_output, threshold=1e-6)
gr_results = gr_evaluator.run_evaluation()
print(f"Generalization Rank: {gr_results['generalization_rank']}")
```

### Example 6: Performance Benchmarking

```python
# Run the benchmark script to measure evaluator computation times
# python rcbench/examples/benchmark_evaluators.py --runs 3

# The script measures execution time for:
# - MemoryCapacityEvaluator
# - SinxEvaluator
# - NltEvaluator
# - NarmaEvaluator
# - NonlinearMemoryEvaluator
# - IPCEvaluator
# - KernelRankEvaluator
# - GeneralizationRankEvaluator
```

---

## Available Benchmark Tasks

### 1. Nonlinear Transformation (NLT)
Evaluates the reservoir's ability to perform various nonlinear transformations:
- Square wave generation
- Phase-shifted signals (π/2)
- Frequency doubling
- Triangular wave generation

**Key Parameters:**
- `waveform_type`: 'sine' or 'triangular'
- `metric`: 'NMSE', 'RNMSE', or 'MSE'

### 2. NARMA (Nonlinear Auto-Regressive Moving Average)
Tests temporal and nonlinear processing with NARMA time series:

**NARMA-N:** `y[t+1] = α·y[t] + β·y[t]·Σy[t-i] + γ·u[t-N]·u[t] + δ`

**NARMA-2:** `y[t] = α·y[t-1] + β·y[t-1]·y[t-2] + γ·(u[t-1])³ + δ`

**Key Parameters:**
- `order`: Order of the NARMA system (2, 10, etc.)
- `alpha, beta, gamma, delta`: NARMA coefficients

### 3. Memory Capacity
Measures the reservoir's ability to recall past inputs:

**Task:** Predict `y(t) = s(t - τ)` for various delays τ

**Output:** Total memory capacity (sum of squared correlations across delays)

**Key Parameters:**
- `max_delay`: Maximum delay to test

### 4. Nonlinear Memory Benchmark
Maps the memory-nonlinearity trade-off surface:

**Task:** `y(t) = sin(ν · s(t - τ))`

**Parameters:**
- `τ (tau)`: Delay (tests memory depth)
- `ν (nu)`: Nonlinearity strength

**Output:** Capacity matrix C(τ, ν) revealing trade-offs

### 5. Information Processing Capacity (IPC)
Comprehensive capacity framework from Dambre et al. (2012):

**Task:** Measure capacity using Legendre polynomial basis functions

**Output:**
- Total capacity (bounded by N nodes)
- Linear memory capacity (degree-1 polynomials)
- Nonlinear capacity (degree > 1 polynomials)
- Capacity decomposition by delay and polynomial degree

**Reference:** Dambre et al., "Information Processing Capacity of Dynamical Systems", Scientific Reports 2, 514 (2012)

### 6. Sin(x) Approximation
Tests ability to compute nonlinear functions:

**Task:** Transform input signal x to sin(x)

### 7. Kernel Rank
Evaluates the effective dimensionality and nonlinearity of the reservoir's kernel matrix.

**Implementation:** Based on Wringe et al. (2025) "Reservoir Computing Benchmarks: a tutorial review and critique"

**Features:**
- Combined mode: Includes input signal with reservoir states
- Nodes-only mode: Uses only reservoir node outputs
- Linear and RBF kernel options
- Efficient N×N computation (instead of T×T) for linear kernels

### 8. Generalization Rank
Assesses how well the reservoir generalizes across similar datasets using SVD-based rank computation.

---

## Visualization System

RCbench features a unified visualization system with task-specific plotters:

### Configuration Classes

Each task has a configuration class to control plotting:

```python
from rcbench import (
    NLTPlotConfig,
    MCPlotConfig,
    NarmaPlotConfig,
    SinxPlotConfig,
    NonlinearMemoryPlotConfig,
    IPCPlotConfig
)

# Example: IPC configuration
plot_config = IPCPlotConfig(
    figsize=(10, 6),
    dpi=100,
    save_dir="./plots",
    show_plot=True,
    
    # IPC-specific plots
    plot_capacity_by_degree=True,
    plot_tradeoff=True,
    plot_summary=True,
    
    # General plots
    plot_input_signal=True,
    plot_output_responses=True,
    plot_nonlinearity=True,
    
    train_ratio=0.8
)
```

### Generated Plots

For each task, RCbench can generate:

1. **General Reservoir Properties:**
   - Input signal time series
   - Node output responses
   - Input-output nonlinearity scatter plots
   - Frequency spectrum analysis

2. **Task-Specific Visualizations:**
   - NLT: Target transformations and predictions
   - Memory Capacity: MC vs delay curves, cumulative MC
   - Nonlinear Memory: Capacity heatmaps, trade-off curves
   - IPC: Capacity by degree, memory-nonlinearity trade-off
   - NARMA: Time series predictions with error analysis

---

## Dataset Classes

### ElecResDataset

For electrical reservoir computing measurements:

```python
from rcbench import ElecResDataset

dataset = ElecResDataset(
    source="measurement_file.txt",  # or pandas DataFrame
    time_column='Time[s]',
    ground_threshold=1e-2,
    input_nodes=None,  # Auto-detect or force specific nodes
    ground_nodes=None,
    nodes=None
)

# Access data
input_voltages = dataset.get_input_voltages()  # Dict[str, np.ndarray]
ground_voltages = dataset.get_ground_voltages()
node_voltages = dataset.get_node_voltages()    # np.ndarray

# Node information
print(dataset.input_nodes)  # List of input node names
print(dataset.ground_nodes)  # List of ground node names
print(dataset.nodes)         # List of computation node names

# Summary
summary = dataset.summary()
```

### ReservoirDataset

General parent class for any reservoir data:

```python
from rcbench import ReservoirDataset

dataset = ReservoirDataset(
    source="data_file.csv",
    time_column='Time[s]'
)

time = dataset.time
dataframe = dataset.dataframe
```

---

## Feature Selection

RCbench includes flexible feature selection:

```python
# In any evaluator
result = evaluator.run_evaluation(
    feature_selection_method='kbest',  # or 'pca', 'none'
    num_features=10,  # or 'all'
    # ... other parameters
)

# Access selected features
selected_features = result['selected_features']  # Indices
selected_names = evaluator.selected_feature_names  # Names
```

**Available Methods:**
- `'kbest'`: Select k best features using f_regression
- `'pca'`: Principal Component Analysis
- `'none'`: Use all features

---

## Logging

RCbench uses a custom logger with different levels:

```python
from rcbench.logger import get_logger
import logging

logger = get_logger(__name__)
logger.setLevel(logging.INFO)  # INFO, DEBUG, WARNING, ERROR

logger.info("Information message")
logger.output("Output result (level 25)")
logger.warning("Warning message")
logger.error("Error message")
```

**Log Levels:**
- `OUTPUT` (25): For displaying results
- `INFO` (20): For process information
- `DEBUG` (10): For detailed debugging
- `WARNING` (30): For warnings
- `ERROR` (40): For errors

---

## Framework Integration

### ReservoirPy

RCbench integrates with [ReservoirPy](https://github.com/reservoirpy/reservoirpy) for simulating Echo State Networks:

```python
from reservoirpy.nodes import Reservoir
from rcbench import MemoryCapacityEvaluator, SinxEvaluator

# Create and run ReservoirPy reservoir
reservoir = Reservoir(units=100, lr=0.3, sr=0.9)
states = reservoir.run(input_signal.reshape(-1, 1))

# Benchmark with RCbench
evaluator = MemoryCapacityEvaluator(
    input_signal=input_signal,
    nodes_output=np.array(states),
    max_delay=30
)
```

See `rcbench/examples/example_reservoirpy.py` for a complete example.

### PRCpy

RCbench integrates with [PRCpy](https://github.com/PRCpy/PRCpy) for processing physical reservoir computing data:

```python
from prcpy.RC import Pipeline
from rcbench import MemoryCapacityEvaluator

# Process data with PRCpy pipeline
# ...

# Benchmark with RCbench
evaluator = MemoryCapacityEvaluator(
    input_signal=input_signal,
    nodes_output=readout_matrix,
    max_delay=30
)
```

See `rcbench/examples/example_prcpy.py` for a complete example.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

### Development Setup

```bash
git clone https://github.com/nanotechdave/RCbench.git
cd RCbench
pip install -e ".[dev,test]"
```

### Running Tests

```bash
pytest tests/
```

---

## Documentation

For detailed documentation on specific tasks:
- See `rcbench/tasks/taskManuals/NONLINEARMEMORY_README.md` for the Nonlinear Memory benchmark
- See `rcbench/tasks/taskManuals/IPC_README.md` for the Information Processing Capacity evaluator
- Check example scripts in `rcbench/examples/` for usage patterns
- Each task class includes comprehensive docstrings

---

## Issues & Support

- **Issue Tracker:** https://github.com/nanotechdave/RCbench/issues
- **Pull Requests:** https://github.com/nanotechdave/RCbench/pulls
- **Discussions:** https://github.com/nanotechdave/RCbench/discussions

---

## License

RCbench is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Authors

- **Davide Pilati**  - [nanotechdave](https://github.com/nanotechdave)

---

## Acknowledgments

This work was developed in collaboration between the Italian National Institute of Metrology (INRiM), Polytechnic of Turin and University of Pisa. This work was funded by the Europen Union (ERC, "MEMBRAIN", No. 101160604), website: https://membrain-eu.com, and NEURONE, a project funded by the European Union - Next Generation EU, M4C1 CUP I53D23003600006, under program PRIN 2022 (prj code 20229JRTZA). Views and opinions expressed are however those of the authors only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them.

---

## Citation

If you use RCbench in your research, please cite:

```bibtex
@article{rcbench2025,
    author={Pilati, Davide and Ceni, Andrea and Michieletti, Fabio and Gallicchio, Claudio and Ricciardi, Carlo and Milano, Gianluca},
	title={RCbench: a unified framework for benchmarking reservoir computing systems},
	journal={Neuromorphic Computing and Engineering},
	url={http://iopscience.iop.org/article/10.1088/2634-4386/ae441f},
	year={2026},
}
```

---

## References

Key papers referenced in this implementation:

1. Dambre, J., Verstraeten, D., Schrauwen, B. & Massar, S. "Information Processing Capacity of Dynamical Systems". *Scientific Reports* 2, 514 (2012).

2. Jaeger, H. "Short Term Memory in Echo State Networks". Fraunhofer Institute AIS, Tech. rep. 152 (2002).

3. Wringe, C., Trefzer, M., & Stepney, S. "Reservoir Computing Benchmarks: a tutorial review and critique". arXiv:2405.06561 (2025).

4. Dale, M., Miller, J.F., Stepney, S. & Trefzer, M.A. "A substrate-independent framework to characterize reservoir computers". *Proc. R. Soc. A* 475 (2019).

---

**Version:** 0.1.50  
**Last Updated:** January 2026
