# RCDA - Reservoir Computing Data Analysis Toolkit

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Cython](https://img.shields.io/badge/built%20with-Cython-orange)

**RCDA (Reservoir Computing Data Analysis)** is a flexible, efficient, and easy-to-use Python package designed to help you perform and analyze reservoir computing tasks. It features built-in support for key benchmarks such as Nonlinear Transformation (NLT), Memory Capacity (MC), and custom reservoir computing tasks.

## 🚀 Overview

RCDA provides:

- **Automatic measurement parsing** to easily handle experimental reservoir data.
- **Built-in evaluators** for common reservoir computing benchmarks:
  - **NLT (Nonlinear Transformation)** tasks
  - **Memory Capacity** evaluations
  - **Custom function approximation**
- **Visualization tools** to quickly assess and present results.
- **Efficient Cython-based modules** for computationally intensive operations, ensuring high performance.

---

## 📂 Project Structure

```plaintext
RCDA/
├── rcda/
│   ├── __init__.py
│   ├── measurements/      # Data loading and parsing utilities
│   ├── tasks/             # Reservoir computing tasks (NLT, Memory Capacity)
│   │   ├── nlt.py
│   │   ├── memorycapacity.py
│   │   ├── baseevaluator.py
│   │   └── fast_module.pyx  # Cython module
│   ├── visualization/     # Plotting and visualization utilities
│   │   ├── __init__.py
│   │   └── nlt_plotter.py
│   └── logger.py          # Custom logger configuration
├── tests/                 # Unit tests
├── setup.py               # Installation script
├── pyproject.toml         # Build configuration
└── README.md              # Project description and usage instructions
```
---

## 🔧 Installation

**Install directly from GitHub:**

```bash
pip install git+https://github.com/nanotechdave/RCDA.git
```


Or, install locally (development mode):

```bash
git clone https://github.com/nanotechdave/RCDA.git
cd RCDA
pip install -e .
```
Note: The package contains compiled Cython modules, which will be built automatically during installation.

## 🚦 Usage Example
Here's a quick example demonstrating how to perform an NLT evaluation:

```python
from rcda.measurements.loader import MeasurementLoader
from rcda.measurements.parser import MeasurementParser
from rcda.tasks.nlt import NltEvaluator

# Load and parse measurement data
measurement_file = 'your_measurement_file.txt'  # Update with your file path
dataset = MeasurementLoader(measurement_file).get_dataset()
parser = MeasurementParser(dataset)

# Extract signals
input_electrode = parser.input_electrodes[0]
input_signal = parser.get_input_voltages()[input_electrode]
nodes_output = parser.get_node_voltages()
time = dataset.time

# Initialize evaluator
evaluator = NltEvaluator(
    input_signal=input_signal,
    nodes_output=nodes_output,
    time_array=time,
    waveform_type='sine'  # or 'triangular'
)

# Run evaluation for a chosen target, e.g., 'square_wave'
result = evaluator.run_evaluation(
    target_name='square_wave',
    metric='NMSE',
    plot=True
)

print(f"NLT square wave accuracy: {result['accuracy']:.4f}")
```

## 📈 Visualization Tools
RCDA includes built-in visualization functions to help you inspect your results:

- Cycle-based alignment plots for NLT tasks.

- Memory Capacity curves for analyzing reservoir memory.

- Custom waveform visualization for validating data and model predictions.

## ⚡️ High Performance (Cython Integration)
RCDA leverages Cython to accelerate performance-critical operations:
```python
from rcda.tasks.fast_module import sum_of_squares

print(sum_of_squares(1000000))  # This function runs significantly faster than its pure Python counterpart.
```

## 📝 Contributions & Issues
Contributions are welcome! Please open a pull request or an issue on GitHub.

- Issue Tracker: https://github.com/nanotechdave/RCDA/issues

- Pull Requests: https://github.com/nanotechdave/RCDA/pulls

## 📜 License
RCDA is licensed under the MIT License. See the LICENSE file for details.