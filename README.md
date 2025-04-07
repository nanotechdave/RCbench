# RCbench - Reservoir Computing Benchmark Toolkit

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-GPLv3-blue)
![Cython](https://img.shields.io/badge/built%20with-Cython-orange)


**RCbench (Reservoir Computing Benchmark Toolkit)** is a comprehensive Python package for evaluating and benchmarking reservoir computing systems. It provides standardized tasks, flexible visualization tools, and efficient evaluation methods for both physical and simulated reservoirs.

## 🚀 Features

RCbench provides:

-**Multiple Benchmark Tasks:**
  -**NLT (Nonlinear Transformation):** Evaluate reservoir performance on standard nonlinear transformations
  -**NARMA:** Test with Nonlinear Auto-Regressive Moving Average models of different orders
  -**Memory Capacity:** Measure the short and long-term memory capabilities
  -**Sin(x):** Assess reservoir ability to approximate sine functions
-**Advanced Visualization:**
  -Task-specific plotters with customizable configurations
  -General reservoir properties visualization (input signals, output responses, nonlinearity)
  -Frequency domain analysis of reservoir behavior
  -Target vs. prediction comparison with proper time alignment
-**Efficient Data Handling:**
  -Automatic measurement loading and parsing
  -Support for various experimental data formats
  -Feature selection and dimensionality reduction tools
-**Performance Optimization:**
  -Cython-accelerated modules for computationally intensive operations

---

## 📂 Project Structure

```plaintext
RCbench/
├── rcbench/
│   ├── measurements/          # Data handling
│   │   ├── dataset.py         # ReservoirDataset class
│   │   ├── loader.py          # Data loading utilities
│   │   └── parser.py          # Data parsing utilities
│   ├── tasks/                 # Reservoir computing tasks
│   │   ├── baseevaluator.py   # Base evaluation methods
│   │   ├── nlt.py             # Nonlinear Transformation
│   │   ├── narma.py           # NARMA tasks
│   │   ├── memorycapacity.py  # Memory Capacity
│   │   ├── sinx.py            # Sin(x) approximation
│   │   └── featureselector.py # Feature selection tools
│   ├── visualization/         # Plotting utilities
│   │   ├── base_plotter.py    # Base plotting functionality
│   │   ├── plot_config.py     # Plot configurations
│   │   ├── nlt_plotter.py     # NLT visualization
│   │   ├── narma_plotter.py   # NARMA visualization
│   │   └── sinx_plotter.py    # Sin(x) visualization
│   └── logger.py              # Logging utilities
└── examples/                  # Example scripts
    ├── example_nlt.py
    ├── example_NARMA.py
    ├── example_sinx.py
    └── example_MC.py
```
---

## 🔧 Installation

**Install directly from GitHub:**

```bash
pip install git+https://github.com/nanotechdave/RCbench.git
```


Or, install locally (development mode):

```bash
git clone https://github.com/nanotechdave/RCbench.git
cd RCbench
pip install -e .
```
Note: The package contains compiled Cython modules, which will be built automatically during installation.

## 🚦 Usage Example
Here's a quick example demonstrating how to perform an NLT evaluation:

```python
from rcbench.measurements.dataset import ReservoirDataset
from rcbench.tasks.narma import NarmaEvaluator
from rcbench.visualization.plot_config import NarmaPlotConfig

# Load dataset
dataset = ReservoirDataset("your_measurement_file.txt")
electrodes_info = dataset.summary()

# Get input and node data
input_signal = dataset.get_input_voltages()[electrodes_info['input_electrodes'][0]]
nodes_output = dataset.get_node_voltages()
electrode_names = electrodes_info['node_electrodes']

# Configure plotting
plot_config = NarmaPlotConfig(
    plot_input_signal=True,
    plot_output_responses=True,
    plot_nonlinearity=True,
    plot_frequency_analysis=True,
    plot_target_prediction=True
)

# Initialize evaluator and run
evaluator = NarmaEvaluator(
    input_signal=input_signal,
    nodes_output=nodes_output,
    electrode_names=electrode_names,
    order=2,  # NARMA-2
    plot_config=plot_config
)

results = evaluator.run_evaluation(
    metric='NMSE',
    feature_selection_method='pca',
    num_features='all'
)

print(f"NARMA-2 Accuracy: {results['accuracy']:.5f}")

# Generate all plots (both general reservoir properties and task-specific)
evaluator.plot_results(existing_results=results)
```

## 📈 Visualization Tools
RCbench features a unified visualization system with:
-**Task-Specific Plotters:** Dedicated plotters for each task (NLTPlotter, NarmaPlotter, SinxPlotter)
-**Customizable Configurations:** Control which plots to generate through configuration objects
-**Comprehensive Visualization:** For each task, view:
  -General reservoir properties (input signals, node responses, nonlinearity)
  -Frequency domain analysis
  -Target vs. prediction comparisons
All plotters are built on a common base architecture, ensuring consistent styling and behavior.


## 📝 Contributions & Issues
Contributions are welcome! Please open a pull request or an issue on GitHub.

- Issue Tracker: https://github.com/nanotechdave/RCbench/issues

- Pull Requests: https://github.com/nanotechdave/RCbench/pulls

## 📜 License

RCbench is licensed under the GPLv3 License. See the LICENSE file for details.

