# RCbench - Reservoir Computing Benchmark Toolkit

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-blue)
![Version](https://img.shields.io/badge/version-0.1.20-green)

**RCbench (Reservoir Computing Benchmark Toolkit)** is a comprehensive Python package for evaluating and benchmarking reservoir computing systems. It provides standardized tasks, flexible visualization tools, and efficient evaluation methods for both physical and simulated reservoirs.

## 🚀 Features

RCbench provides a complete suite of benchmark tasks and evaluation tools:

### **Benchmark Tasks**

- **NLT (Nonlinear Transformation):** Evaluate reservoir performance on standard nonlinear transformations (square wave, phase shift, double frequency, triangular wave)
- **NARMA (Nonlinear Auto-Regressive Moving Average):** Test with NARMA models of different orders (NARMA-2, NARMA-10, etc.)
- **Memory Capacity:** Measure short and long-term memory capabilities with linear memory capacity evaluation
- **Nonlinear Memory:** Map the memory-nonlinearity trade-off using `y(t) = sin(ν * s(t-τ))` benchmark
- **Sin(x) Approximation:** Assess reservoir ability to transform a random signal into sin(x)
- **Kernel Rank:** Evaluate the nonlinearity and kernel quality of the reservoir
- **Generalization Rank:** Assess the generalization capabilities across different datasets

### **Advanced Visualization**

- Task-specific plotters with customizable configurations
- General reservoir property visualization (input signals, output responses, nonlinearity)
- Frequency domain analysis of reservoir behavior
- Target vs. prediction comparison with proper time alignment
- Heatmaps for capacity matrices and parameter sweeps
  
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

## 📂 Project Structure

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
│   │   ├── example_KR.py            # Kernel Rank
│   │   ├── example_KR_matrix.py     # Kernel Rank with synthetic data
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
│   │   ├── sinx.py                  # Sin(x) approximation task
│   │   ├── kernelrank.py            # Kernel Rank evaluation
│   │   └── generalizationrank.py    # Generalization Rank evaluation
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

## 🔧 Installation

### Install from PyPI (coming soon):

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

---

## 🚦 Quick Start

### Example 1: NLT Evaluation with Real Data

```python
import logging
from pathlib import Path
from rcbench import ElecResDataset, NltEvaluator
from rcbench.visualization.plot_config import NLTPlotConfig
from rcbench.logger import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.INFO)

# Load measurement data
measurement_file = Path("your_measurement_file.txt")
dataset = ElecResDataset(measurement_file)

# Get data from dataset
input_voltages = dataset.get_input_voltages()
nodes_output = dataset.get_node_voltages()
node_names = dataset.nodes

# Use first input node
primary_input = dataset.input_nodes[0]
input_signal = input_voltages[primary_input]
time = dataset.time

# Create plot configuration
plot_config = NLTPlotConfig(
    save_dir=None,
    plot_input_signal=True,
    plot_output_responses=True,
    plot_nonlinearity=True,
    plot_frequency_analysis=True,
    plot_target_prediction=True,
    show_plot=True
)

# Create evaluator
evaluator = NltEvaluator(
    input_signal=input_signal,
    nodes_output=nodes_output,
    time_array=time,
    waveform_type='sine',
    node_names=node_names,
    plot_config=plot_config
)

# Run evaluation for all targets
for target_name in evaluator.targets:
    result = evaluator.run_evaluation(
        target_name=target_name,
        metric='NMSE',
        feature_selection_method='kbest',
        num_features='all',
        modeltype="Ridge",
        regression_alpha=0.1,
        train_ratio=0.8
    )
    logger.output(f"Target: {target_name}")
    logger.output(f"  Accuracy (NMSE): {result['accuracy']:.6f}")

# Generate comprehensive plots
evaluator.plot_results()
```

### Example 2: Memory Capacity Evaluation

```python
from rcbench import ElecResDataset, MemoryCapacityEvaluator
from rcbench.visualization.plot_config import MCPlotConfig

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

logger.output(f"Total Memory Capacity: {results['total_memory_capacity']:.4f}")

# Generate plots
evaluator.plot_results()
```

### Example 3: Nonlinear Memory Benchmark

```python
from rcbench import ElecResDataset, NonlinearMemoryEvaluator

# Load data
dataset = ElecResDataset("your_measurement_file.txt")
input_signal = dataset.get_input_voltages()[dataset.input_nodes[0]]
nodes_output = dataset.get_node_voltages()

# Create evaluator with parameter ranges
evaluator = NonlinearMemoryEvaluator(
    input_signal=input_signal,
    nodes_output=nodes_output,
    tau_values=[1, 2, 3, 4, 5, 6, 7, 8],  # Delay values
    nu_values=[0.1, 0.3, 1.0, 3.0, 10.0],  # Nonlinearity strengths
    random_state=42,
    node_names=dataset.nodes
)

# Run parameter sweep
results = evaluator.run_parameter_sweep(
    feature_selection_method='kbest',
    num_features='all',
    modeltype='Ridge',
    regression_alpha=0.1,
    train_ratio=0.8,
    metric='NMSE'
)

# Get summary
summary = evaluator.summary()
logger.output(f"Average capacity: {summary['average_capacity']:.4f}")
logger.output(f"Best (τ, ν): ({summary['best_tau']}, {summary['best_nu']})")

# Analyze trade-offs
tradeoff = evaluator.get_memory_vs_nonlinearity_tradeoff()

# Generate plots
evaluator.plot_results(save_dir="./results")
```

### Example 4: Using Synthetic Data

```python
import numpy as np
from rcbench import NarmaEvaluator

# Generate synthetic data
n_samples = 2000
n_nodes = 15

np.random.seed(42)
input_signal = np.random.uniform(-1, 1, n_samples)

# Create synthetic reservoir responses
nodes_output = np.zeros((n_samples, n_nodes))
for i in range(n_nodes):
    delay = np.random.randint(1, 5)
    nonlinearity = 0.5 + np.random.rand() * 2
    nodes_output[:, i] = np.tanh(nonlinearity * np.roll(input_signal, delay))
    nodes_output[:, i] += np.random.randn(n_samples) * 0.05

node_names = [f'Node_{i}' for i in range(n_nodes)]

# Evaluate
evaluator = NarmaEvaluator(
    input_signal=input_signal,
    nodes_output=nodes_output,
    node_names=node_names,
    order=2
)

result = evaluator.run_evaluation(
    metric='NMSE',
    feature_selection_method='kbest',
    num_features='all',
    modeltype="Ridge",
    regression_alpha=1.0,
    train_ratio=0.8
)

logger.output(f"NARMA-{evaluator.order} Accuracy: {result['accuracy']:.6f}")
```

---

## 📊 Available Benchmark Tasks

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

### 5. Sin(x) Approximation
Tests ability to compute nonlinear functions:

**Task:** Transform input signal x to sin(x)

### 6. Kernel Rank
Evaluates the effective dimensionality and nonlinearity of the reservoir's kernel matrix.

### 7. Generalization Rank
Assesses how well the reservoir generalizes across different datasets.

---

## 🎨 Visualization System

RCbench features a unified visualization system with task-specific plotters:

### Configuration Classes

Each task has a configuration class to control plotting:

```python
from rcbench.visualization.plot_config import (
    NLTPlotConfig,
    MCPlotConfig,
    NarmaPlotConfig,
    SinxPlotConfig
)

# Example: NLT configuration
plot_config = NLTPlotConfig(
    figsize=(10, 6),
    dpi=100,
    save_dir="./plots",
    show_plot=True,
    
    # General plots
    plot_input_signal=True,
    plot_output_responses=True,
    plot_nonlinearity=True,
    plot_frequency_analysis=True,
    
    # Task-specific plots
    plot_target_prediction=True,
    
    # Styling
    nonlinearity_plot_style='scatter',
    frequency_range=(0, 50),
    prediction_sample_count=200
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
   - NARMA: Time series predictions with error analysis

---

## 🔬 Dataset Classes

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

## 🧪 Feature Selection

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

## 📝 Logging

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

## 🧑‍💻 Contributing

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

## 📚 Documentation

For detailed documentation on specific tasks:
- See `rcbench/tasks/NONLINEARMEMORY_README.md` for the Nonlinear Memory benchmark
- Check example scripts in `rcbench/examples/` for usage patterns
- Each task class includes comprehensive docstrings

---

## 🐛 Issues & Support

- **Issue Tracker:** https://github.com/nanotechdave/RCbench/issues
- **Pull Requests:** https://github.com/nanotechdave/RCbench/pulls
- **Discussions:** https://github.com/nanotechdave/RCbench/discussions

---

## 📜 License

RCbench is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Davide Pilati** - *Initial work* - [nanotechdave](https://github.com/nanotechdave)

---

## 🙏 Acknowledgments

This toolkit was developed at Politecnico di Torino for benchmarking physical reservoir computing systems, particularly nanowire networks and other unconventional computing substrates.

---

## 📖 Citation

If you use RCbench in your research, please cite:

```bibtex
@software{rcbench2025,
  author = {Pilati, Davide},
  title = {RCbench: Reservoir Computing Benchmark Toolkit},
  year = {2025},
  url = {https://github.com/nanotechdave/RCbench},
  version = {0.1.20}
}
```

---

**Version:** 0.1.20  
**Last Updated:** January 2025
