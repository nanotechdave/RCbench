# NLT (Nonlinear Transformation) Task Manual

## Overview

The **NLT (Nonlinear Transformation)** task evaluates a reservoir's ability to perform various nonlinear transformations on an input signal. This benchmark tests the reservoir's capacity to generate different waveform types from a given input (typically sinusoidal).

## Task Description

### Goal
Transform an input signal into different target waveforms through nonlinear processing and linear readout.

### Available Transformations

1. **Square Wave**: Convert sine/triangle wave to square wave
2. **Phase Shifted (π/2)**: Generate a signal shifted by π/2 radians
3. **Double Frequency**: Generate a signal with twice the input frequency
4. **Triangular Wave**: Convert sine wave to triangular wave

### Mathematical Formulation

Given an input signal `s(t)` (sine or triangular wave), the task generates multiple target transformations:

- **Square Wave**: `sgn(sin(2πf·t))`
- **Phase Shifted**: `sin(2πf·t + π/2)`
- **Double Frequency**: `sin(4πf·t)`
- **Triangular Wave**: Piecewise linear approximation

The reservoir's readout layer must learn to approximate these transformations using a linear combination of reservoir node outputs.

## Usage

### Basic Usage

```python
from rcbench import ElecResDataset, NltEvaluator
from rcbench.visualization.plot_config import NLTPlotConfig
from rcbench.logger import get_logger

logger = get_logger(__name__)

# Load data
dataset = ElecResDataset("measurement_file.txt")

# Extract signals
input_voltages = dataset.get_input_voltages()
input_signal = input_voltages[dataset.input_nodes[0]]
nodes_output = dataset.get_node_voltages()
time = dataset.time
node_names = dataset.nodes

# Create evaluator
evaluator = NltEvaluator(
    input_signal=input_signal,
    nodes_output=nodes_output,
    time_array=time,
    waveform_type='sine',  # or 'triangular'
    node_names=node_names
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
    logger.output(f"  NMSE: {result['accuracy']:.6f}")
```

### With Plotting Configuration

```python
# Create plot configuration
plot_config = NLTPlotConfig(
    save_dir="./nlt_results",
    show_plot=True,
    
    # General reservoir plots
    plot_input_signal=True,
    plot_output_responses=True,
    plot_nonlinearity=True,
    plot_frequency_analysis=True,
    
    # Task-specific plots
    plot_target_prediction=True,
    
    # Styling
    nonlinearity_plot_style='scatter',  # or 'line'
    frequency_range=(0, 50),
    prediction_sample_count=200
)

evaluator = NltEvaluator(
    input_signal=input_signal,
    nodes_output=nodes_output,
    time_array=time,
    waveform_type='sine',
    node_names=node_names,
    plot_config=plot_config
)

# Run evaluations and generate plots
evaluator.plot_results()
```

## Parameters

### Constructor Parameters

- **`input_signal`** (np.ndarray): Input signal array (sine or triangular wave)
- **`nodes_output`** (np.ndarray): Reservoir node outputs, shape `[time_steps, n_nodes]`
- **`time_array`** (np.ndarray): Time values corresponding to signals
- **`waveform_type`** (str): Type of input waveform - `'sine'` or `'triangular'`
- **`node_names`** (List[str], optional): Names of the nodes for plotting
- **`plot_config`** (NLTPlotConfig, optional): Configuration for plotting

### Evaluation Parameters

- **`target_name`** (str): Name of target transformation to evaluate
  - Available: `'square_wave'`, `'pi_half_shifted'`, `'double_frequency'`, `'triangular_wave'`
- **`metric`** (str): Evaluation metric
  - `'NMSE'` - Normalized Mean Squared Error (default)
  - `'RNMSE'` - Root Normalized Mean Squared Error
  - `'MSE'` - Mean Squared Error
- **`feature_selection_method`** (str): Method for feature selection
  - `'kbest'` - K-best features using f_regression
  - `'pca'` - Principal Component Analysis
  - `'none'` - Use all features
- **`num_features`** (int or 'all'): Number of features to use
- **`modeltype`** (str): Regression model type - `'Ridge'` or `'Linear'`
- **`regression_alpha`** (float): Regularization parameter for Ridge regression
- **`train_ratio`** (float): Ratio of data to use for training (0-1)
- **`plot`** (bool): Whether to generate plots during evaluation

## Output

### Result Dictionary

```python
{
    'accuracy': float,           # Error metric value
    'metric': str,               # Name of metric used
    'selected_features': list,   # Indices of selected features
    'model': sklearn.Model,      # Trained regression model
    'y_pred': np.ndarray,        # Predicted values
    'y_test': np.ndarray,        # True test values
    'train_ratio': float         # Train/test split ratio
}
```

### Available Targets

Access all available targets:

```python
available_targets = list(evaluator.targets.keys())
# Output: ['square_wave', 'pi_half_shifted', 'double_frequency', 'triangular_wave']
```

## Metrics

### NMSE (Normalized Mean Squared Error)
```
NMSE = MSE(y_true, y_pred) / Var(y_true)
```
- **Lower is better**
- NMSE = 0: Perfect prediction
- NMSE ≥ 1: Worse than predicting the mean

### RNMSE (Root Normalized Mean Squared Error)
```
RNMSE = sqrt(NMSE)
```
- **Lower is better**
- More interpretable scale than NMSE

### MSE (Mean Squared Error)
```
MSE = mean((y_true - y_pred)²)
```
- **Lower is better**
- Absolute error in signal units

## Visualization

### Generated Plots

1. **Input Signal**: Time series of input waveform
2. **Node Responses**: Output responses of reservoir nodes
3. **Nonlinearity**: Input-output relationship showing nonlinear transformations
4. **Frequency Analysis**: Frequency spectra of input and node outputs
5. **Target Prediction**: Comparison of true vs predicted transformations

### Customization

```python
plot_config = NLTPlotConfig(
    figsize=(12, 8),              # Figure size
    dpi=150,                      # Resolution
    save_dir="./results",         # Where to save plots
    show_plot=True,               # Display plots
    frequency_range=(0, 100),     # Hz range for frequency plots
    prediction_sample_count=500   # Samples to show in predictions
)
```

## Examples

### Example 1: Evaluate All Targets

```python
all_results = {}

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
    all_results[target_name] = result

# Find best and worst
accuracies = {k: v['accuracy'] for k in all_results}
best = min(accuracies, key=accuracies.get)
worst = max(accuracies, key=accuracies.get)

print(f"Best: {best} (NMSE: {accuracies[best]:.4f})")
print(f"Worst: {worst} (NMSE: {accuracies[worst]:.4f})")
```

### Example 2: Compare Feature Selection Methods

```python
methods = ['kbest', 'pca', 'none']
results_by_method = {}

for method in methods:
    result = evaluator.run_evaluation(
        target_name='square_wave',
        metric='NMSE',
        feature_selection_method=method,
        num_features=10,
        modeltype="Ridge",
        regression_alpha=0.1,
        train_ratio=0.8
    )
    results_by_method[method] = result['accuracy']

print("Feature Selection Comparison:")
for method, nmse in results_by_method.items():
    print(f"  {method}: {nmse:.6f}")
```

### Example 3: Using Synthetic Data

```python
import numpy as np

# Generate synthetic data
n_samples = 2000
t = np.linspace(0, 10, n_samples)
input_signal = np.sin(2 * np.pi * t)

# Create synthetic reservoir (simple delays and nonlinearities)
n_nodes = 15
nodes_output = np.zeros((n_samples, n_nodes))
for i in range(n_nodes):
    delay = np.random.randint(1, 5)
    nodes_output[:, i] = np.tanh(2 * np.roll(input_signal, delay))

# Evaluate
evaluator = NltEvaluator(
    input_signal=input_signal,
    nodes_output=nodes_output,
    time_array=t,
    waveform_type='sine'
)

result = evaluator.run_evaluation(
    target_name='square_wave',
    metric='NMSE',
    modeltype="Ridge",
    regression_alpha=0.1
)

print(f"Square wave NMSE: {result['accuracy']:.6f}")
```

## Interpretation Guide

### Performance Benchmarks

- **NMSE < 0.1**: Excellent performance
- **0.1 ≤ NMSE < 0.3**: Good performance
- **0.3 ≤ NMSE < 0.5**: Moderate performance
- **NMSE ≥ 0.5**: Poor performance
- **NMSE ≈ 1.0**: Reservoir provides no benefit (baseline)

### Expected Difficulty

Transformations ranked by difficulty (typical):

1. **Triangular Wave**: Easiest (piecewise linear)
2. **Phase Shift**: Easy (linear phase transformation)
3. **Square Wave**: Moderate (requires sharp transitions)
4. **Double Frequency**: Hardest (requires higher-order nonlinearity)

### Common Issues

1. **High NMSE for all targets**: 
   - Check if reservoir has sufficient nonlinearity
   - Try different feature selection methods
   - Increase number of nodes

2. **Good performance on some targets, poor on others**:
   - Normal behavior - indicates reservoir specialization
   - Adjust reservoir parameters for balanced performance

3. **Overfitting (train << test error)**:
   - Increase `regression_alpha`
   - Reduce `num_features`
   - Use more training data

## Tips and Best Practices

1. **Input Scaling**: Ensure input signal is properly scaled (typically [-1, 1])

2. **Feature Selection**: Start with `'kbest'` and `num_features='all'`, then optimize

3. **Regularization**: Adjust `regression_alpha` based on dataset size:
   - Small datasets (< 1000 samples): α = 1.0 - 10.0
   - Medium datasets (1000-5000): α = 0.1 - 1.0
   - Large datasets (> 5000): α = 0.01 - 0.1

4. **Train/Test Split**: Use 80/20 split for most cases, adjust if needed

5. **Visualization**: Always check frequency analysis to ensure proper sampling

## Related Tasks

- **NARMA**: Tests temporal nonlinear processing
- **Sin(x)**: Tests function approximation
- **Nonlinear Memory**: Tests memory-nonlinearity trade-off

## References

- Standard nonlinear transformation benchmarks in reservoir computing
- Echo State Networks (Jaeger, 2001)
- Liquid State Machines (Maass et al., 2002)

## See Also

- `NARMA_README.md` - NARMA task documentation
- `MEMORYCAPACITY_README.md` - Memory capacity evaluation
- `NONLINEARMEMORY_README.md` - Nonlinear memory benchmark

