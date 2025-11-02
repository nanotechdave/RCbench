# Sin(x) Approximation Task Manual

## Overview

The **Sin(x) Approximation** task evaluates a reservoir's ability to compute nonlinear functions. Specifically, it tests whether the reservoir can transform a random input signal `x` into `sin(x)` using only linear readout.

## Task Description

### Goal
Learn a mapping from an input signal x to sin(x) through reservoir dynamics and linear readout.

### Mathematical Formulation

Given:
- Input signal: `x(t)` - random or structured signal
- Target: `y(t) = sin(x(t))`

The reservoir processes x(t) through its nonlinear dynamics, and a linear readout must learn to approximate sin(x) from the reservoir states.

### Why It's Challenging

This task requires:
1. **Nonlinear Processing**: Transform arbitrary values into sinusoidal output
2. **Universal Approximation**: Work across the entire input range
3. **Smooth Mapping**: Maintain continuity and smoothness

The key insight: A good reservoir should provide diverse enough representations that a simple linear readout can approximate complex functions.

## Usage

### Basic Usage

```python
from rcbench import ElecResDataset, SinxEvaluator
from rcbench.logger import get_logger

logger = get_logger(__name__)

# Load data
dataset = ElecResDataset("measurement_file.txt")

# Extract signals
input_voltages = dataset.get_input_voltages()
input_signal = input_voltages[dataset.input_nodes[0]]
nodes_output = dataset.get_node_voltages()
node_names = dataset.nodes

# Create evaluator
evaluator = SinxEvaluator(
    input_signal=input_signal,
    nodes_output=nodes_output,
    node_names=node_names
)

# Run evaluation
result = evaluator.run_evaluation(
    metric='NMSE',
    feature_selection_method='kbest',
    num_features=10,
    modeltype="Ridge",
    regression_alpha=1.0,
    train_ratio=0.8
)

logger.output(f"Sin(x) Approximation:")
logger.output(f"  NMSE: {result['accuracy']:.6f}")
logger.output(f"  Features used: {len(result['selected_features'])}")
```

### With Plotting

```python
from rcbench.visualization.plot_config import SinxPlotConfig

# Create plot configuration
plot_config = SinxPlotConfig(
    save_dir="./sinx_results",
    show_plot=True,
    
    # General plots
    plot_input_signal=True,
    plot_output_responses=True,
    plot_nonlinearity=True,
    plot_frequency_analysis=True,
    
    # Styling
    nonlinearity_plot_style='scatter',
    prediction_sample_count=500,
    frequency_range=(0, 50)
)

evaluator = SinxEvaluator(
    input_signal=input_signal,
    nodes_output=nodes_output,
    node_names=node_names,
    plot_config=plot_config
)

result = evaluator.run_evaluation(
    metric='NMSE',
    feature_selection_method='kbest',
    num_features=10,
    modeltype="Ridge",
    regression_alpha=1.0
)

# Generate comprehensive plots
evaluator.plot_results()
```

## Parameters

### Constructor Parameters

- **`input_signal`** (np.ndarray): Input signal (will be mapped to [0, 2π])
- **`nodes_output`** (np.ndarray): Reservoir node outputs, shape `[time_steps, n_nodes]`
- **`node_names`** (List[str], optional): Names of nodes for plotting
- **`plot_config`** (SinxPlotConfig, optional): Configuration for plotting

### Evaluation Parameters

- **`metric`** (str): Evaluation metric
  - `'NMSE'` - Normalized Mean Squared Error (default)
  - `'RNMSE'` - Root Normalized Mean Squared Error
  - `'MSE'` - Mean Squared Error
- **`feature_selection_method`** (str): Feature selection method
  - `'kbest'` - K-best features (recommended)
  - `'pca'` - Principal Component Analysis
  - `'none'` - Use all features
- **`num_features`** (int or 'all'): Number of features to use (default: 10)
- **`modeltype`** (str): Regression model - `'Ridge'` or `'Linear'`
- **`regression_alpha`** (float): Regularization parameter (default: 1.0)
- **`train_ratio`** (float): Training data ratio (default: 0.8)

## Output

### Result Dictionary

```python
{
    'accuracy': float,           # Error metric value
    'metric': str,               # Metric used
    'selected_features': list,   # Indices of selected features
    'model': sklearn.Model,      # Trained regression model
    'y_pred': np.ndarray,        # Predicted sin(x) values
    'y_test': np.ndarray,        # True sin(x) values
    'train_ratio': float         # Train/test split
}
```

### Target Access

```python
# Access generated sin(x) target
sinx_target = evaluator.target

# Access normalized input (mapped to [0, 2π])
normalized_input = evaluator._normalize_input(input_signal)
```

## Input Normalization

The task automatically normalizes the input signal to the range [0, 2π]:

```python
x_normalized = 2π * (x - x_min) / (x_max - x_min)
target = sin(x_normalized)
```

This ensures:
- Full coverage of sine wave period
- Consistent difficulty across different input ranges
- Standard comparison across datasets

## Metrics

### NMSE (Normalized Mean Squared Error)

```
NMSE = MSE(y_true, y_pred) / Var(y_true)
```

For sin(x), Var(sin(x)) ≈ 0.5, so:
- **NMSE < 0.05**: Excellent approximation
- **0.05 ≤ NMSE < 0.15**: Good approximation
- **0.15 ≤ NMSE < 0.30**: Moderate approximation
- **NMSE ≥ 0.30**: Poor approximation

### RNMSE (Root Normalized Mean Squared Error)

```
RNMSE = sqrt(NMSE)
```

Roughly indicates fractional error:
- RNMSE = 0.1 → ~10% error
- RNMSE = 0.2 → ~20% error

### MSE (Mean Squared Error)

```
MSE = mean((y_true - y_pred)²)
```

Absolute error, typically in range [0, 2] for sin(x).

## Visualization

### Generated Plots

1. **Input Signal**: Original and normalized input
2. **Node Responses**: Reservoir node activations
3. **Nonlinearity**: Scatter plot showing x vs reservoir outputs
4. **Frequency Analysis**: Spectral content
5. **Sin(x) Prediction**: True sin(x) vs predicted values
6. **Function Approximation**: y_pred vs y_true scatter

### Plot Configuration

```python
plot_config = SinxPlotConfig(
    figsize=(10, 6),
    dpi=150,
    save_dir="./results",
    show_plot=True,
    
    # Control plots
    plot_input_signal=True,
    plot_output_responses=True,
    plot_nonlinearity=True,
    plot_frequency_analysis=True,
    
    # Styling
    nonlinearity_plot_style='scatter',  # or 'line'
    prediction_sample_count=500,
    frequency_range=(0, 100)
)
```

## Examples

### Example 1: Basic Sin(x) Approximation

```python
evaluator = SinxEvaluator(
    input_signal=input_signal,
    nodes_output=nodes_output
)

result = evaluator.run_evaluation(
    metric='NMSE',
    feature_selection_method='kbest',
    num_features=10,
    modeltype="Ridge",
    regression_alpha=1.0
)

print(f"Sin(x) NMSE: {result['accuracy']:.6f}")
```

### Example 2: Optimize Number of Features

```python
feature_counts = [3, 5, 10, 15, 20, 'all']
nmse_values = []

evaluator = SinxEvaluator(
    input_signal=input_signal,
    nodes_output=nodes_output
)

for n_features in feature_counts:
    result = evaluator.run_evaluation(
        metric='NMSE',
        feature_selection_method='kbest',
        num_features=n_features,
        modeltype="Ridge",
        regression_alpha=1.0
    )
    nmse_values.append(result['accuracy'])
    print(f"Features: {str(n_features):>3}, NMSE: {result['accuracy']:.6f}")

# Find optimal
best_idx = np.argmin(nmse_values)
print(f"\nOptimal: {feature_counts[best_idx]} features")
```

### Example 3: Compare Regression Models

```python
models = ['Linear', 'Ridge']
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]

for model in models:
    print(f"\n{model} Regression:")
    
    if model == 'Linear':
        result = evaluator.run_evaluation(
            metric='NMSE',
            modeltype=model,
            num_features=10
        )
        print(f"  NMSE: {result['accuracy']:.6f}")
    else:
        for alpha in alphas:
            result = evaluator.run_evaluation(
                metric='NMSE',
                modeltype=model,
                regression_alpha=alpha,
                num_features=10
            )
            print(f"  α={alpha:6.2f}: NMSE = {result['accuracy']:.6f}")
```

### Example 4: Using Synthetic Data

```python
import numpy as np

# Generate random input
n_samples = 2000
np.random.seed(42)
input_signal = np.random.uniform(-2, 2, n_samples)

# Create synthetic reservoir with nonlinear transformations
n_nodes = 18
nodes_output = np.zeros((n_samples, n_nodes))

for i in range(n_nodes):
    # Each node applies different nonlinear transformation
    if i % 3 == 0:
        nodes_output[:, i] = np.tanh(input_signal * (i+1) * 0.5)
    elif i % 3 == 1:
        nodes_output[:, i] = input_signal ** 2 * np.sign(input_signal)
    else:
        nodes_output[:, i] = np.tanh(input_signal) * input_signal
    
    # Add small noise
    nodes_output[:, i] += np.random.randn(n_samples) * 0.03

# Evaluate
evaluator = SinxEvaluator(
    input_signal=input_signal,
    nodes_output=nodes_output
)

result = evaluator.run_evaluation(
    metric='NMSE',
    feature_selection_method='kbest',
    num_features=10,
    modeltype="Ridge",
    regression_alpha=1.0
)

print(f"Synthetic Reservoir Sin(x) NMSE: {result['accuracy']:.6f}")
```

### Example 5: Feature Selection Comparison

```python
methods = ['kbest', 'pca', 'none']

for method in methods:
    result = evaluator.run_evaluation(
        metric='NMSE',
        feature_selection_method=method,
        num_features=10 if method != 'none' else 'all',
        modeltype="Ridge",
        regression_alpha=1.0
    )
    print(f"{method:>6}: NMSE = {result['accuracy']:.6f}")
```

## Interpretation Guide

### Performance Benchmarks

- **NMSE < 0.05**: Excellent - near-perfect approximation
- **0.05 ≤ NMSE < 0.15**: Good - useful approximation
- **0.15 ≤ NMSE < 0.30**: Moderate - rough approximation
- **NMSE ≥ 0.30**: Poor - insufficient nonlinearity

### What Good Performance Indicates

1. **Rich Nonlinear Dynamics**: Reservoir can generate diverse nonlinear transformations
2. **Sufficient Dimensionality**: Enough distinct features for function approximation
3. **Appropriate Nonlinearity**: Not too linear, not too chaotic

### Common Patterns

1. **Perfect Linear Reservoir**: NMSE ≈ 0.5-1.0 (can't approximate sine)
2. **Optimal Nonlinear Reservoir**: NMSE ≈ 0.01-0.10
3. **Too Chaotic**: NMSE variable, poor generalization

## Tips and Best Practices

### 1. Input Signal

- Use **random uniform** or **random normal** distribution
- Cover wide range to test full function
- Length: 1000-3000 samples recommended

### 2. Feature Selection

- Start with **k-best** and 10 features
- Optimal typically: 5-20 features
- Too many features → overfitting
- Too few features → underfitting

### 3. Regularization

- Ridge regression recommended
- Start with α = 1.0
- Increase if overfitting (α = 10-100)
- Decrease for large datasets (α = 0.1-1.0)

### 4. Number of Nodes

- Minimum: ~10 nodes
- Recommended: 15-30 nodes
- More nodes → more diverse features

### 5. Nonlinearity

- Reservoir must have nonlinear activation
- Without nonlinearity: impossible to approximate sine
- Check node responses show varied patterns

### 6. Troubleshooting

**If NMSE > 0.3:**
- Check reservoir has nonlinearity
- Verify nodes show diverse responses
- Try more features
- Check input covers wide range
- Ensure sufficient training data

## Comparison with Other Functions

The Sin(x) task can be modified to test other functions:

### Easy Functions
- `y = x²` (easier than sine)
- `y = |x|` (piecewise linear)

### Similar Difficulty
- `y = cos(x)` (similar to sine)
- `y = tanh(x)` (bounded like sine)

### Harder Functions
- `y = x³` (requires higher-order terms)
- `y = exp(x)` (unbounded, numerical issues)

## Relationship to Other Tasks

### Sin(x) vs NLT

- **Sin(x)**: Function approximation (x → sin(x))
- **NLT**: Waveform generation (sine → square, etc.)
- Sin(x) tests instantaneous nonlinear mapping
- NLT tests temporal pattern generation

### Sin(x) vs NARMA

- **Sin(x)**: Memoryless nonlinear function
- **NARMA**: Temporal nonlinear dynamics
- Sin(x) is simpler (no memory required)

### Sin(x) vs Kernel Rank

- **Sin(x)**: Task-specific performance
- **Kernel Rank**: General nonlinearity measure
- Good Kernel Rank usually → good Sin(x) performance

## Advanced Usage

### Custom Target Function

Modify the evaluator to test different functions:

```python
# After creating evaluator, replace target
custom_target = np.tanh(2 * evaluator.input_signal)  # Test tanh instead
evaluator.target = custom_target

result = evaluator.run_evaluation(
    metric='NMSE',
    modeltype="Ridge",
    regression_alpha=1.0
)
```

### Analyze Feature Contributions

```python
result = evaluator.run_evaluation(
    metric='NMSE',
    feature_selection_method='kbest',
    num_features=10,
    modeltype="Ridge",
    regression_alpha=1.0
)

# Get model coefficients
coefficients = result['model'].coef_
selected_features = result['selected_features']

print("Feature contributions:")
for i, (feat_idx, coef) in enumerate(zip(selected_features, coefficients)):
    node_name = node_names[feat_idx]
    print(f"  {node_name}: {coef:.4f}")
```

## Common Issues

### Issue 1: Very High NMSE (> 0.5)

**Possible Causes:**
- Reservoir is too linear
- Insufficient nodes
- Poor feature selection

**Solutions:**
- Verify reservoir has nonlinear elements
- Increase number of nodes
- Try different feature selection methods

### Issue 2: Good Training, Poor Test

**Cause:** Overfitting

**Solutions:**
- Increase regularization (α)
- Reduce number of features
- Use more training data

### Issue 3: Unstable Results

**Causes:**
- Insufficient training data
- Poor train/test split
- Extreme regularization

**Solutions:**
- Use more samples
- Set random seed for reproducibility
- Adjust regularization

## References

- Jaeger, H. (2001). "The echo state approach"
- Maass, W., Natschläger, T., & Markram, H. (2002). "Real-time computing without stable states"
- Verstraeten, D., et al. (2007). "An experimental unification of reservoir computing methods"

## See Also

- `NLT_README.md` - Nonlinear transformation task
- `NARMA_README.md` - NARMA temporal task
- `KERNELRANK_README.md` - Kernel quality evaluation

