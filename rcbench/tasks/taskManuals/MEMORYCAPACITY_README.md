# Memory Capacity Task Manual

## Overview

The **Memory Capacity** task evaluates a reservoir's ability to recall past inputs over different time delays. It measures both short-term and long-term memory capabilities by training linear readouts to reconstruct time-delayed versions of the input signal.

## Task Description

### Goal
Measure how well the reservoir can remember and recall input values from τ time steps in the past.

### Mathematical Formulation

For each delay τ ∈ {1, 2, ..., max_delay}, train a linear readout to predict:

```
y(t) = s(t - τ)
```

where `s(t)` is the input signal at time t.

### Memory Capacity Metric

The memory capacity for delay τ is computed as the squared correlation coefficient:

```
MC(τ) = (cov(y_true, y_pred)²) / (var(y_true) × var(y_pred))
```

**Total Memory Capacity** is the sum:

```
MC_total = Σ MC(τ)  for τ = 1 to max_delay
```

### Theoretical Maximum

For a linear system with N independent nodes, the maximum theoretical memory capacity is N.

## Usage

### Basic Usage

```python
from rcbench import ElecResDataset, MemoryCapacityEvaluator
from rcbench.visualization.plot_config import MCPlotConfig
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
evaluator = MemoryCapacityEvaluator(
    input_signal=input_signal,
    nodes_output=nodes_output,
    max_delay=30,  # Test up to 30 time steps
    random_state=42,
    node_names=node_names
)

# Calculate total memory capacity
results = evaluator.calculate_total_memory_capacity(
    feature_selection_method='pca',
    num_features='all',
    modeltype='Ridge',
    regression_alpha=0.1,
    train_ratio=0.8
)

# Display results
logger.output(f"Total Memory Capacity: {results['total_memory_capacity']:.4f}")

# Show per-delay results
for delay, mc_value in results['delay_results'].items():
    logger.output(f"  Delay {delay:2d}: MC = {mc_value:.4f}")
```

### With Plotting

```python
# Create plot configuration
plot_config = MCPlotConfig(
    save_dir="./mc_results",
    show_plot=True,
    
    # MC-specific plots
    plot_mc_curve=True,           # MC vs delay curve
    plot_predictions=True,        # Predictions for each delay
    plot_total_mc=True,           # Cumulative memory capacity
    max_delays_to_plot=5,         # Show first 5 delays
    
    # General reservoir plots
    plot_input_signal=True,
    plot_output_responses=True,
    plot_nonlinearity=True,
    plot_frequency_analysis=True,
    
    # Styling
    nonlinearity_plot_style='scatter',
    frequency_range=(0, 50)
)

evaluator = MemoryCapacityEvaluator(
    input_signal=input_signal,
    nodes_output=nodes_output,
    max_delay=30,
    random_state=42,
    node_names=node_names,
    plot_config=plot_config
)

results = evaluator.calculate_total_memory_capacity(
    feature_selection_method='pca',
    num_features='all',
    modeltype='Ridge',
    regression_alpha=0.1,
    train_ratio=0.8
)

# Generate all plots
evaluator.plot_results()
```

### Evaluate Single Delay

```python
# Evaluate specific delay
result = evaluator.run_evaluation(
    delay=5,  # 5 time steps
    modeltype="Ridge",
    regression_alpha=1.0,
    train_ratio=0.8
)

print(f"Memory capacity at delay 5: {result['memory_capacity']:.4f}")
```

## Parameters

### Constructor Parameters

- **`input_signal`** (np.ndarray): Input stimulation signal array
- **`nodes_output`** (np.ndarray): Reservoir node outputs, shape `[time_steps, n_nodes]`
- **`max_delay`** (int): Maximum delay steps to evaluate (default: 30)
- **`random_state`** (int): Random seed for reproducibility (default: 42)
- **`node_names`** (List[str], optional): Names of nodes for plotting
- **`plot_config`** (MCPlotConfig, optional): Configuration for plotting

### Evaluation Parameters

#### calculate_total_memory_capacity()

- **`feature_selection_method`** (str): Feature selection method
  - `'pca'` - Principal Component Analysis (recommended)
  - `'kbest'` - K-best features
  - `'none'` - Use all features
- **`num_features`** (int or 'all'): Number of features to use
- **`modeltype`** (str): Regression model - `'Ridge'` or `'Linear'`
- **`regression_alpha`** (float): Regularization parameter for Ridge
- **`train_ratio`** (float): Training data ratio (0-1)

#### run_evaluation()

- **`delay`** (int): Specific delay to evaluate
- **`modeltype`** (str): Regression model type
- **`regression_alpha`** (float): Regularization parameter
- **`train_ratio`** (float): Training data ratio

## Output

### Total Memory Capacity Results

```python
{
    'total_memory_capacity': float,     # Sum of all MC values
    'delay_results': {                  # MC for each delay
        1: float,
        2: float,
        ...
        max_delay: float
    },
    'all_results': {                    # Full results per delay
        1: {...},
        2: {...},
        ...
    }
}
```

### Single Delay Results

```python
{
    'delay': int,               # Delay value
    'memory_capacity': float,   # MC value for this delay
    'model': sklearn.Model,     # Trained model
    'y_pred': np.ndarray,       # Predictions
    'y_test': np.ndarray,       # True values
    'time_test': np.ndarray     # Time indices for test data
}
```

## Memory Capacity Metric

### Calculation

The memory capacity for each delay is the squared correlation coefficient:

```python
MC(τ) = r²(y_true, y_pred)
```

where `r` is the Pearson correlation coefficient.

### Properties

- **Range**: [0, 1]
- **MC = 1**: Perfect memory (perfect prediction)
- **MC = 0**: No memory (no correlation)
- **Sum over delays**: Total memory capacity

### Interpretation

- **High MC at small delays**: Good short-term memory
- **High MC at large delays**: Good long-term memory
- **Rapid decay**: Limited memory depth
- **Slow decay**: Extended memory capabilities

## Visualization

### Generated Plots

1. **MC vs Delay Curve**: Shows how memory capacity degrades with delay
2. **Cumulative MC**: Running sum of memory capacity
3. **Prediction Results**: True vs predicted signals for selected delays
4. **Input Signal**: Original input time series
5. **Node Responses**: Reservoir node activations
6. **Nonlinearity**: Input-output relationships
7. **Frequency Analysis**: Spectral content

### Plot Configuration

```python
plot_config = MCPlotConfig(
    figsize=(10, 6),
    dpi=150,
    save_dir="./results",
    show_plot=True,
    
    # Control which plots to generate
    plot_mc_curve=True,
    plot_predictions=True,
    plot_total_mc=True,
    max_delays_to_plot=5,
    
    # General properties
    plot_input_signal=True,
    plot_output_responses=True,
    plot_nonlinearity=True,
    plot_frequency_analysis=True,
    
    # Styling
    prediction_sample_count=200,
    frequency_range=(0, 100)
)
```

## Examples

### Example 1: Basic Memory Capacity

```python
evaluator = MemoryCapacityEvaluator(
    input_signal=input_signal,
    nodes_output=nodes_output,
    max_delay=20
)

results = evaluator.calculate_total_memory_capacity(
    feature_selection_method='pca',
    num_features='all',
    modeltype='Ridge',
    regression_alpha=0.1
)

print(f"Total MC: {results['total_memory_capacity']:.2f}")
print(f"Average MC per delay: {results['total_memory_capacity'] / 20:.4f}")
```

### Example 2: Compare Different Feature Counts

```python
feature_counts = [5, 10, 15, 'all']
mc_values = []

for n_features in feature_counts:
    evaluator = MemoryCapacityEvaluator(
        input_signal=input_signal,
        nodes_output=nodes_output,
        max_delay=25
    )
    
    results = evaluator.calculate_total_memory_capacity(
        feature_selection_method='pca',
        num_features=n_features,
        modeltype='Ridge',
        regression_alpha=0.1
    )
    
    mc = results['total_memory_capacity']
    mc_values.append(mc)
    print(f"Features: {n_features:>3}, MC: {mc:.4f}")
```

### Example 3: Analyze Memory Decay

```python
results = evaluator.calculate_total_memory_capacity(
    feature_selection_method='pca',
    num_features='all',
    modeltype='Ridge',
    regression_alpha=0.1
)

delay_results = results['delay_results']

# Find effective memory horizon (MC drops below threshold)
threshold = 0.1
effective_horizon = max_delay

for delay in sorted(delay_results.keys()):
    if delay_results[delay] < threshold:
        effective_horizon = delay - 1
        break

print(f"Effective memory horizon: {effective_horizon} steps")
print(f"MC at horizon: {delay_results.get(effective_horizon, 0):.4f}")
```

### Example 4: Using Synthetic Data

```python
import numpy as np

# Generate random input
n_samples = 3000
np.random.seed(42)
input_signal = np.random.uniform(-1, 1, n_samples)

# Create reservoir with built-in memory
n_nodes = 20
nodes_output = np.zeros((n_samples, n_nodes))

for i in range(n_nodes):
    delay = np.random.randint(0, 10)
    decay = 0.7 + 0.25 * np.random.random()
    gain = 0.5 + 1.0 * np.random.random()
    
    node_signal = np.zeros(n_samples)
    for t in range(delay, n_samples):
        current_input = gain * input_signal[t - delay]
        if t > 0:
            node_signal[t] = current_input + decay * node_signal[t-1]
        else:
            node_signal[t] = current_input
        node_signal[t] = np.tanh(node_signal[t])
    
    node_signal += 0.05 * np.random.normal(0, 1, n_samples)
    nodes_output[:, i] = node_signal

# Evaluate
evaluator = MemoryCapacityEvaluator(
    input_signal=input_signal,
    nodes_output=nodes_output,
    max_delay=25
)

results = evaluator.calculate_total_memory_capacity(
    feature_selection_method='pca',
    num_features=15,
    modeltype='Ridge',
    regression_alpha=0.1
)

print(f"Total MC: {results['total_memory_capacity']:.4f}")
```

## Interpretation Guide

### Performance Benchmarks

For N nodes:
- **MC ≈ N**: Excellent (near-theoretical maximum)
- **MC ≈ 0.7N - 0.9N**: Very good
- **MC ≈ 0.4N - 0.7N**: Good
- **MC < 0.4N**: Limited memory capacity

### Typical Memory Curves

1. **Exponential Decay**: 
   - Common in physical reservoirs
   - MC(τ) ≈ MC₀ · exp(-τ/τ₀)
   - τ₀ is the memory time constant

2. **Power Law Decay**:
   - Some nonlinear reservoirs
   - MC(τ) ≈ MC₀ · τ^(-α)

3. **Plateau Then Drop**:
   - Multi-timescale systems
   - Short-term memory plateau, then rapid decay

### Common Observations

- **Initial MC close to 1.0**: Excellent short-term memory
- **Slow decay**: Long memory horizon
- **Oscillations in MC curve**: Periodic reservoir dynamics
- **Near-zero MC for large τ**: Memory horizon reached

## Tips and Best Practices

1. **Choose max_delay**:
   - Start with max_delay = 2 × (expected memory horizon)
   - Typical range: 20-50 for most reservoirs

2. **Feature Selection**:
   - PCA often works best for memory capacity
   - Use enough components to capture 95% variance

3. **Regularization**:
   - Higher α for noisy systems (α = 0.1 - 1.0)
   - Lower α for clean data (α = 0.01 - 0.1)

4. **Input Signal**:
   - Use white noise or random uniform signal
   - Ensure sufficient excitation across frequencies

5. **Sampling Rate**:
   - Higher sampling improves delay resolution
   - Balance with computational cost

6. **Interpretation**:
   - Compare MC_total to number of nodes
   - Look for memory decay pattern
   - Check if memory meets task requirements

## Common Issues

### Issue 1: Very Low MC

**Possible Causes:**
- Insufficient reservoir nonlinearity
- Too much noise
- Poor feature selection
- Input signal not well-distributed

**Solutions:**
- Check node responses are varied
- Try different feature selection methods
- Reduce noise level
- Use better input signal

### Issue 2: MC > 1 for Some Delays

**Cause**: Numerical instability in correlation calculation

**Solution**: This shouldn't occur with proper implementation, check data quality

### Issue 3: Oscillating MC Curve

**Cause**: Periodic dynamics in reservoir

**Interpretation**: Not necessarily bad, may indicate rich dynamics

### Issue 4: Flat MC Curve

**Causes:**
- All delays equally (poorly) predicted
- Insufficient memory capabilities

**Solutions:**
- Check reservoir parameters
- Ensure proper connectivity
- Verify input signal quality

## Relationship to Other Tasks

### Memory Capacity vs Nonlinear Memory

- **Memory Capacity**: Tests **linear** memory (delay recall)
- **Nonlinear Memory**: Tests **nonlinear** transformations of delayed inputs
- Nonlinear Memory is a superset including nonlinearity

### Memory Capacity vs NARMA

- **Memory Capacity**: Pure memory test
- **NARMA**: Memory + nonlinearity combined
- Good MC often correlates with good NARMA performance

## References

- Jaeger, H. (2001). "The echo state approach to analysing and training recurrent neural networks"
- Dambre et al. (2012). "Information processing capacity of dynamical systems"
- Verstraeten et al. (2007). "Memory versus non-linearity in reservoirs"

## See Also

- `NONLINEARMEMORY_README.md` - Nonlinear memory benchmark
- `NARMA_README.md` - NARMA task documentation
- `NLT_README.md` - Nonlinear transformation task

