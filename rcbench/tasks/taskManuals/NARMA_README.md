# NARMA (Nonlinear Auto-Regressive Moving Average) Task Manual

## Overview

The **NARMA** task is a standard benchmark for evaluating reservoir computing systems' ability to handle nonlinear temporal dynamics. It tests both memory and nonlinear processing capabilities through a challenging time series prediction task.

## Task Description

### Goal
Predict the next value in a NARMA time series based on the current reservoir state.

### Mathematical Formulation

#### NARMA-N (General Form)

```
y[t+1] = α·y[t] + β·y[t]·Σ(y[t-i] for i=0 to N-1) + γ·u[t-N]·u[t] + δ
```

Where:
- `y[t]` = output at time t
- `u[t]` = input at time t (normalized to [0, 0.5])
- `N` = order of the system
- `α, β, γ, δ` = coefficients controlling system dynamics

#### NARMA-2 (Special Case)

```
y[t] = α·y[t-1] + β·y[t-1]·y[t-2] + γ·(u[t-1])³ + δ
```

### Default Coefficients

- **α** = 0.4 (linear feedback)
- **β** = 0.4 (nonlinear feedback)
- **γ** = 0.6 (input influence)
- **δ** = 0.1 (bias/offset)

### Difficulty

- **NARMA-2**: Easier, requires short-term memory
- **NARMA-10**: Standard benchmark, moderate difficulty
- **NARMA-20+**: Challenging, requires long-term memory

## Usage

### Basic Usage

```python
from rcbench import ElecResDataset, NarmaEvaluator
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
evaluator = NarmaEvaluator(
    input_signal=input_signal,
    nodes_output=nodes_output,
    node_names=node_names,
    order=2,  # NARMA-2
    alpha=0.4,
    beta=0.4,
    gamma=0.6,
    delta=0.1
)

# Run evaluation
result = evaluator.run_evaluation(
    metric='NMSE',
    feature_selection_method='kbest',
    num_features='all',
    modeltype="Ridge",
    regression_alpha=1.0,
    train_ratio=0.8
)

logger.output(f"NARMA-{evaluator.order} Performance:")
logger.output(f"  NMSE: {result['accuracy']:.6f}")
logger.output(f"  Selected features: {len(result['selected_features'])}")
```

### With Custom Coefficients

```python
# Create evaluator with custom coefficients
evaluator = NarmaEvaluator(
    input_signal=input_signal,
    nodes_output=nodes_output,
    node_names=node_names,
    order=10,
    alpha=0.3,  # Reduced linear feedback
    beta=0.05,  # Reduced nonlinear feedback  
    gamma=1.5,  # Increased input influence
    delta=0.1
)

# Or modify after creation
evaluator.set_coefficients(
    alpha=0.3,
    beta=0.05,
    gamma=1.5,
    delta=0.1
)

# Regenerate targets with new coefficients
evaluator.targets = evaluator.target_generator()
```

### With Plotting

```python
from rcbench.visualization.plot_config import NarmaPlotConfig

plot_config = NarmaPlotConfig(
    save_dir="./narma_results",
    show_plot=True,
    
    # General plots
    plot_input_signal=True,
    plot_output_responses=True,
    plot_nonlinearity=True,
    plot_frequency_analysis=True,
    
    # Styling
    prediction_sample_count=500,
    frequency_range=(0, 50)
)

evaluator = NarmaEvaluator(
    input_signal=input_signal,
    nodes_output=nodes_output,
    node_names=node_names,
    order=2,
    plot_config=plot_config
)

result = evaluator.run_evaluation(
    metric='NMSE',
    modeltype="Ridge",
    regression_alpha=1.0
)

# Generate comprehensive plots
evaluator.plot_results()
```

## Parameters

### Constructor Parameters

- **`input_signal`** (np.ndarray): Driving input for the NARMA system
- **`nodes_output`** (np.ndarray): Reservoir node outputs, shape `[time_steps, n_nodes]`
- **`node_names`** (List[str], optional): Names of nodes for plotting
- **`order`** (int): Order of NARMA system (default: 2)
  - Common values: 2, 10, 20, 30
- **`alpha`** (float): Linear feedback coefficient (default: 0.4)
- **`beta`** (float): Nonlinear feedback coefficient (default: 0.4)
- **`gamma`** (float): Input influence coefficient (default: 0.6)
- **`delta`** (float): Bias/offset term (default: 0.1)
- **`plot_config`** (NarmaPlotConfig, optional): Configuration for plotting

### Evaluation Parameters

- **`metric`** (str): Evaluation metric
  - `'NMSE'` - Normalized Mean Squared Error (default)
  - `'RNMSE'` - Root Normalized Mean Squared Error
  - `'MSE'` - Mean Squared Error
- **`feature_selection_method`** (str): Feature selection method
  - `'kbest'` - K-best features (recommended)
  - `'pca'` - Principal Component Analysis
  - `'none'` - Use all features
- **`num_features`** (int or 'all'): Number of features to use
- **`modeltype`** (str): Regression model - `'Ridge'` or `'Linear'`
- **`regression_alpha`** (float): Regularization parameter (default: 1.0)
- **`train_ratio`** (float): Training data ratio (default: 0.8)
- **`plot`** (bool): Generate plots during evaluation (default: False)

## Output

### Result Dictionary

```python
{
    'accuracy': float,           # Error metric value
    'metric': str,               # Metric used
    'selected_features': list,   # Indices of selected features
    'model': sklearn.Model,      # Trained regression model
    'y_pred': np.ndarray,        # Predicted NARMA values
    'y_test': np.ndarray,        # True NARMA values
    'train_ratio': float         # Train/test split
}
```

### Target Access

```python
# Access generated NARMA target
narma_target = evaluator.targets['narma']

# Access coefficients
coeffs = evaluator.coefficients
print(f"Alpha: {coeffs['alpha']}")
print(f"Beta: {coeffs['beta']}")
print(f"Gamma: {coeffs['gamma']}")
print(f"Delta: {coeffs['delta']}")
```

## Metrics

### NMSE (Normalized Mean Squared Error)

```
NMSE = MSE(y_true, y_pred) / Var(y_true)
```

- **Lower is better**
- NMSE = 0: Perfect prediction
- NMSE ≥ 1: Worse than predicting the mean

**Interpretation for NARMA:**
- NMSE < 0.1: Excellent
- 0.1 ≤ NMSE < 0.3: Good
- 0.3 ≤ NMSE < 0.5: Moderate
- NMSE ≥ 0.5: Poor

### RNMSE (Root Normalized Mean Squared Error)

```
RNMSE = sqrt(NMSE)
```

- More interpretable scale
- Roughly indicates fractional error

### MSE (Mean Squared Error)

```
MSE = mean((y_true - y_pred)²)
```

- Absolute error in signal units
- Depends on target signal scale

## Visualization

### Generated Plots

1. **Input Signal**: Time series of driving input
2. **Node Responses**: Reservoir node activations
3. **Nonlinearity**: Input-output relationships
4. **Frequency Analysis**: Spectral content of signals
5. **NARMA Prediction**: True vs predicted NARMA series
6. **Prediction Error**: Error over time

### Plot Configuration

```python
plot_config = NarmaPlotConfig(
    figsize=(12, 8),
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

### Example 1: Compare Different Orders

```python
orders = [2, 5, 10, 15]
results = {}

for order in orders:
    evaluator = NarmaEvaluator(
        input_signal=input_signal,
        nodes_output=nodes_output,
        order=order
    )
    
    result = evaluator.run_evaluation(
        metric='NMSE',
        modeltype="Ridge",
        regression_alpha=1.0
    )
    
    results[order] = result['accuracy']
    print(f"NARMA-{order:2d}: NMSE = {result['accuracy']:.6f}")

# Find best order
best_order = min(results, key=results.get)
print(f"\nBest performance: NARMA-{best_order}")
```

### Example 2: Optimize Regularization

```python
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
nmse_values = []

evaluator = NarmaEvaluator(
    input_signal=input_signal,
    nodes_output=nodes_output,
    order=10
)

for alpha in alphas:
    result = evaluator.run_evaluation(
        metric='NMSE',
        modeltype="Ridge",
        regression_alpha=alpha
    )
    nmse_values.append(result['accuracy'])
    print(f"α = {alpha:6.2f}: NMSE = {result['accuracy']:.6f}")

# Find optimal alpha
best_idx = np.argmin(nmse_values)
print(f"\nOptimal α: {alphas[best_idx]}")
```

### Example 3: Feature Selection Comparison

```python
methods = ['kbest', 'pca', 'none']
feature_counts = [5, 10, 15, 'all']

for method in methods:
    print(f"\nMethod: {method}")
    for n_features in feature_counts:
        result = evaluator.run_evaluation(
            metric='NMSE',
            feature_selection_method=method,
            num_features=n_features,
            modeltype="Ridge",
            regression_alpha=1.0
        )
        print(f"  {str(n_features):>3} features: NMSE = {result['accuracy']:.6f}")
```

### Example 4: Using Synthetic Data

```python
import numpy as np

# Generate synthetic data
n_samples = 2000
np.random.seed(42)

# Random input
input_signal = np.random.uniform(-1, 1, n_samples)

# Create synthetic reservoir
n_nodes = 15
nodes_output = np.zeros((n_samples, n_nodes))

for i in range(n_nodes):
    delay = np.random.randint(1, 5)
    nonlinearity = 0.5 + np.random.rand() * 2
    nodes_output[:, i] = np.tanh(nonlinearity * np.roll(input_signal, delay))
    nodes_output[:, i] += np.random.randn(n_samples) * 0.05

# Evaluate
evaluator = NarmaEvaluator(
    input_signal=input_signal,
    nodes_output=nodes_output,
    order=2
)

result = evaluator.run_evaluation(
    metric='NMSE',
    modeltype="Ridge",
    regression_alpha=1.0
)

print(f"NARMA-2 NMSE: {result['accuracy']:.6f}")
```

### Example 5: Custom Coefficients

```python
# Create chaotic NARMA system
evaluator = NarmaEvaluator(
    input_signal=input_signal,
    nodes_output=nodes_output,
    order=10,
    alpha=0.3,  # Less stable
    beta=0.8,   # More nonlinear
    gamma=1.0,  # More input driven
    delta=0.0   # No bias
)

result = evaluator.run_evaluation(
    metric='NMSE',
    modeltype="Ridge",
    regression_alpha=1.0
)

print(f"Chaotic NARMA-10 NMSE: {result['accuracy']:.6f}")
```

## Interpretation Guide

### Performance Benchmarks

#### NARMA-2
- NMSE < 0.05: Excellent
- 0.05 ≤ NMSE < 0.15: Good
- 0.15 ≤ NMSE < 0.30: Moderate
- NMSE ≥ 0.30: Poor

#### NARMA-10
- NMSE < 0.10: Excellent
- 0.10 ≤ NMSE < 0.30: Good
- 0.30 ≤ NMSE < 0.50: Moderate
- NMSE ≥ 0.50: Poor

#### NARMA-20+
- NMSE < 0.20: Excellent
- 0.20 ≤ NMSE < 0.50: Good
- 0.50 ≤ NMSE < 0.80: Moderate
- NMSE ≥ 0.80: Poor

### Common Observations

1. **Performance degrades with order**: Higher orders require more memory

2. **Nonlinearity is crucial**: Linear systems fail at NARMA tasks

3. **Memory requirements scale with order**: 
   - NARMA-2: 2-3 time steps
   - NARMA-10: 10-15 time steps
   - NARMA-30: 30+ time steps

4. **Training data requirements**:
   - Minimum: 500-1000 samples
   - Recommended: 2000-5000 samples
   - More samples needed for higher orders

### What Makes NARMA Difficult?

1. **Nonlinear Feedback**: y[t] depends nonlinearly on past y values
2. **Long-term Dependencies**: Need to remember inputs N steps ago
3. **Multiplication of States**: Requires multiplicative interactions
4. **Accumulation**: Errors can compound over time

## Tips and Best Practices

### 1. Choosing NARMA Order

- **NARMA-2**: Quick test of basic capabilities
- **NARMA-10**: Standard benchmark, widely reported
- **NARMA-20+**: Stress test for memory and nonlinearity

### 2. Input Signal

- Use **random uniform** in range [0, 0.5] or normalized to this range
- Ensure sufficient data length (> 2000 samples recommended)
- Avoid periodic inputs (mask memory requirements)

### 3. Feature Selection

- **K-best** often works well for NARMA
- Start with `num_features='all'`, then optimize
- More features needed for higher orders

### 4. Regularization

- Start with α = 1.0
- Increase if overfitting (α = 10-100)
- Decrease for larger datasets (α = 0.1-1.0)

### 5. Train/Test Split

- Use 80/20 split typically
- Ensure test set is long enough (> 200 samples)
- Consider multiple random splits for robustness

### 6. Troubleshooting Poor Performance

If NMSE > 0.5:
- Check reservoir has nonlinearity
- Verify sufficient nodes (> 10 for NARMA-2, > 50 for NARMA-10)
- Ensure adequate memory capacity
- Try different feature selection methods
- Check input signal is properly scaled

## Relationship to Other Tasks

### NARMA vs Memory Capacity

- **Memory Capacity**: Pure linear memory
- **NARMA**: Memory + nonlinearity combined
- Good MC → necessary but not sufficient for NARMA

### NARMA vs NLT

- **NLT**: Instantaneous nonlinear transformations
- **NARMA**: Temporal + nonlinear
- NARMA is more challenging

### NARMA vs Nonlinear Memory

- **NARMA**: Specific nonlinear time series
- **Nonlinear Memory**: Parametric sweep of delay + nonlinearity
- Nonlinear Memory helps understand NARMA performance

## Common Issues

### Issue 1: NaN or Inf in Predictions

**Causes:**
- Unstable NARMA generation (β too large)
- Numerical overflow in reservoir

**Solutions:**
- Reduce β coefficient
- Check reservoir outputs for extreme values
- Normalize reservoir states

### Issue 2: High Variance Across Runs

**Causes:**
- Insufficient training data
- Unstable NARMA dynamics
- Poor initialization

**Solutions:**
- Use more training data
- Set random_state for reproducibility
- Average over multiple runs

### Issue 3: Good Training, Poor Test Performance

**Cause:** Overfitting

**Solutions:**
- Increase regularization (α)
- Reduce number of features
- Use more training data
- Check for data leakage

## Advanced Usage

### Generate NARMA Target Manually

```python
from rcbench.tasks.narma import generate_narma_target

# Generate custom NARMA sequence
target = generate_narma_target(
    u=input_signal,
    order=10,
    coefficients={
        'alpha': 0.3,
        'beta': 0.05,
        'gamma': 1.5,
        'delta': 0.1
    }
)
```

### Normalize Input

```python
from rcbench.tasks.narma import normalize_to_range

# Normalize to [0, 0.5] range (NARMA standard)
normalized_input = normalize_to_range(input_signal, 0, 0.5)
```

## References

- Atiya, A. F., & Parlos, A. G. (2000). "New results on recurrent network training"
- Jaeger, H., & Haas, H. (2004). "Harnessing nonlinearity: Predicting chaotic systems"
- Lukosevicius, M., & Jaeger, H. (2009). "Reservoir computing approaches to recurrent neural network training"

## See Also

- `MEMORYCAPACITY_README.md` - Memory capacity evaluation
- `NONLINEARMEMORY_README.md` - Nonlinear memory benchmark
- `NLT_README.md` - Nonlinear transformation task

