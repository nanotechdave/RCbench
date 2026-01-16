# Information Processing Capacity (IPC) Evaluator

## Overview

The **Information Processing Capacity (IPC) Evaluator** implements the computational capacity framework introduced by Dambre et al. (2012). It provides a comprehensive measure of a reservoir's ability to process information, decomposed into linear memory and nonlinear components.

## Theoretical Background

### Reference

> Dambre, J., Verstraeten, D., Schrauwen, B. & Massar, S. "Information Processing Capacity of Dynamical Systems". *Scientific Reports* 2, 514 (2012). [DOI: 10.1038/srep00514](https://www.nature.com/articles/srep00514)

### Key Concepts

1. **Total Capacity Bound**: The total computational capacity of a dynamical system is bounded by N (the number of linearly independent state variables). If the system has fading memory, this bound is tight.

2. **Orthonormal Basis Functions**: For uniform inputs in [-1, 1], the capacity is measured using Legendre polynomials as basis functions. These polynomials are orthonormal under the uniform distribution.

3. **Capacity Decomposition**: The total capacity can be decomposed into:
   - **Linear Memory Capacity**: Ability to reconstruct delayed inputs (degree-1 polynomials)
   - **Nonlinear Capacity**: Ability to compute nonlinear functions of past inputs (degree > 1)

### Mathematical Formulation

For each basis function z_i(t), the capacity C_i is the squared correlation coefficient:

```
C_i = cov(z_i, ŷ_i)² / (var(z_i) * var(ŷ_i))
```

Where ŷ_i is the linear readout prediction.

#### Basis Functions

Single-variable terms:
```
z_{d,τ}(t) = P_d(u(t-τ))
```

Cross-delay terms (products):
```
z_{d₁,τ₁,d₂,τ₂}(t) = P_{d₁}(u(t-τ₁)) * P_{d₂}(u(t-τ₂))
```

Where:
- P_d is the normalized Legendre polynomial of degree d
- u(t) is the input signal
- τ is the delay

#### Total Capacity

```
C_total = Σᵢ Cᵢ ≤ N
```

## Usage Example

```python
import numpy as np
from rcbench import IPCEvaluator, IPCPlotConfig

# Generate random input (uniform in [-1, 1])
np.random.seed(42)
input_signal = np.random.uniform(-1, 1, size=5000)

# Simulate reservoir (your reservoir states)
# reservoir_states shape: [time_steps, n_nodes]
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
    include_cross_terms=True,  # Include product terms
    random_state=42,
    plot_config=plot_config
)

# Calculate total information processing capacity
results = evaluator.calculate_total_capacity(
    feature_selection_method='pca',
    num_features='all',
    modeltype='Ridge',
    regression_alpha=0.1,
    train_ratio=0.8
)

# Print results
print(f"Total Capacity: {results['total_capacity']:.4f}")
print(f"Linear Memory Capacity: {results['linear_memory_capacity']:.4f}")
print(f"Nonlinear Capacity: {results['nonlinear_capacity']:.4f}")
print(f"Theoretical Maximum: {results['theoretical_max']}")
print(f"Efficiency: {results['total_capacity']/results['theoretical_max']*100:.1f}%")

# Analyze trade-off
tradeoff = evaluator.get_memory_nonlinearity_tradeoff()
print("\nCapacity by delay:")
for i, delay in enumerate(tradeoff['delays']):
    print(f"  τ={delay}: Linear={tradeoff['linear_capacity'][i]:.3f}, "
          f"Nonlinear={tradeoff['nonlinear_capacity'][i]:.3f}")

# Generate plots
evaluator.plot_results()

# Get summary
summary = evaluator.summary()
```

## API Reference

### IPCEvaluator

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_signal` | array-like | Required | Input signal (should be uniform in [-1, 1]) |
| `nodes_output` | np.ndarray | Required | Reservoir states, shape [time_steps, n_nodes] |
| `max_delay` | int | 10 | Maximum delay τ to consider |
| `max_degree` | int | 3 | Maximum polynomial degree per variable |
| `max_total_degree` | int | None | Maximum total degree for cross-terms |
| `include_cross_terms` | bool | True | Whether to include product terms |
| `random_state` | int | 42 | Random seed for reproducibility |
| `node_names` | List[str] | None | Names of reservoir nodes |
| `plot_config` | IPCPlotConfig | None | Plotting configuration |

#### Key Methods

##### `calculate_total_capacity()`

Evaluates all basis functions and computes total IPC.

**Parameters:**
- `feature_selection_method` (str): 'pca', 'kbest', or 'none'
- `num_features` (int or 'all'): Number of features to use
- `modeltype` (str): 'Ridge' or 'Linear'
- `regression_alpha` (float): Ridge regularization parameter
- `train_ratio` (float): Training data ratio

**Returns:** Dictionary containing:
- `total_capacity`: Sum of all capacities
- `linear_memory_capacity`: Sum of degree-1 capacities
- `nonlinear_capacity`: Sum of degree > 1 capacities
- `capacity_by_degree`: Dict mapping degree to total capacity
- `capacity_by_delay`: Dict mapping delay to capacity breakdown
- `theoretical_max`: Number of reservoir nodes (N)

##### `get_memory_nonlinearity_tradeoff()`

Analyzes capacity as a function of delay, separated into linear and nonlinear components.

**Returns:** Dictionary with:
- `delays`: Array of delay values
- `linear_capacity`: Linear capacity at each delay
- `nonlinear_capacity`: Nonlinear capacity at each delay
- `total_capacity`: Total capacity at each delay

##### `plot_results()`

Generates visualization plots:
1. Capacity by polynomial degree (bar chart)
2. Memory-nonlinearity trade-off (stacked area)
3. Summary comparison

##### `summary()`

Returns comprehensive summary statistics.

### IPCPlotConfig

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `figsize` | Tuple[int, int] | (10, 6) | Figure size |
| `dpi` | int | 100 | Figure resolution |
| `save_dir` | str | None | Directory to save plots |
| `show_plot` | bool | True | Whether to display plots |
| `plot_capacity_by_degree` | bool | True | Plot capacity by degree |
| `plot_tradeoff` | bool | True | Plot memory-nonlinearity trade-off |
| `plot_summary` | bool | True | Plot capacity summary |

## Interpretation Guide

### Understanding the Results

1. **Total Capacity Close to N**: The reservoir efficiently uses all its degrees of freedom for computation. This is ideal.

2. **High Linear Memory, Low Nonlinear**: The reservoir is essentially linear, good for memory tasks but limited nonlinear processing.

3. **High Nonlinear, Low Linear Memory**: The reservoir performs strong nonlinear transformations but loses temporal information quickly.

4. **Balanced Profile**: Optimal for general-purpose reservoir computing, indicating operation near the "edge of chaos".

### The Memory-Nonlinearity Trade-off

A fundamental result from Dambre et al. is that dynamical systems face a trade-off:
- Systems optimized for memory (linear transformations with long delays) sacrifice nonlinear processing
- Systems optimized for nonlinearity sacrifice memory depth

The optimal "edge of chaos" configuration balances both, achieving high capacity across different delay-degree combinations.

## Comparison with Memory Capacity

| Aspect | Memory Capacity (MC) | Information Processing Capacity (IPC) |
|--------|---------------------|---------------------------------------|
| Measures | Linear memory only | Total computational capacity |
| Basis | Delayed inputs u(t-k) | Legendre polynomials P_d(u(t-τ)) |
| Theoretical max | N | N |
| Includes nonlinearity | No | Yes |
| Reference | Jaeger (2002) | Dambre et al. (2012) |

## Notes

1. **Input Distribution**: The Legendre polynomial basis is orthonormal only for uniform inputs in [-1, 1]. For Gaussian inputs, Hermite polynomials would be appropriate (not yet implemented).

2. **Computational Cost**: IPC evaluation involves many more basis functions than standard MC. For large `max_delay` and `max_degree`, computation time increases significantly.

3. **Cross-Terms**: Including cross-terms (`include_cross_terms=True`) provides a more complete picture but increases the number of basis functions combinatorially.

## See Also

- `MemoryCapacityEvaluator`: For standard linear memory capacity
- `NonlinearMemoryEvaluator`: For y(t) = sin(ν * s(t-τ)) benchmark
- `NltEvaluator`: For nonlinear transformation tasks

## References

1. Dambre, J., Verstraeten, D., Schrauwen, B. & Massar, S. "Information Processing Capacity of Dynamical Systems". *Scientific Reports* 2, 514 (2012).

2. Jaeger, H. "Short Term Memory in Echo State Networks". Fraunhofer Institute AIS, Tech. rep. 152 (2002).

3. Boyd, S. & Chua, L. "Fading memory and the problem of approximating nonlinear operators with Volterra series". IEEE Trans. Circuits Syst. 32, 1150-1161 (1985).

