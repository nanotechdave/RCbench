# Generalization Rank (GR) Task Manual

## Overview

The **Generalization Rank (GR)** evaluator measures a reservoir's ability to generalize by analyzing how similar the reservoir states are when driven by multiple noisy versions of the same input. A low GR indicates good generalization (robust to noise), while a high GR indicates sensitivity to input variations.

## Theoretical Background

### Reference

> Vidamour, I. T., et al. (2022). "Reconfigurable reservoir computing in a magnetic metamaterial". *Nanotechnology* 33, 485203.

### Key Concepts

1. **Generalization**: The ability of a reservoir to produce similar outputs for inputs that are noisy versions of each other. Good generalization means the reservoir is robust to input perturbations.

2. **State Matrix Construction**: Supply m distinct (noisy) input streams to the reservoir, each producing an n-dimensional state vector x. Arrange these as columns of matrix M (shape n × m).

3. **Effective Rank**: The number of significant singular values indicates how many "independent" directions exist in the state responses to noisy inputs.

### Mathematical Formulation

#### State Matrix Construction

Given m input streams (e.g., noisy versions of the same signal), collect reservoir states:

```
M = [x_u1 | x_u2 | ... | x_um]    (shape: n × m)
```

where x_ui is the reservoir state vector for input stream i.

#### Singular Value Decomposition

Compute SVD of M:

```
M = U Σ V^T
```

#### Effective Rank

The generalization rank is:

```
GR = count(σ_i > threshold × σ_max)
```

where σ_i are the singular values in Σ.

### Interpretation

- **Low GR (close to 1)**: Strong generalization — the reservoir maps all noisy inputs to nearly the same state direction.
- **High GR (close to min(n, m))**: Weak generalization — each noisy input produces a distinct state.

## Usage

### Basic Usage

```python
import numpy as np
from rcbench.tasks.generalizationrank import GeneralizationRankEvaluator
from rcbench.logger import get_logger

logger = get_logger(__name__)

# Suppose we have reservoir states from multiple noisy input trials
# states shape: (m, n) where m = number of trials, n = number of nodes
# Each row is the state vector from one trial

# Example: 10 noisy input trials, 50-node reservoir
states = np.random.randn(10, 50)  # Replace with actual data

# Create evaluator
evaluator = GeneralizationRankEvaluator(
    states=states,
    threshold=1e-3
)

# Run evaluation
results = evaluator.run_evaluation()

# Display results
logger.output(f"Generalization Rank: {results['generalization_rank']}")
logger.output(f"Matrix shape: {results['M_shape']}")
```

### With Experimental Data

```python
import numpy as np
from rcbench.tasks.generalizationrank import GeneralizationRankEvaluator

# Scenario: Apply m noisy versions of an input signal to the reservoir
# and collect the final state vector from each trial

def run_noisy_trials(reservoir, base_input, n_trials, noise_level=0.1):
    """Run reservoir with noisy versions of the same input."""
    states = []
    for _ in range(n_trials):
        noisy_input = base_input + noise_level * np.random.randn(len(base_input))
        final_state = reservoir.run(noisy_input)[-1]  # Last state
        states.append(final_state)
    return np.array(states)

# Collect states
n_trials = 20
states = run_noisy_trials(reservoir, base_input, n_trials)

# Evaluate generalization
evaluator = GeneralizationRankEvaluator(states, threshold=1e-3)
results = evaluator.run_evaluation()

print(f"Generalization Rank: {results['generalization_rank']}")
print(f"Maximum possible: {min(states.shape)}")
print(f"Generalization ratio: {results['generalization_rank'] / min(states.shape):.2%}")
```

## Parameters

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `states` | np.ndarray | Required | 2D array of state vectors. Shape (m, n) where m = number of trials/inputs, n = number of nodes |
| `threshold` | float | 1e-3 | Relative threshold as fraction of max singular value |

### Key Methods

#### `run_evaluation()`

Runs the generalization rank evaluation.

**Returns:** Dictionary containing:
- `generalization_rank`: The computed effective rank
- `singular_values`: All singular values of M
- `M_shape`: Shape of the state matrix
- `threshold`: Threshold used for computation

#### `compute_generalization_rank()`

Computes the effective rank via SVD.

**Returns:** Tuple of (effective_rank, singular_values)

## Output

### Results Dictionary

```python
{
    'generalization_rank': int,       # Effective rank
    'singular_values': np.ndarray,    # Singular values (sorted descending)
    'M_shape': tuple,                 # Shape of state matrix (m, n)
    'threshold': float                # Threshold used
}
```

## Interpretation Guide

### Understanding the Results

1. **GR ≈ 1**: Excellent generalization. All noisy inputs map to essentially the same state direction. The reservoir is very robust but may lack discrimination ability.

2. **GR ≈ min(m, n)**: Poor generalization. Each input variant produces a distinct state. The reservoir is sensitive to noise.

3. **Intermediate GR**: Balance between robustness and sensitivity.

### Performance Benchmarks

| GR / min(m,n) | Interpretation |
|---------------|----------------|
| < 0.2 | Very strong generalization (possibly over-generalized) |
| 0.2 - 0.4 | Good generalization |
| 0.4 - 0.6 | Moderate generalization |
| 0.6 - 0.8 | Weak generalization |
| > 0.8 | Poor generalization (noise-sensitive) |

### Trade-offs

- **High generalization (low GR)**: Robust to noise, but may fail to discriminate between genuinely different inputs.
- **Low generalization (high GR)**: Can distinguish fine differences, but sensitive to noise.

## Examples

### Example 1: Synthetic Noise Analysis

```python
import numpy as np
from rcbench.tasks.generalizationrank import GeneralizationRankEvaluator

np.random.seed(42)

# Simulate reservoir with different noise levels
n_nodes = 50
n_trials = 15

# Base state (what the reservoir would produce with clean input)
base_state = np.random.randn(n_nodes)

# Different noise levels
noise_levels = [0.01, 0.1, 0.5, 1.0]

for noise in noise_levels:
    # Create noisy state matrix
    states = np.array([base_state + noise * np.random.randn(n_nodes) 
                      for _ in range(n_trials)])
    
    evaluator = GeneralizationRankEvaluator(states, threshold=1e-3)
    results = evaluator.run_evaluation()
    
    print(f"Noise level: {noise:.2f}, GR: {results['generalization_rank']}, "
          f"Ratio: {results['generalization_rank']/min(n_trials, n_nodes):.2%}")
```

### Example 2: Analyze Singular Value Spectrum

```python
import matplotlib.pyplot as plt
import numpy as np
from rcbench.tasks.generalizationrank import GeneralizationRankEvaluator

# Create sample data
np.random.seed(42)
states = np.random.randn(20, 50)

evaluator = GeneralizationRankEvaluator(states, threshold=1e-3)
results = evaluator.run_evaluation()

# Plot singular value spectrum
singular_values = results['singular_values']
threshold_value = results['threshold'] * singular_values[0]

plt.figure(figsize=(10, 5))

# Linear scale
plt.subplot(1, 2, 1)
plt.bar(range(len(singular_values)), singular_values, color='steelblue')
plt.axhline(y=threshold_value, color='red', linestyle='--', label='Threshold')
plt.xlabel('Index')
plt.ylabel('Singular Value')
plt.title('Singular Value Spectrum')
plt.legend()

# Cumulative explained variance
plt.subplot(1, 2, 2)
cumulative_var = np.cumsum(singular_values**2) / np.sum(singular_values**2)
plt.plot(range(1, len(cumulative_var)+1), cumulative_var, 'b-o')
plt.axhline(y=0.99, color='red', linestyle='--', label='99% variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title(f'GR = {results["generalization_rank"]}')
plt.legend()

plt.tight_layout()
plt.show()
```

### Example 3: Compare Reservoirs

```python
import numpy as np
from rcbench.tasks.generalizationrank import GeneralizationRankEvaluator

def evaluate_reservoir_generalization(reservoir_name, states):
    """Evaluate and report generalization rank."""
    evaluator = GeneralizationRankEvaluator(states, threshold=1e-3)
    results = evaluator.run_evaluation()
    
    gr = results['generalization_rank']
    max_rank = min(states.shape)
    
    print(f"{reservoir_name}:")
    print(f"  Generalization Rank: {gr}")
    print(f"  Maximum possible: {max_rank}")
    print(f"  GR ratio: {gr/max_rank:.2%}")
    print(f"  Top 3 singular values: {results['singular_values'][:3].round(4)}")
    print()
    
    return results

# Compare different reservoir configurations
# (Replace with actual reservoir outputs)
np.random.seed(42)

# Reservoir A: Good generalization (responses cluster together)
base_A = np.random.randn(30)
states_A = np.array([base_A + 0.1 * np.random.randn(30) for _ in range(15)])

# Reservoir B: Poor generalization (diverse responses)
states_B = np.random.randn(15, 30)

results_A = evaluate_reservoir_generalization("Reservoir A (Low noise)", states_A)
results_B = evaluate_reservoir_generalization("Reservoir B (High diversity)", states_B)
```

### Example 4: With RCbench Data

```python
from rcbench import ElecResDataset
from rcbench.tasks.generalizationrank import GeneralizationRankEvaluator
import numpy as np

# If you have multiple experimental runs
measurement_files = [
    "experiment_run1.txt",
    "experiment_run2.txt",
    # ... more runs
]

# Collect final states from each run
states = []
for file in measurement_files:
    dataset = ElecResDataset(file)
    nodes_output = dataset.get_node_voltages()
    # Use final state or time-averaged state
    final_state = nodes_output[-100:].mean(axis=0)  # Average of last 100 samples
    states.append(final_state)

states = np.array(states)

# Evaluate
evaluator = GeneralizationRankEvaluator(states, threshold=1e-3)
results = evaluator.run_evaluation()

print(f"Generalization Rank: {results['generalization_rank']}")
```

## Experimental Protocol

### Recommended Procedure

1. **Choose base input signal**: Select a representative input signal for your task.

2. **Generate noisy variants**: Create m noisy versions by adding noise:
   ```python
   noisy_input = base_input + noise_level * np.random.randn(len(base_input))
   ```

3. **Run reservoir**: For each noisy input, drive the reservoir and collect the state vector.

4. **Choose state representation**:
   - **Final state**: The state at the last timestep
   - **Time-averaged state**: Mean over a time window
   - **Full trajectory**: Concatenate multiple time points (higher dimensional)

5. **Evaluate GR**: Use the collected state matrix.

### Noise Level Selection

- Start with noise_level = 0.1 × std(base_input)
- Test multiple noise levels to characterize robustness
- Plot GR vs noise_level to understand the reservoir's noise tolerance

## Tips and Best Practices

1. **Number of Trials (m)**: Use at least 10-20 noisy input variants for reliable statistics.

2. **Threshold Selection**:
   - Default 1e-3 works for most cases
   - Lower threshold (1e-4) for higher precision
   - Higher threshold (1e-2) for noisy experimental data

3. **State Representation**:
   - Final state is simplest but may be noisy
   - Time-averaged state is more robust
   - Consider using states from steady-state region

4. **Normalization**: Consider normalizing state vectors before analysis for fair comparison.

## Common Issues

### Issue 1: GR Always Equals 1

**Possible Causes:**
- States are nearly identical (too little noise)
- Threshold too high
- Reservoir over-generalizes

**Solutions:**
- Increase noise level
- Lower threshold
- Check if reservoir produces diverse outputs

### Issue 2: GR Equals min(m, n)

**Possible Causes:**
- States are completely independent (too much noise)
- Threshold too low
- Reservoir doesn't generalize

**Solutions:**
- Decrease noise level
- Increase threshold
- Check reservoir dynamics

### Issue 3: Inconsistent Results

**Possible Causes:**
- Non-stationary reservoir dynamics
- Insufficient warmup period
- Measurement noise

**Solutions:**
- Discard initial transient
- Use longer input sequences
- Average over multiple measurements

## Relationship to Other Metrics

### Generalization Rank vs Kernel Rank

| Aspect | Generalization Rank (GR) | Kernel Rank (KR) |
|--------|--------------------------|------------------|
| Input | Multiple noisy input streams | Single time series |
| Matrix | States from different trials (n × m) | States over time (T × N) |
| Measures | Noise robustness | State space dimensionality |
| Goal | Quantify generalization ability | Quantify computational capacity |

### GR vs Memory Capacity

- **GR**: Measures robustness to input variations
- **MC**: Measures ability to recall past inputs
- A reservoir can have high MC but low GR (good memory, sensitive to noise)

## References

1. Vidamour, I. T., et al. (2022). "Reconfigurable reservoir computing in a magnetic metamaterial". *Nanotechnology* 33, 485203.

2. Tanaka, G., et al. (2019). "Recent advances in physical reservoir computing: A review". *Neural Networks* 115, 100-123.

3. Appeltant, L., et al. (2011). "Information processing using a single dynamical node as complex system". *Nature Communications* 2, 468.

## See Also

- `KERNELRANK_README.md` - Kernel rank evaluation
- `MEMORYCAPACITY_README.md` - Memory capacity task
- `IPC_README.md` - Information processing capacity
