# Kernel Rank (KR) Task Manual

## Overview

The **Kernel Rank (KR)** evaluator measures the computational capacity of a reservoir by analyzing the effective rank of the state matrix (or its kernel matrix). It quantifies how many linearly independent computational dimensions are available in the reservoir's response to input stimulation.

## Theoretical Background

### Reference

> Wringe, C., Trefzer, M., & Stepney, S. (2025). "Reservoir Computing Benchmarks: a tutorial review and critique". arXiv:2405.06561

### Key Concepts

1. **Kernel Rank**: The effective dimensionality of the reservoir state space. A higher kernel rank indicates more independent computational dimensions.

2. **Combined Mode (Recommended)**: Concatenates the input signal with reservoir states to capture the combined dynamics of input and reservoir responses, as recommended by Wringe et al. (2025).

3. **Kernel Types**:
   - **Linear Kernel**: Computes the Gram matrix K = X.T @ X in feature space
   - **RBF (Radial Basis Function)**: Gaussian kernel K[i,j] = exp(-||x_i - x_j||² / (2σ²))

### Mathematical Formulation

Given a state matrix X with shape (T, N) where T is the number of timesteps and N is the number of features:

#### Linear Kernel (Feature Space)
```
K = X.T @ X    (shape: N × N)
```

#### RBF Kernel (Sample Space)
```
K[i,j] = exp(-||x_i - x_j||² / (2σ²))    (shape: T × T)
```

#### Effective Rank
The effective rank is computed via Singular Value Decomposition (SVD):

```
rank_effective = count(σ_i > threshold × σ_max)
```

where σ_i are the singular values of K, and σ_max is the maximum singular value.

### Theoretical Maximum

For a system with N linearly independent nodes, the maximum kernel rank is N. Including the input signal (combined mode) gives a maximum of N + 1.

## Usage

### Basic Usage (Combined Mode - Recommended)

```python
from rcbench import ElecResDataset
from rcbench.tasks.kernelrank import KernelRankEvaluator
from rcbench.logger import get_logger

logger = get_logger(__name__)

# Load data
dataset = ElecResDataset("measurement_file.txt")

# Extract signals
input_voltages = dataset.get_input_voltages()
input_signal = input_voltages[dataset.input_nodes[0]]
nodes_output = dataset.get_node_voltages()

# Create evaluator with input signal (Combined Mode)
evaluator = KernelRankEvaluator(
    nodes_output=nodes_output,
    input_signal=input_signal,  # Include input for combined mode
    kernel='linear',
    threshold=1e-6
)

# Run evaluation
results = evaluator.run_evaluation()

# Display results
logger.output(f"Kernel Rank: {results['kernel_rank']}")
logger.output(f"Features used: {results['n_features']}")
logger.output(f"Include input: {results['include_input']}")
```

### Nodes-Only Mode (Backward Compatible)

```python
# Create evaluator without input signal
evaluator = KernelRankEvaluator(
    nodes_output=nodes_output,
    kernel='linear',
    threshold=1e-6
)

results = evaluator.run_evaluation()
logger.output(f"Kernel Rank: {results['kernel_rank']}")
```

### Using RBF Kernel

```python
# RBF kernel for non-linear feature mappings
evaluator = KernelRankEvaluator(
    nodes_output=nodes_output,
    input_signal=input_signal,
    kernel='rbf',
    sigma=1.0,
    threshold=1e-6
)

results = evaluator.run_evaluation()
logger.output(f"Kernel Rank (RBF): {results['kernel_rank']}")
```

## Parameters

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nodes_output` | np.ndarray | Required | Reservoir states, shape [T, N] |
| `input_signal` | np.ndarray | None | Input signal, shape [T,] or [T, 1]. If provided, concatenated with nodes for combined mode |
| `kernel` | str | 'linear' | Kernel type: 'linear' or 'rbf' |
| `sigma` | float | 1.0 | RBF kernel parameter (ignored for linear) |
| `threshold` | float | 1e-6 | Relative threshold for counting significant singular values |

### Key Methods

#### `run_evaluation()`

Runs the kernel rank evaluation.

**Returns:** Dictionary containing:
- `kernel_rank`: The computed effective rank
- `singular_values`: All singular values of the kernel matrix
- `kernel`: The kernel type used
- `sigma`: Sigma parameter (for RBF kernel)
- `threshold`: The threshold used for rank computation
- `n_features`: Number of features (input + nodes or just nodes)
- `n_samples`: Number of time samples
- `include_input`: Whether input signal was included

#### `compute_kernel_matrix()`

Computes the kernel (Gram) matrix.

**Returns:** 
- Linear kernel: Feature-space Gram matrix, shape (N, N)
- RBF kernel: Sample-space kernel matrix, shape (T, T)

#### `compute_kernel_rank()`

Computes the effective rank using SVD.

**Returns:** Tuple of (effective_rank, singular_values)

## Output

### Results Dictionary

```python
{
    'kernel_rank': int,          # Effective rank
    'singular_values': ndarray,  # All singular values (sorted descending)
    'kernel': str,               # 'linear' or 'rbf'
    'sigma': float,              # RBF parameter
    'threshold': float,          # Threshold used
    'n_features': int,           # Number of features
    'n_samples': int,            # Number of time samples
    'include_input': bool        # Whether input was included
}
```

## Interpretation Guide

### Understanding the Results

1. **High Kernel Rank (close to N)**: The reservoir nodes provide highly independent responses, indicating rich computational capacity.

2. **Low Kernel Rank (much less than N)**: Many nodes exhibit correlated or redundant responses, limiting effective computation.

3. **Linear vs RBF**: 
   - Linear kernel captures linear dependencies
   - RBF kernel can capture nonlinear relationships in the data

### Performance Benchmarks

For N features:
- **KR ≈ N**: Excellent (maximal linear independence)
- **KR ≈ 0.7N - 0.9N**: Very good
- **KR ≈ 0.4N - 0.7N**: Good
- **KR < 0.4N**: Limited effective dimensionality

### Combined vs Nodes-Only Mode

| Mode | Use Case |
|------|----------|
| Combined (input+nodes) | Standard evaluation (recommended per Wringe et al.) |
| Nodes-only | Measuring intrinsic reservoir dimensionality |

## Examples

### Example 1: Compare Kernel Types

```python
# Linear kernel
kr_linear = KernelRankEvaluator(
    nodes_output=nodes_output,
    input_signal=input_signal,
    kernel='linear',
    threshold=1e-6
)
result_linear = kr_linear.run_evaluation()

# RBF kernel
kr_rbf = KernelRankEvaluator(
    nodes_output=nodes_output,
    input_signal=input_signal,
    kernel='rbf',
    sigma=1.0,
    threshold=1e-6
)
result_rbf = kr_rbf.run_evaluation()

print(f"Linear Kernel Rank: {result_linear['kernel_rank']}")
print(f"RBF Kernel Rank: {result_rbf['kernel_rank']}")
```

### Example 2: Analyze Singular Value Spectrum

```python
import matplotlib.pyplot as plt
import numpy as np

evaluator = KernelRankEvaluator(
    nodes_output=nodes_output,
    input_signal=input_signal,
    kernel='linear',
    threshold=1e-6
)
results = evaluator.run_evaluation()

# Plot singular value spectrum
singular_values = results['singular_values']
plt.figure(figsize=(10, 5))

# Linear scale
plt.subplot(1, 2, 1)
plt.plot(singular_values, 'b-o')
plt.xlabel('Index')
plt.ylabel('Singular Value')
plt.title('Singular Value Spectrum')

# Log scale
plt.subplot(1, 2, 2)
plt.semilogy(singular_values, 'b-o')
plt.axhline(y=results['threshold'] * singular_values[0], color='r', linestyle='--', label='Threshold')
plt.xlabel('Index')
plt.ylabel('Singular Value (log)')
plt.title(f'Effective Rank: {results["kernel_rank"]}')
plt.legend()

plt.tight_layout()
plt.show()
```

### Example 3: Synthetic Data

```python
import numpy as np

# Generate synthetic reservoir data
np.random.seed(42)
n_samples = 3000
n_nodes = 20

# Input signal
input_signal = np.random.uniform(-1, 1, n_samples)

# Synthetic node outputs with varying degrees of independence
nodes_output = np.zeros((n_samples, n_nodes))
for i in range(n_nodes):
    delay = np.random.randint(0, 10)
    noise_level = 0.1 * np.random.random()
    nodes_output[:, i] = np.tanh(np.roll(input_signal, delay) + noise_level * np.random.randn(n_samples))

# Evaluate
evaluator = KernelRankEvaluator(
    nodes_output=nodes_output,
    input_signal=input_signal,
    kernel='linear',
    threshold=1e-6
)

results = evaluator.run_evaluation()
print(f"Kernel Rank: {results['kernel_rank']} / {results['n_features']} features")
print(f"Efficiency: {results['kernel_rank'] / results['n_features'] * 100:.1f}%")
```

## Computational Efficiency

### Linear Kernel Optimization

For linear kernels, the implementation uses feature-space computation:
- Computes K = X.T @ X (shape N × N) instead of K = X @ X.T (shape T × T)
- Both have identical non-zero eigenvalues: rank(X.T @ X) = rank(X @ X.T) = rank(X)
- When N << T (typical: 50 nodes, 3000+ samples), this is much faster: O(N³) vs O(T³)

### RBF Kernel

For RBF kernels, sample-space computation is required:
- Computes pairwise distances between all samples
- Complexity: O(T² × N) for distance computation, O(T³) for SVD
- More expensive for large T

## Tips and Best Practices

1. **Use Combined Mode**: Include input signal for comprehensive evaluation as per Wringe et al. (2025).

2. **Choose Appropriate Threshold**:
   - Default 1e-6 works well for most cases
   - Increase threshold for noisy data (1e-4 to 1e-3)
   - Decrease for very clean data (1e-8)

3. **Linear vs RBF**:
   - Start with linear kernel for basic analysis
   - Use RBF when nonlinear relationships are expected

4. **RBF Sigma Selection**:
   - σ = 1.0 is a good starting point
   - Larger σ: smoother kernel, potentially higher rank
   - Smaller σ: more localized kernel, potentially lower rank

## Common Issues

### Issue 1: Kernel Rank Very Low

**Possible Causes:**
- Highly correlated node responses
- Insufficient input stimulation
- Node saturation (all in same nonlinear region)

**Solutions:**
- Check node response diversity
- Use different input signal characteristics
- Adjust reservoir parameters

### Issue 2: Numerical Instability

**Possible Causes:**
- Very small or very large values in state matrix
- Ill-conditioned kernel matrix

**Solutions:**
- Normalize state matrix before evaluation
- Adjust threshold parameter
- Check for NaN or Inf values in data

### Issue 3: RBF Kernel Rank Always T

**Possible Causes:**
- σ too small (each sample becomes nearly orthogonal)
- Threshold too low

**Solutions:**
- Increase σ parameter
- Increase threshold value

## Relationship to Other Metrics

### Kernel Rank vs Generalization Rank

| Aspect | Kernel Rank (KR) | Generalization Rank (GR) |
|--------|------------------|--------------------------|
| Input | Time series data | Multiple input streams |
| Measures | Linear independence of features | Response similarity under noise |
| Use case | Single experiment | Noise robustness analysis |

### Kernel Rank vs Memory Capacity

| Aspect | Kernel Rank | Memory Capacity |
|--------|-------------|-----------------|
| Measures | State space dimensionality | Ability to recall past inputs |
| Method | SVD of kernel matrix | Regression on delayed targets |
| Maximum | N (or N+1 with input) | N |

## References

1. Wringe, C., Trefzer, M., & Stepney, S. (2025). "Reservoir Computing Benchmarks: a tutorial review and critique". arXiv:2405.06561

2. Dambre, J., Verstraeten, D., Schrauwen, B. & Massar, S. (2012). "Information Processing Capacity of Dynamical Systems". Scientific Reports 2, 514.

3. Verstraeten, D., Schrauwen, B., D'Haene, M., & Stroobandt, D. (2007). "An experimental unification of reservoir computing methods". Neural Networks, 20(3), 391-403.

## See Also

- `GENERALIZATIONRANK_README.md` - Generalization rank evaluation
- `MEMORYCAPACITY_README.md` - Memory capacity task
- `IPC_README.md` - Information processing capacity
