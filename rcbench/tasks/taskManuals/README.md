# RCbench Task Manuals

This directory contains comprehensive manuals for each benchmark task in RCbench.

## Available Task Manuals

### Core Benchmark Tasks

1. **[NLT (Nonlinear Transformation)](NLT_README.md)**
   - Tests ability to perform nonlinear waveform transformations
   - Targets: square wave, phase shift, double frequency, triangular wave
   - Difficulty: Moderate
   - Use case: Evaluate instantaneous nonlinear processing

2. **[Memory Capacity](MEMORYCAPACITY_README.md)**
   - Measures linear memory capabilities
   - Tests recall of time-delayed inputs
   - Difficulty: Easy to understand, performance varies with reservoir
   - Use case: Quantify temporal memory depth

3. **[Nonlinear Memory](NONLINEARMEMORY_README.md)**
   - Maps memory-nonlinearity trade-off
   - Task: `y(t) = sin(ν * s(t-τ))`
   - Difficulty: Comprehensive (parameter sweep)
   - Use case: Understand fundamental reservoir capabilities

4. **[NARMA (Nonlinear Auto-Regressive Moving Average)](NARMA_README.md)**
   - Standard temporal nonlinear benchmark
   - Tests memory + nonlinearity combined
   - Difficulty: Moderate to Hard (depends on order)
   - Use case: Evaluate temporal nonlinear dynamics

5. **[Sin(x) Approximation](SINX_README.md)**
   - Tests nonlinear function approximation
   - Task: Map x → sin(x)
   - Difficulty: Moderate
   - Use case: Evaluate instantaneous nonlinear mapping

### Specialized Tasks

6. **Kernel Rank** (Documentation in progress)
   - Evaluates kernel quality and nonlinearity
   - Use case: Measure reservoir dimensionality

7. **Generalization Rank** (Documentation in progress)
   - Assesses cross-dataset generalization
   - Use case: Test robustness and adaptability

## Quick Reference

### Task Selection Guide

| Task | Memory | Nonlinearity | Temporal | Recommended For |
|------|--------|--------------|----------|-----------------|
| **NLT** | Low | High | No | Waveform generation, pattern transformation |
| **Memory Capacity** | High | Low | Yes | Pure memory evaluation |
| **Nonlinear Memory** | High | High | Yes | Comprehensive characterization |
| **NARMA** | High | High | Yes | Standard benchmark, temporal dynamics |
| **Sin(x)** | Low | High | No | Function approximation |
| **Kernel Rank** | Low | High | No | Dimensionality assessment |
| **Generalization Rank** | Medium | Medium | Yes | Robustness testing |

### Typical Performance Metrics (NMSE)

| Task | Excellent | Good | Moderate | Poor |
|------|-----------|------|----------|------|
| **NLT** | < 0.10 | 0.10-0.30 | 0.30-0.50 | > 0.50 |
| **Memory Capacity** | > 0.8N | 0.5N-0.8N | 0.3N-0.5N | < 0.3N |
| **NARMA-2** | < 0.05 | 0.05-0.15 | 0.15-0.30 | > 0.30 |
| **NARMA-10** | < 0.10 | 0.10-0.30 | 0.30-0.50 | > 0.50 |
| **Sin(x)** | < 0.05 | 0.05-0.15 | 0.15-0.30 | > 0.30 |

*N = number of reservoir nodes for Memory Capacity

### Common Parameters

All tasks share these common parameters:

#### Constructor
- `input_signal`: Input signal array
- `nodes_output`: Reservoir states [time_steps, n_nodes]
- `node_names`: Optional node labels
- `plot_config`: Optional plotting configuration

#### Evaluation
- `metric`: 'NMSE', 'RNMSE', or 'MSE'
- `feature_selection_method`: 'kbest', 'pca', or 'none'
- `num_features`: Number of features or 'all'
- `modeltype`: 'Ridge' or 'Linear'
- `regression_alpha`: Regularization parameter
- `train_ratio`: Training data fraction (0-1)

## Getting Started

### 1. Choose a Task

Select based on what you want to evaluate:
- **Quick test**: Sin(x) or NLT
- **Memory focus**: Memory Capacity
- **Comprehensive**: Nonlinear Memory + NARMA
- **Standard benchmark**: NARMA-10

### 2. Load Your Data

```python
from rcbench import ElecResDataset

dataset = ElecResDataset("your_measurement_file.txt")
input_signal = dataset.get_input_voltages()[dataset.input_nodes[0]]
nodes_output = dataset.get_node_voltages()
node_names = dataset.nodes
```

### 3. Run Evaluation

```python
from rcbench import NltEvaluator  # or other evaluator

evaluator = NltEvaluator(
    input_signal=input_signal,
    nodes_output=nodes_output,
    node_names=node_names
)

result = evaluator.run_evaluation(
    metric='NMSE',
    modeltype="Ridge",
    regression_alpha=0.1
)

print(f"Performance: {result['accuracy']:.6f}")
```

### 4. Visualize Results

```python
from rcbench.visualization.plot_config import NLTPlotConfig

plot_config = NLTPlotConfig(
    save_dir="./results",
    show_plot=True
)

evaluator = NltEvaluator(
    input_signal=input_signal,
    nodes_output=nodes_output,
    node_names=node_names,
    plot_config=plot_config
)

evaluator.plot_results()
```

## Best Practices

### Data Requirements

| Task | Min Samples | Recommended | Input Type |
|------|-------------|-------------|------------|
| **NLT** | 500 | 1000-2000 | Sine/Triangle wave |
| **Memory Capacity** | 1000 | 2000-5000 | Random uniform |
| **Nonlinear Memory** | 1500 | 2500-5000 | Random uniform |
| **NARMA** | 1000 | 2000-5000 | Random uniform [0, 0.5] |
| **Sin(x)** | 500 | 1000-3000 | Random uniform |

### Feature Selection Recommendations

- **NLT**: k-best with all features
- **Memory Capacity**: PCA with all or top components
- **Nonlinear Memory**: k-best with all features
- **NARMA**: k-best with all features
- **Sin(x)**: k-best with 5-20 features

### Regularization Guidelines

- **Small datasets (< 1000)**: α = 1.0 - 10.0
- **Medium datasets (1000-5000)**: α = 0.1 - 1.0
- **Large datasets (> 5000)**: α = 0.01 - 0.1

Increase if overfitting, decrease if underfitting.

## Typical Workflow

### Complete Reservoir Characterization

```python
# 1. Quick nonlinearity test
sinx_result = run_sinx_evaluation()

# 2. Memory evaluation
mc_result = run_memory_capacity()

# 3. Combined test
narma_result = run_narma_evaluation(order=10)

# 4. Comprehensive analysis
nlm_result = run_nonlinear_memory_sweep()

# 5. Waveform generation
nlt_results = run_nlt_all_targets()
```

### Performance Comparison

```python
# Compare across tasks
tasks = {
    'Sin(x)': sinx_result['accuracy'],
    'NARMA-10': narma_result['accuracy'],
    'NLT-Square': nlt_results['square_wave']['accuracy'],
    'MC Total': mc_result['total_memory_capacity']
}

for task, performance in tasks.items():
    print(f"{task:15}: {performance:.4f}")
```

## Troubleshooting

### Common Issues Across Tasks

1. **High Error (NMSE > 0.5)**
   - Check reservoir has nonlinearity
   - Verify sufficient nodes (> 10-15)
   - Try different feature selection
   - Check data quality

2. **Overfitting**
   - Increase regularization
   - Reduce number of features
   - Use more training data

3. **Unstable Results**
   - Set random_state for reproducibility
   - Use more training data
   - Check for numerical issues

4. **Slow Evaluation**
   - Reduce max_delay (Memory Capacity)
   - Reduce parameter grid (Nonlinear Memory)
   - Use fewer features
   - Sample data if necessary

## Additional Resources

### Example Scripts

Check `rcbench/examples/` for working examples:
- `example_<task>.py`: Using real measurement data
- `example_<task>_matrix.py`: Using synthetic data

### Main Documentation

See the main [README.md](../../README.md) for:
- Installation instructions
- API reference
- General usage guidelines

### Support

- GitHub Issues: https://github.com/nanotechdave/RCbench/issues
- Documentation: Check individual task README files

## Contributing

To add documentation for new tasks:

1. Follow the format of existing README files
2. Include:
   - Overview and mathematical formulation
   - Usage examples (basic and advanced)
   - Parameter descriptions
   - Output format
   - Interpretation guide
   - Tips and best practices
   - Common issues and solutions
3. Add entry to this index file
4. Update the Quick Reference table

## Version Information

These manuals are for RCbench version 0.1.20

Last updated: January 2025

