# Nonlinear Memory Benchmark

## Overview

The **Nonlinear Memory Benchmark** is a task designed to evaluate the memory-nonlinearity trade-off in reservoir computing systems. It maps how well a reservoir can handle tasks that require both temporal memory and nonlinear processing.

## Task Formulation

The benchmark is based on a simple nonlinear function-approximation task with tunable delay (memory) and tunable nonlinearity:

```
y(t) = f(s(t − τ)) = sin(ν * s(t − τ))
```

Where:
- **s(t)**: Input signal
- **τ (tau)**: Delay in time steps (controls memory depth required)
- **ν (nu)**: Nonlinearity strength parameter (controls nonlinearity required)

### Task Difficulty

The difficulty is controlled by two independent parameters:

1. **Larger τ → Longer memory required**
   - Tests the reservoir's ability to retain information over time
   - Linear reservoirs typically excel at large τ values

2. **Larger ν → Stronger nonlinearity required**
   - Tests the reservoir's ability to perform nonlinear transformations
   - Nonlinear reservoirs typically excel at large ν values

## Target Generation

For each time step t ≥ τ:

```python
y(t) = sin(ν * s(t − τ))
```

For t < τ, set y(t) = 0 (these points are excluded during training/testing).

## Benchmark Sweep

To map the memory-nonlinearity trade-off surface, the benchmark evaluates over a grid of parameter pairs (τ, ν).

### Default Parameter Ranges

- **τ**: [1, 2, 3, 4, 5, 6, 7, 8] (delay in time steps)
- **ν**: [0.1, 0.3, 1.0, 3.0, 10.0] (nonlinearity strength)

### Evaluation Process

For each pair (τ, ν):

1. Generate input signal s(t) and corresponding targets y(t)
2. Train a linear readout (Ridge regression) on reservoir states to minimize MSE
3. Compute normalized test error:
   ```
   E(τ, ν) = ⟨(y(t) − ŷ(t))²⟩ / Var(y(t))
   ```
4. Compute capacity:
   ```
   C(τ, ν) = max(0, 1 − E(τ, ν))
   ```

### Results Visualization

The benchmark produces:

1. **Capacity Heatmap C(τ, ν)**: 2D visualization of performance across the parameter space
2. **Memory Performance Plot**: Average capacity vs. delay τ (averaged over ν)
3. **Nonlinearity Performance Plot**: Average capacity vs. nonlinearity ν (averaged over τ)

## Expected Outcomes

The benchmark reveals the fundamental trade-off in reservoir computing:

### Linear Reservoirs
- ✅ Excel at large τ (long-memory tasks)
- ❌ Fail at high ν (high nonlinearity)
- **Capacity profile**: High performance for small ν, degrades as delay increases

### Nonlinear Reservoirs
- ✅ Excel at small τ (high-nonlinearity tasks)
- ❌ Lose long-term memory capability
- **Capacity profile**: Can handle large ν, but performance drops quickly with increasing τ

### Optimal "Edge-of-Chaos" Reservoirs
- 🎯 Balance both memory and nonlinearity
- Trace the Pareto front between the two extremes
- Examples: Gallicchio's ES2N, properly tuned Echo State Networks

## Usage Example

```python
from rcbench import NonlinearMemoryEvaluator, ElecResDataset

# Load measurement data
dataset = ElecResDataset("measurement_file.txt")
input_voltages = dataset.get_input_voltages()
nodes_output = dataset.get_node_voltages()

# Get input signal from first input node
input_signal = input_voltages[dataset.input_nodes[0]]

# Create evaluator
evaluator = NonlinearMemoryEvaluator(
    input_signal=input_signal,
    nodes_output=nodes_output,
    tau_values=[1, 2, 3, 4, 5, 6, 7, 8],
    nu_values=[0.1, 0.3, 1.0, 3.0, 10.0],
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
print(f"Average capacity: {summary['average_capacity']:.4f}")
print(f"Best (τ, ν): ({summary['best_tau']}, {summary['best_nu']})")

# Analyze trade-offs
tradeoff = evaluator.get_memory_vs_nonlinearity_tradeoff()

# Generate plots
evaluator.plot_results(save_dir="./results")
```

## API Reference

### NonlinearMemoryEvaluator

#### Constructor Parameters

- `input_signal` (array-like): Input stimulation signal
- `nodes_output` (np.ndarray): Reservoir node outputs, shape [time_steps, n_nodes]
- `tau_values` (List[int], optional): Delay values to test. Default: [1-8]
- `nu_values` (List[float], optional): Nonlinearity strengths to test. Default: [0.1, 0.3, 1.0, 3.0, 10.0]
- `random_state` (int): Random seed for reproducibility
- `node_names` (List[str], optional): Names of the nodes
- `plot_config` (BasePlotConfig, optional): Plotting configuration

#### Key Methods

##### `run_parameter_sweep()`
Runs the complete benchmark across all (τ, ν) combinations.

**Parameters:**
- `feature_selection_method` (str): 'kbest', 'pca', etc.
- `num_features` (int or 'all'): Number of features to use
- `modeltype` (str): 'Ridge' or 'Linear'
- `regression_alpha` (float): Ridge regularization parameter
- `train_ratio` (float): Training data ratio
- `metric` (str): 'NMSE', 'RNMSE', 'MSE', or 'Capacity'

**Returns:**
- Dictionary with results, capacity_matrix, error_matrix, and metadata

##### `run_evaluation(tau, nu, ...)`
Evaluates a single (τ, ν) combination.

**Parameters:**
- `tau` (int): Specific delay value
- `nu` (float): Specific nonlinearity value
- Other parameters same as `run_parameter_sweep()`

**Returns:**
- Dictionary with error, capacity, predictions, and model

##### `get_best_performance()`
Returns the (τ, ν) combination with highest capacity.

**Returns:**
- Dictionary with best_tau, best_nu, capacity, and full results

##### `get_memory_vs_nonlinearity_tradeoff()`
Analyzes average performance along each dimension.

**Returns:**
- Dictionary with:
  - `memory_performance`: Average capacity for each τ (over ν)
  - `nonlinearity_performance`: Average capacity for each ν (over τ)

##### `plot_results(save_dir=None)`
Generates visualization plots.

Creates:
1. Capacity heatmap C(τ, ν)
2. Memory vs delay plot
3. Nonlinearity performance plot

##### `summary()`
Returns comprehensive summary statistics.

**Returns:**
- Dictionary with statistics, best parameters, and performance metrics

## Interpretation Guide

### Reading the Capacity Matrix

```
High capacity (C → 1): Reservoir successfully performs the task
Low capacity (C → 0):  Reservoir fails at the task
```

### Common Patterns

1. **"Memory Wall"**: Capacity drops sharply as τ increases
   - Indicates limited temporal memory
   - Common in highly nonlinear systems

2. **"Nonlinearity Ceiling"**: Capacity drops as ν increases
   - Indicates limited nonlinear processing
   - Common in linear or weakly nonlinear systems

3. **"Balanced Profile"**: Gradual degradation in both dimensions
   - Indicates well-tuned reservoir at edge-of-chaos
   - Optimal for general-purpose computing

## Metrics

### Normalized Mean Squared Error (NMSE)
```
NMSE = MSE(y_true, y_pred) / Var(y_true)
```
- Lower is better
- NMSE = 0: Perfect prediction
- NMSE ≥ 1: Worse than predicting the mean

### Capacity
```
C = max(0, 1 - NMSE)
```
- Higher is better
- C = 1: Perfect prediction
- C = 0: No better than predicting the mean
- Range: [0, 1]

## References

This benchmark is based on concepts from:
- Memory-nonlinearity trade-off in reservoir computing
- Jaeger's Echo State Networks
- Gallicchio's Edge of Stability analysis (ES2N)
- Nonlinear time series prediction benchmarks

## See Also

- `MemoryCapacityEvaluator`: For pure linear memory capacity
- `NarmaEvaluator`: For NARMA nonlinear benchmark
- `NltEvaluator`: For nonlinear transformation tasks

