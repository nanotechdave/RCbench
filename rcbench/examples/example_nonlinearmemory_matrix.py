"""
Nonlinear Memory Benchmark with Synthetic Data Example

This example demonstrates how to perform Nonlinear Memory evaluation using artificially
generated data without using any dataset classes (ReservoirDataset or ElecResDataset).

This approach is useful when:
1. You have your own data format and don't want to convert to the dataset classes
2. You want to test the evaluation framework with synthetic data
3. You want to explore memory-nonlinearity trade-offs with controlled reservoir properties

The example creates:
- A time vector with regularly spaced samples
- An input signal (random or structured)
- A matrix of synthetic node outputs with tunable memory and nonlinearity characteristics
- Evaluates the reservoir's capacity across different (τ, ν) parameter combinations

This benchmark reveals the fundamental memory-nonlinearity trade-off:
- Linear reservoirs: good at memory (large τ), poor at nonlinearity (large ν)
- Nonlinear reservoirs: good at nonlinearity, poor at long-term memory
- Optimal reservoirs: balance both capabilities at the edge-of-chaos

Author: Davide Pilati
Date: 2025
"""

import logging
import numpy as np
import matplotlib.pyplot as plt

from rcbench.tasks.nonlinearmemory import NonlinearMemoryEvaluator
from rcbench.visualization.plot_config import NonlinearMemoryPlotConfig
from rcbench.logger import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.INFO)


def generate_synthetic_reservoir_data(n_samples=2500, 
                                     n_nodes=20, 
                                     input_amplitude=1.0, 
                                     noise_level=0.05,
                                     reservoir_type='balanced'):
    """
    Generate synthetic reservoir computing data for testing Nonlinear Memory benchmark.
    
    The reservoir_type parameter controls the memory-nonlinearity trade-off:
    - 'linear': Strong memory, weak nonlinearity (linear reservoirs)
    - 'nonlinear': Weak memory, strong nonlinearity (highly nonlinear reservoirs)
    - 'balanced': Balanced memory and nonlinearity (edge-of-chaos)
    
    Args:
        n_samples (int): Number of time samples
        n_nodes (int): Number of reservoir nodes
        input_amplitude (float): Amplitude of input signal
        noise_level (float): Level of noise to add to node outputs
        reservoir_type (str): Type of reservoir ('linear', 'nonlinear', 'balanced')
    
    Returns:
        tuple: (time_vector, input_signal, nodes_output_matrix, node_names)
    """
    # Create time vector
    dt = 0.01  # time step
    time_vector = np.arange(n_samples) * dt
    
    # Generate random input signal (white noise)
    np.random.seed(42)  # for reproducibility
    input_signal = input_amplitude * np.random.uniform(-1, 1, n_samples)
    
    # Set reservoir parameters based on type
    if reservoir_type == 'linear':
        # Linear reservoir: high memory decay, low nonlinearity
        decay_range = (0.85, 0.95)  # High decay = long memory
        nonlin_strength_range = (0.1, 0.3)  # Low nonlinearity
        logger.info("Creating LINEAR reservoir (strong memory, weak nonlinearity)")
        
    elif reservoir_type == 'nonlinear':
        # Nonlinear reservoir: low memory decay, high nonlinearity
        decay_range = (0.3, 0.5)  # Low decay = short memory
        nonlin_strength_range = (1.5, 3.0)  # High nonlinearity
        logger.info("Creating NONLINEAR reservoir (weak memory, strong nonlinearity)")
        
    else:  # 'balanced'
        # Balanced reservoir: moderate memory and nonlinearity
        decay_range = (0.6, 0.8)  # Moderate decay
        nonlin_strength_range = (0.5, 1.2)  # Moderate nonlinearity
        logger.info("Creating BALANCED reservoir (edge-of-chaos)")
    
    # Generate synthetic node outputs
    nodes_output = np.zeros((n_samples, n_nodes))
    
    for i in range(n_nodes):
        # Each node has different characteristics
        delay = np.random.randint(0, 8)  # Random delay 0-7 steps
        decay = np.random.uniform(*decay_range)  # Decay factor
        gain = 0.5 + 0.8 * np.random.random()  # Input gain
        nonlin_strength = np.random.uniform(*nonlin_strength_range)  # Nonlinearity strength
        
        # Create node response with memory
        node_signal = np.zeros(n_samples)
        for t in range(delay, n_samples):
            # Current input + decayed previous state
            current_input = gain * input_signal[t - delay]
            
            if t > 0:
                # Memory: combine current input with previous state
                node_signal[t] = current_input + decay * node_signal[t-1]
            else:
                node_signal[t] = current_input
            
            # Apply nonlinearity with controlled strength
            # tanh(α*x) where α controls nonlinearity strength
            node_signal[t] = np.tanh(nonlin_strength * node_signal[t])
        
        # Add noise
        node_signal += noise_level * np.random.normal(0, 1, n_samples)
        nodes_output[:, i] = node_signal
    
    # Create node names
    node_names = [f'Node_{i+1}' for i in range(n_nodes)]
    
    logger.info(f"Generated reservoir properties:")
    logger.info(f"  - Decay range: {decay_range}")
    logger.info(f"  - Nonlinearity range: {nonlin_strength_range}")
    logger.info(f"  - Max delay: {8} steps")
    
    return time_vector, input_signal, nodes_output, node_names


def main():
    """Main function to demonstrate Nonlinear Memory benchmark with synthetic data."""
    
    logger.info("="*60)
    logger.info("NONLINEAR MEMORY BENCHMARK - SYNTHETIC DATA")
    logger.info("="*60 + "\n")
    
    # ====================================================================
    # STEP 1: Generate synthetic reservoir data
    # ====================================================================
    
    logger.info("Step 1: Generating synthetic reservoir data...")
    
    # Choose reservoir type to demonstrate different trade-offs:
    # - 'linear': Good at memory (high τ), poor at nonlinearity (high ν)
    # - 'nonlinear': Good at nonlinearity (high ν), poor at memory (high τ)
    # - 'balanced': Balanced performance (edge-of-chaos)
    
    reservoir_type = 'balanced'  # Change this to 'linear' or 'nonlinear' to see differences
    
    time_vector, input_signal, nodes_output, node_names = generate_synthetic_reservoir_data(
        n_samples=2500,
        n_nodes=20,
        input_amplitude=0.8,
        noise_level=0.05,
        reservoir_type=reservoir_type
    )
    
    logger.info(f"\nGenerated data:")
    logger.info(f"  - Time samples: {len(time_vector)}")
    logger.info(f"  - Input signal shape: {input_signal.shape}")
    logger.info(f"  - Input range: [{input_signal.min():.3f}, {input_signal.max():.3f}]")
    logger.info(f"  - Nodes output shape: {nodes_output.shape}")
    logger.info(f"  - Number of nodes: {len(node_names)}\n")
    
    # ====================================================================
    # STEP 2: Configure benchmark parameters
    # ====================================================================
    
    logger.info("Step 2: Configuring benchmark parameters...")
    
    # Define parameter ranges for the benchmark
    # You can customize these based on your needs
    tau_values = [1, 2, 3, 4, 5, 6, 7, 8]  # Delay values (memory depth)
    nu_values = [0.1, 0.3, 1.0, 3.0, 10.0]  # Nonlinearity strength values
    
    logger.info(f"Parameter sweep configuration:")
    logger.info(f"  - τ (delay) values: {tau_values}")
    logger.info(f"  - ν (nonlinearity) values: {nu_values}")
    logger.info(f"  - Total combinations: {len(tau_values) * len(nu_values)}\n")
    
    # ====================================================================
    # STEP 3: Create evaluator
    # ====================================================================
    
    logger.info("Step 3: Creating Nonlinear Memory Evaluator...")
    
    # Create plot configuration
    plot_config = NonlinearMemoryPlotConfig(
        save_dir=None,  # Set to './results' to save plots
        plot_capacity_heatmap=True,
        plot_tradeoff_analysis=True,
        show_plot=True
    )
    
    evaluator = NonlinearMemoryEvaluator(
        input_signal=input_signal,
        nodes_output=nodes_output,
        tau_values=tau_values,
        nu_values=nu_values,
        random_state=42,
        node_names=node_names,
        plot_config=plot_config
    )
    
    logger.info("Evaluator created successfully!\n")
    
    # ====================================================================
    # STEP 4: Run parameter sweep
    # ====================================================================
    
    logger.info("Step 4: Running parameter sweep (this may take a moment)...")
    logger.info("-" * 60 + "\n")
    
    results = evaluator.run_parameter_sweep(
        feature_selection_method='kbest',  # or 'pca'
        num_features='all',  # Use all features, or specify a number like 10
        modeltype='Ridge',
        regression_alpha=0.1,
        train_ratio=0.8,
        metric='NMSE'  # Options: 'NMSE', 'RNMSE', 'MSE'
    )
    
    logger.info("\nParameter sweep completed!\n")
    
    # ====================================================================
    # STEP 5: Display results
    # ====================================================================
    
    logger.output("\n" + "="*60)
    logger.output("BENCHMARK RESULTS SUMMARY")
    logger.output("="*60)
    
    summary = evaluator.summary()
    
    logger.output(f"\nReservoir Type: {reservoir_type.upper()}")
    
    logger.output(f"\nParameter Space:")
    logger.output(f"  - τ values: {summary['tau_values']}")
    logger.output(f"  - ν values: {summary['nu_values']}")
    logger.output(f"  - Total combinations: {summary['total_combinations']}")
    
    logger.output(f"\nOverall Performance:")
    logger.output(f"  - Average capacity: {summary['average_capacity']:.4f}")
    logger.output(f"  - Maximum capacity: {summary['max_capacity']:.4f}")
    logger.output(f"  - Minimum capacity: {summary['min_capacity']:.4f}")
    
    logger.output(f"\nBest Performance:")
    logger.output(f"  - Best τ (delay): {summary['best_tau']}")
    logger.output(f"  - Best ν (nonlinearity): {summary['best_nu']}")
    logger.output(f"  - Best capacity: {summary['best_capacity']:.4f}")
    
    logger.output(f"\nFeature Selection:")
    logger.output(f"  - Method: {results['feature_selection_method']}")
    logger.output(f"  - Features used: {results['num_features']}")
    
    # ====================================================================
    # STEP 6: Analyze trade-offs
    # ====================================================================
    
    logger.output("\n" + "="*60)
    logger.output("MEMORY vs NONLINEARITY TRADE-OFF ANALYSIS")
    logger.output("="*60)
    
    tradeoff = evaluator.get_memory_vs_nonlinearity_tradeoff()
    
    logger.output("\nMemory Performance (averaged over ν):")
    logger.output("(Shows how capacity changes with increasing delay)")
    for tau, perf in zip(tradeoff['tau_values'], tradeoff['memory_performance']):
        bar = "█" * int(perf * 30)  # Visual bar
        logger.output(f"  τ={tau:2d}: {perf:.4f} {bar}")
    
    logger.output("\nNonlinearity Performance (averaged over τ):")
    logger.output("(Shows how capacity changes with increasing nonlinearity)")
    for nu, perf in zip(tradeoff['nu_values'], tradeoff['nonlinearity_performance']):
        bar = "█" * int(perf * 30)  # Visual bar
        logger.output(f"  ν={nu:5.1f}: {perf:.4f} {bar}")
    
    # ====================================================================
    # STEP 7: Display capacity matrix
    # ====================================================================
    
    logger.output("\n" + "="*60)
    logger.output("CAPACITY MATRIX C(τ, ν)")
    logger.output("="*60)
    logger.output("\nRows: τ (delay), Columns: ν (nonlinearity)")
    logger.output("Higher values = better performance\n")
    
    capacity_matrix = results['capacity_matrix']
    
    # Print header
    header = "   τ  |" + "".join([f"  ν={nu:4.1f}" for nu in nu_values])
    logger.output(header)
    logger.output("-" * len(header))
    
    # Print each row
    for i, tau in enumerate(tau_values):
        row = f"  {tau:2d}  |"
        for j in range(len(nu_values)):
            capacity = capacity_matrix[i, j]
            row += f"  {capacity:6.4f}"
        logger.output(row)
    
    # ====================================================================
    # STEP 8: Interpretation guide
    # ====================================================================
    
    logger.output("\n" + "="*60)
    logger.output("INTERPRETATION GUIDE")
    logger.output("="*60)
    
    # Analyze the pattern
    memory_trend = tradeoff['memory_performance']
    nonlin_trend = tradeoff['nonlinearity_performance']
    
    memory_decay = memory_trend[0] - memory_trend[-1]
    nonlin_decay = nonlin_trend[0] - nonlin_trend[-1]
    
    logger.output("\nObserved Trade-off Pattern:")
    
    if memory_decay < 0.2 and nonlin_decay > 0.5:
        logger.output("  ✓ STRONG MEMORY, WEAK NONLINEARITY")
        logger.output("    - Capacity remains high as delay (τ) increases")
        logger.output("    - Capacity drops sharply as nonlinearity (ν) increases")
        logger.output("    → This is typical of LINEAR reservoirs")
        
    elif memory_decay > 0.5 and nonlin_decay < 0.3:
        logger.output("  ✓ WEAK MEMORY, STRONG NONLINEARITY")
        logger.output("    - Capacity drops quickly as delay (τ) increases")
        logger.output("    - Capacity remains high as nonlinearity (ν) increases")
        logger.output("    → This is typical of HIGHLY NONLINEAR reservoirs")
        
    else:
        logger.output("  ✓ BALANCED MEMORY-NONLINEARITY")
        logger.output("    - Moderate degradation with increasing delay (τ)")
        logger.output("    - Moderate degradation with increasing nonlinearity (ν)")
        logger.output("    → This suggests an EDGE-OF-CHAOS reservoir")
    
    logger.output(f"\nMemory decay: {memory_decay:.4f}")
    logger.output(f"Nonlinearity decay: {nonlin_decay:.4f}")
    
    # ====================================================================
    # STEP 9: Generate plots
    # ====================================================================
    
    logger.info("\n" + "="*60)
    logger.info("Generating visualization plots...")
    logger.info("="*60 + "\n")
    
    evaluator.plot_results()
    
    logger.info("\n" + "="*60)
    logger.info("BENCHMARK COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    
    # ====================================================================
    # Optional: Compare different reservoir types
    # ====================================================================
    
    logger.output("\n" + "="*60)
    logger.output("EXPERIMENT SUGGESTIONS")
    logger.output("="*60)
    
    logger.output("\n1. Try different reservoir types:")
    logger.output("  - reservoir_type='linear'    → Strong memory, weak nonlinearity")
    logger.output("  - reservoir_type='nonlinear' → Weak memory, strong nonlinearity")
    logger.output("  - reservoir_type='balanced'  → Balanced (edge-of-chaos)")
    
    logger.output("\n2. Try different metrics:")
    logger.output("  - metric='NMSE'   → Normalized Mean Squared Error (default)")
    logger.output("  - metric='RNMSE'  → Root Normalized Mean Squared Error")
    logger.output("  - metric='MSE'    → Mean Squared Error")
    
    logger.output("\n3. Test individual (τ, ν) combinations:")
    logger.output("  Use evaluator.run_evaluation(tau=X, nu=Y, metric='NMSE', ...)")
    
    logger.output("\nObserve how the capacity matrix C(τ, ν) changes!")
    
    return evaluator, results


def compare_metrics():
    """
    Optional: Compare different evaluation metrics.
    Shows how NMSE, RNMSE, and MSE relate to each other.
    """
    logger.info("\n" + "="*60)
    logger.info("COMPARING EVALUATION METRICS")
    logger.info("="*60 + "\n")
    
    # Generate data
    n_samples = 2000
    np.random.seed(42)
    input_signal = np.random.uniform(-1, 1, n_samples)
    
    # Create balanced reservoir
    n_nodes = 15
    nodes_output = np.zeros((n_samples, n_nodes))
    
    for i in range(n_nodes):
        delay = np.random.randint(0, 8)
        decay = np.random.uniform(0.6, 0.8)
        gain = 0.5 + 0.8 * np.random.random()
        nonlin_strength = np.random.uniform(0.5, 1.2)
        
        node_signal = np.zeros(n_samples)
        for t in range(delay, n_samples):
            current_input = gain * input_signal[t - delay]
            if t > 0:
                node_signal[t] = current_input + decay * node_signal[t-1]
            else:
                node_signal[t] = current_input
            node_signal[t] = np.tanh(nonlin_strength * node_signal[t])
        
        node_signal += 0.05 * np.random.normal(0, 1, n_samples)
        nodes_output[:, i] = node_signal
    
    # Test different metrics
    metrics = ['NMSE', 'RNMSE', 'MSE']
    metric_results = {}
    
    for metric in metrics:
        logger.info(f"Evaluating with {metric}...")
        
        evaluator = NonlinearMemoryEvaluator(
            input_signal=input_signal,
            nodes_output=nodes_output,
            tau_values=[1, 2, 3, 4, 5],
            nu_values=[0.1, 1.0, 10.0],
            random_state=42
        )
        
        results = evaluator.run_parameter_sweep(
            feature_selection_method='kbest',
            num_features='all',
            modeltype='Ridge',
            regression_alpha=0.1,
            train_ratio=0.8,
            metric=metric
        )
        
        metric_results[metric] = {
            'avg_capacity': np.nanmean(results['capacity_matrix']),
            'avg_error': np.nanmean(results['error_matrix']),
            'best_capacity': np.nanmax(results['capacity_matrix'])
        }
    
    # Display comparison
    logger.output("\n" + "="*60)
    logger.output("METRIC COMPARISON RESULTS")
    logger.output("="*60)
    
    for metric in metrics:
        logger.output(f"\n{metric}:")
        logger.output(f"  Average capacity: {metric_results[metric]['avg_capacity']:.4f}")
        logger.output(f"  Average {metric}: {metric_results[metric]['avg_error']:.6f}")
        logger.output(f"  Best capacity: {metric_results[metric]['best_capacity']:.4f}")
    
    logger.output("\nNote: Capacity values should be similar across metrics")
    logger.output("      (since Capacity = 1 - NMSE regardless of error metric)")


def compare_reservoir_types():
    """
    Optional: Run benchmark for all three reservoir types and compare results.
    This demonstrates the fundamental memory-nonlinearity trade-off.
    """
    logger.info("\n" + "="*60)
    logger.info("COMPARING RESERVOIR TYPES")
    logger.info("="*60 + "\n")
    
    reservoir_types = ['linear', 'balanced', 'nonlinear']
    all_results = {}
    
    for res_type in reservoir_types:
        logger.info(f"\nEvaluating {res_type.upper()} reservoir...")
        
        # Generate data
        _, input_signal, nodes_output, node_names = generate_synthetic_reservoir_data(
            n_samples=2000,
            n_nodes=15,
            reservoir_type=res_type
        )
        
        # Create evaluator
        evaluator = NonlinearMemoryEvaluator(
            input_signal=input_signal,
            nodes_output=nodes_output,
            tau_values=[1, 2, 3, 4, 5, 6],
            nu_values=[0.1, 1.0, 10.0],
            random_state=42,
            node_names=node_names
        )
        
        # Run sweep
        results = evaluator.run_parameter_sweep(
            feature_selection_method='kbest',
            num_features='all',
            modeltype='Ridge',
            regression_alpha=0.1,
            train_ratio=0.8
        )
        
        all_results[res_type] = evaluator.summary()
    
    # Display comparison
    logger.output("\n" + "="*60)
    logger.output("COMPARISON RESULTS")
    logger.output("="*60)
    
    for res_type in reservoir_types:
        summary = all_results[res_type]
        logger.output(f"\n{res_type.upper()} Reservoir:")
        logger.output(f"  - Average capacity: {summary['average_capacity']:.4f}")
        logger.output(f"  - Best at: (τ={summary['best_tau']}, ν={summary['best_nu']})")
        logger.output(f"  - Best capacity: {summary['best_capacity']:.4f}")


if __name__ == "__main__":
    # Run main benchmark
    evaluator, results = main()
    
    # Uncomment to compare different evaluation metrics
    # compare_metrics()
    
    # Uncomment to compare different reservoir types
    # compare_reservoir_types()

