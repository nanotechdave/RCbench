#!/usr/bin/env python3
"""
Example script demonstrating the Nonlinear Memory benchmark.

This benchmark evaluates the memory-nonlinearity trade-off of a reservoir
by computing performance on:
    y(t) = sin(ν * s(t - τ))

where τ controls memory depth and ν controls nonlinearity strength.

Author: Davide Pilati
Date: 2025
"""

import logging
import numpy as np
from pathlib import Path

from rcbench import ElecResDataset, NonlinearMemoryEvaluator
from rcbench.visualization.plot_config import NonlinearMemoryPlotConfig
from rcbench.logger import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.INFO)

def main():
    """Main function to demonstrate nonlinear memory benchmark."""
    
    logger.info("=== Nonlinear Memory Benchmark Example ===\n")
    
    # Load data from measurement file (same dataset as NARMA evaluation)
    BASE_DIR = Path(__file__).resolve().parent.parent
    filename = "074_INRiMARC_NWN_Pad129M_gridSE_MemoryCapacity_2024_04_02.txt"
    measurement_file = BASE_DIR.parent / "tests" / "test_files" / filename
    
    logger.info(f"Loading data from: {measurement_file}")
    dataset = ElecResDataset(measurement_file)
    
    # Get information about the nodes
    nodes_info = dataset.summary()
    logger.info(f"Parsed Nodes: {nodes_info}")
    
    # Get input and node voltages
    input_voltages = dataset.get_input_voltages()
    nodes_output = dataset.get_node_voltages()
    
    # Use primary input node (same as NARMA)
    primary_input_node = nodes_info['input_nodes'][0]
    input_signal = input_voltages[primary_input_node]
    
    # Get node names for computation nodes
    node_names = nodes_info['nodes']
    
    logger.info(f"Loaded dataset:")
    logger.info(f"  Input node: {primary_input_node}")
    logger.info(f"  Input signal shape: {input_signal.shape}")
    logger.info(f"  Nodes output shape: {nodes_output.shape}")
    logger.info(f"  Computation nodes: {len(node_names)}\n")
    
    # Define parameter ranges for the benchmark
    tau_values = [1, 2, 3, 4, 5, 6, 7, 8]  # Delay values (memory depth)
    nu_values = [0.1, 0.3, 1.0, 3.0, 10.0]  # Nonlinearity strength values
    
    logger.info("Parameter sweep configuration:")
    logger.info(f"  τ (delay) values: {tau_values}")
    logger.info(f"  ν (nonlinearity) values: {nu_values}")
    logger.info(f"  Total combinations: {len(tau_values) * len(nu_values)}\n")
    
    # Create plot configuration
    plot_config = NonlinearMemoryPlotConfig(
        save_dir=None,  # Set to a directory path to save plots
        plot_capacity_heatmap=True,
        plot_tradeoff_analysis=True,
        show_plot=True
    )
    
    # Create the Nonlinear Memory evaluator
    evaluator = NonlinearMemoryEvaluator(
        input_signal=input_signal,
        nodes_output=nodes_output,
        tau_values=tau_values,
        nu_values=nu_values,
        random_state=42,
        node_names=node_names,
        plot_config=plot_config
    )
    
    # Run the parameter sweep
    logger.info("Running parameter sweep (this may take a while)...\n")
    
    results = evaluator.run_parameter_sweep(
        feature_selection_method='kbest',  # or 'pca'
        num_features='all',  # Use all features, or specify a number
        modeltype='Ridge',
        regression_alpha=0.1,
        train_ratio=0.8,
        metric='NMSE'  # Normalized Mean Squared Error
    )
    
    # Display summary results
    logger.output("\n" + "="*60)
    logger.output("BENCHMARK RESULTS SUMMARY")
    logger.output("="*60)
    
    summary = evaluator.summary()
    
    logger.output(f"\nParameter Space:")
    logger.output(f"  τ values: {summary['tau_values']}")
    logger.output(f"  ν values: {summary['nu_values']}")
    logger.output(f"  Total combinations evaluated: {summary['total_combinations']}")
    
    logger.output(f"\nOverall Performance:")
    logger.output(f"  Average capacity: {summary['average_capacity']:.4f}")
    logger.output(f"  Maximum capacity: {summary['max_capacity']:.4f}")
    logger.output(f"  Minimum capacity: {summary['min_capacity']:.4f}")
    
    logger.output(f"\nBest Performance:")
    logger.output(f"  Best τ (delay): {summary['best_tau']}")
    logger.output(f"  Best ν (nonlinearity): {summary['best_nu']}")
    logger.output(f"  Best capacity: {summary['best_capacity']:.4f}")
    
    logger.output(f"\nFeature Selection:")
    logger.output(f"  Method: {results['feature_selection_method']}")
    logger.output(f"  Number of features used: {results['num_features']}")
    logger.output(f"  Selected features: {results['selected_features'][:5]}..." if len(results['selected_features']) > 5 else f"  Selected features: {results['selected_features']}")
    
    # Analyze trade-offs
    logger.output("\n" + "="*60)
    logger.output("MEMORY vs NONLINEARITY TRADE-OFF ANALYSIS")
    logger.output("="*60)
    
    tradeoff = evaluator.get_memory_vs_nonlinearity_tradeoff()
    
    logger.output("\nMemory Performance (averaged over ν):")
    for tau, perf in zip(tradeoff['tau_values'], tradeoff['memory_performance']):
        logger.output(f"  τ={tau:2d}: {perf:.4f}")
    
    logger.output("\nNonlinearity Performance (averaged over τ):")
    for nu, perf in zip(tradeoff['nu_values'], tradeoff['nonlinearity_performance']):
        logger.output(f"  ν={nu:5.1f}: {perf:.4f}")
    
    # Display capacity matrix
    logger.output("\n" + "="*60)
    logger.output("CAPACITY MATRIX C(τ, ν)")
    logger.output("="*60)
    logger.output("\nRows: τ (delay), Columns: ν (nonlinearity)")
    
    capacity_matrix = results['capacity_matrix']
    
    # Print header
    header = "   τ  |" + "".join([f"  ν={nu:4.1f}" for nu in nu_values])
    logger.output(header)
    logger.output("-" * len(header))
    
    # Print each row
    for i, tau in enumerate(tau_values):
        row = f"  {tau:2d}  |"
        for j in range(len(nu_values)):
            row += f"  {capacity_matrix[i, j]:6.4f}"
        logger.output(row)
    
    # Generate plots
    logger.info("\nGenerating plots...")
    evaluator.plot_results()
    
    logger.info("\n=== Benchmark completed successfully! ===")
    
    return evaluator, results


if __name__ == "__main__":
    evaluator, results = main()

