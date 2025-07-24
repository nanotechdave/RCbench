"""
Memory Capacity Evaluation with Synthetic Data Example

This example demonstrates how to perform Memory Capacity evaluation using artificially
generated data without using any dataset classes (ReservoirDataset or ElecResDataset).

This approach is useful when:
1. You have your own data format and don't want to convert to the dataset classes
2. You want to test the evaluation framework with synthetic data

The example creates:
- A time vector with regularly spaced samples
- An input signal with structured random noise
- A matrix of synthetic node outputs with built-in memory characteristics
- Each node has different delay, decay, and gain properties to simulate reservoir dynamics

Author: Davide Pilati
Date: 2025-07-24
"""

import logging
import numpy as np
import matplotlib.pyplot as plt

from rcbench.tasks.memorycapacity import MemoryCapacityEvaluator
from rcbench.visualization.plot_config import MCPlotConfig
from rcbench.logger import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.INFO)

def generate_synthetic_reservoir_data(n_samples=3000, n_nodes=20, input_amplitude=1.0, noise_level=0.1):
    """
    Generate synthetic reservoir computing data for testing Memory Capacity.
    
    Args:
        n_samples (int): Number of time samples
        n_nodes (int): Number of reservoir nodes
        input_amplitude (float): Amplitude of input signal
        noise_level (float): Level of noise to add to node outputs
    
    Returns:
        tuple: (time_vector, input_signal, nodes_output_matrix, node_names)
    """
    # Create time vector
    dt = 0.01  # time step
    time_vector = np.arange(n_samples) * dt
    
    # Generate random input signal (white noise with some structure)
    np.random.seed(42)  # for reproducibility
    input_signal = input_amplitude * np.random.uniform(-1, 1, n_samples)
    
    # Generate synthetic node outputs that have memory properties
    nodes_output = np.zeros((n_samples, n_nodes))
    
    for i in range(n_nodes):
        # Each node has different memory characteristics
        # Some nodes respond to current input, others to delayed versions
        delay = np.random.randint(0, 10)  # Random delay 0-9 steps
        decay = 0.7 + 0.25 * np.random.random()  # Decay factor between 0.7-0.95
        gain = 0.5 + 1.0 * np.random.random()  # Gain factor between 0.5-1.5
        
        # Create node response with memory
        node_signal = np.zeros(n_samples)
        for t in range(delay, n_samples):
            # Current input + decayed previous state + some nonlinearity
            current_input = gain * input_signal[t - delay]
            if t > 0:
                node_signal[t] = current_input + decay * node_signal[t-1]
            else:
                node_signal[t] = current_input
            
            # Add some nonlinearity (tanh activation)
            node_signal[t] = np.tanh(node_signal[t])
        
        # Add noise
        node_signal += noise_level * np.random.normal(0, 1, n_samples)
        nodes_output[:, i] = node_signal
    
    # Create node names
    node_names = [f'Node_{i+1}' for i in range(n_nodes)]
    
    return time_vector, input_signal, nodes_output, node_names

def main():
    """Main function to demonstrate MC evaluation with synthetic data."""
    
    logger.info("=== Memory Capacity Evaluation with Synthetic Data ===")
    
    # Generate synthetic data
    logger.info("Generating synthetic reservoir data...")
    time_vector, input_signal, nodes_output, node_names = generate_synthetic_reservoir_data(
        n_samples=3000,
        n_nodes=15,
        input_amplitude=0.8,
        noise_level=0.05
    )
    
    logger.info(f"Generated data:")
    logger.info(f"  - Time samples: {len(time_vector)}")
    logger.info(f"  - Input signal shape: {input_signal.shape}")
    logger.info(f"  - Nodes output shape: {nodes_output.shape}")
    logger.info(f"  - Node names: {node_names[:5]}... (showing first 5)")
    
    
    # Create plot configuration
    plot_config = MCPlotConfig(
        plot_mc_curve=True,
        plot_predictions=True,
        plot_total_mc=True,
        show_plot=True  # Set to False if running in headless environment
    )
    
    evaluator = MemoryCapacityEvaluator(
        input_signal=input_signal,
        nodes_output=nodes_output,
        max_delay=25,  # Test memory up to 25 time steps
        random_state=42,
        node_names=node_names,
        plot_config=plot_config
    )
    
    # Run full Memory Capacity evaluation
    logger.info("Running Memory Capacity evaluation...")
    results = evaluator.calculate_total_memory_capacity(
        feature_selection_method='pca',
        num_features=15,  # Use top 10 components
        modeltype="Ridge",
        regression_alpha=0.1,
        train_ratio=0.8
    )
    evaluator.plot_results()

    # Display results
    logger.output(f"\n=== MEMORY CAPACITY RESULTS ===")
    logger.output(f"Total Memory Capacity: {results['total_memory_capacity']:.4f}")
    logger.output(f"Memory capacity per delay:")
    
    for delay, mc_value in results['delay_results'].items():
        logger.output(f"  Delay {delay:2d}: MC = {mc_value:.4f}")
    
    # Show some statistics
    mc_values = list(results['delay_results'].values())
    logger.output(f"\nStatistics:")
    logger.output(f"  Max MC: {max(mc_values):.4f} at delay {max(results['delay_results'], key=results['delay_results'].get)}")
    logger.output(f"  Mean MC: {np.mean(mc_values):.4f}")
    logger.output(f"  Std MC: {np.std(mc_values):.4f}")
    

if __name__ == "__main__":
    main() 