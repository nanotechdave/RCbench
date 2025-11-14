"""
NARMA Task Evaluation with Synthetic Data Example

This example demonstrates how to perform NARMA (Nonlinear Auto-Regressive Moving Average) 
evaluation using artificially generated data without using any dataset classes.

This approach is useful when:
1. You have your own data format and don't want to convert to the dataset classes
2. You want to test the NARMA evaluation framework with synthetic data
3. You need to generate data with specific nonlinear characteristics for benchmarking

The example creates:
- A time vector with regularly spaced samples
- An input signal with uniform random values suitable for NARMA
- A matrix of synthetic node outputs with nonlinear dynamics
- Each node exhibits different nonlinear responses and memory properties

Author: Davide Pilati
Date: 2025
"""

import logging
import numpy as np
import matplotlib.pyplot as plt

from rcbench.tasks.narma import NarmaEvaluator
from rcbench.visualization.plot_config import NarmaPlotConfig
from rcbench.logger import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.INFO)

def generate_synthetic_narma_data(n_samples=5000, n_nodes=25, input_range=(0, 0.5), noise_level=0.02):
    """
    Generate synthetic reservoir computing data for testing NARMA task.
    
    Args:
        n_samples (int): Number of time samples
        n_nodes (int): Number of reservoir nodes
        input_range (tuple): Range of input values (min, max) - NARMA typically uses [0, 0.5]
        noise_level (float): Level of noise to add to node outputs
    
    Returns:
        tuple: (time_vector, input_signal, nodes_output_matrix, node_names)
    """
    # Create time vector
    dt = 0.01  # time step
    time_vector = np.arange(n_samples) * dt
    
    # Generate random input signal suitable for NARMA task
    # NARMA typically uses uniform random input in range [0, 0.5]
    np.random.seed(42)  # for reproducibility
    input_min, input_max = input_range
    input_signal = np.random.uniform(input_min, input_max, n_samples)
    
    # Generate synthetic node outputs with nonlinear dynamics
    nodes_output = np.zeros((n_samples, n_nodes))
    
    for i in range(n_nodes):
        # Each node has different nonlinear characteristics
        # Parameters for nonlinear dynamics
        memory_length = np.random.randint(2, 8)  # Memory length 2-7 steps
        nonlin_strength = 0.1 + 0.3 * np.random.random()  # Nonlinearity strength
        leak_rate = 0.1 + 0.4 * np.random.random()  # Leak rate between 0.1-0.5
        input_scaling = 0.5 + 1.0 * np.random.random()  # Input scaling
        bias = -0.1 + 0.2 * np.random.random()  # Random bias
        
        # Create node response with nonlinear dynamics
        node_signal = np.zeros(n_samples)
        node_history = np.zeros(memory_length)  # Keep history for nonlinear terms
        
        for t in range(n_samples):
            # Current input contribution
            input_contrib = input_scaling * input_signal[t] + bias
            
            # Nonlinear feedback from history
            nonlin_feedback = 0
            if t > 0:
                # Add quadratic nonlinearity based on recent history
                recent_avg = np.mean(node_history[-min(3, memory_length):])
                nonlin_feedback = nonlin_strength * recent_avg * recent_avg
                
                # Add cross-coupling with input
                if t >= memory_length:
                    past_input = input_signal[t - memory_length]
                    nonlin_feedback += 0.1 * past_input * recent_avg
            
            # Leaky integration
            if t > 0:
                node_signal[t] = (1 - leak_rate) * node_signal[t-1] + leak_rate * (input_contrib + nonlin_feedback)
            else:
                node_signal[t] = input_contrib
            
            # Apply nonlinear activation (tanh)
            node_signal[t] = np.tanh(node_signal[t])
            
            # Update history
            node_history = np.roll(node_history, -1)
            node_history[-1] = node_signal[t]
        
        # Add noise
        node_signal += noise_level * np.random.normal(0, 1, n_samples)
        nodes_output[:, i] = node_signal
    
    # Create node names
    node_names = [f'Node_{i+1}' for i in range(n_nodes)]
    
    return time_vector, input_signal, nodes_output, node_names

def main():
    """Main function to demonstrate NARMA evaluation with synthetic data."""
    
    logger.info("=== NARMA Task Evaluation with Synthetic Data ===")
    
    # Generate synthetic data
    logger.info("Generating synthetic reservoir data...")
    time_vector, input_signal, nodes_output, node_names = generate_synthetic_narma_data(
        n_samples=5000,
        n_nodes=20,
        input_range=(0, 0.5),  # Standard range for NARMA
        noise_level=0.01
    )
    
    logger.info(f"Generated data:")
    logger.info(f"  - Time samples: {len(time_vector)}")
    logger.info(f"  - Input signal shape: {input_signal.shape}")
    logger.info(f"  - Input range: [{np.min(input_signal):.3f}, {np.max(input_signal):.3f}]")
    logger.info(f"  - Nodes output shape: {nodes_output.shape}")
    logger.info(f"  - Node names: {node_names[:5]}... (showing first 5)")
    
    # Create NARMA evaluator
    logger.info("Creating NARMA evaluator...")
    
    # Create plot configuration
    plot_config = NarmaPlotConfig(
        show_plot=True,  # Set to False if running in headless environment
        plot_target_prediction=True
    )
    
    evaluator = NarmaEvaluator(
        input_signal=input_signal,
        nodes_output=nodes_output,
        node_names=node_names,
        order=2,  # NARMA-2 task
        alpha=0.4,  # NARMA parameters
        beta=0.4,
        gamma=0.6,
        delta=0.1,
        plot_config=plot_config
    )
    
    # Run NARMA evaluation
    logger.info("Running NARMA evaluation...")
    results = evaluator.run_evaluation(
        metric='NMSE',
        feature_selection_method='pca',
        num_features=10,  # Use top 10 components
        modeltype="Ridge",
        regression_alpha=0.1,
        train_ratio=0.8
    )
    
    # Display results
    logger.output(f"\n=== NARMA-2 TASK RESULTS ===")
    logger.output(f"NARMA Accuracy (NMSE): {results['accuracy']:.4f}")
    logger.output(f"Metric used: {results['metric']}")
    logger.output(f"Selected features: {len(results['selected_features'])}")
    
    # Generate plots
    logger.info("Generating NARMA plots...")
    evaluator.plot_results(results)
    
    # Optional: Show input signal characteristics
    logger.info(f"\nInput signal characteristics:")
    logger.info(f"  Mean: {np.mean(input_signal):.4f}")
    logger.info(f"  Std: {np.std(input_signal):.4f}")
    logger.info(f"  Min: {np.min(input_signal):.4f}")
    logger.info(f"  Max: {np.max(input_signal):.4f}")
    
    # Optional: Show node output characteristics
    logger.info(f"\nNode outputs characteristics:")
    logger.info(f"  Mean across all nodes: {np.mean(nodes_output):.4f}")
    logger.info(f"  Std across all nodes: {np.std(nodes_output):.4f}")
    logger.info(f"  Node with highest variance: {node_names[np.argmax(np.var(nodes_output, axis=0))]}")
    logger.info(f"  Node with lowest variance: {node_names[np.argmin(np.var(nodes_output, axis=0))]}")

if __name__ == "__main__":
    main() 