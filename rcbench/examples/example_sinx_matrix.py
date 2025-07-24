"""
Sin(x) Task Evaluation with Synthetic Data Example

This example demonstrates how to perform Sin(x) approximation evaluation using artificially
generated data without using any dataset classes.

This approach is useful when:
1. You have your own data format and don't want to convert to the dataset classes
2. You want to test the Sin(x) evaluation framework with synthetic data
3. You need to generate data with specific nonlinear transformation capabilities

The example creates:
- A time vector with regularly spaced samples
- An input signal with random values suitable for sin(x) approximation
- A matrix of synthetic node outputs with nonlinear responses
- Each node exhibits different capabilities for function approximation

Author: RCbench team
Date: 2024
"""

import logging
import numpy as np
import matplotlib.pyplot as plt

from rcbench.tasks.sinx import SinxEvaluator
from rcbench.visualization.plot_config import SinxPlotConfig
from rcbench.logger import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.INFO)

def generate_synthetic_sinx_data(n_samples=4000, n_nodes=18, input_range=(-2, 2), noise_level=0.03):
    """
    Generate synthetic reservoir computing data for testing Sin(x) approximation task.
    
    Args:
        n_samples (int): Number of time samples
        n_nodes (int): Number of reservoir nodes
        input_range (tuple): Range of input values (min, max)
        noise_level (float): Level of noise to add to node outputs
    
    Returns:
        tuple: (time_vector, input_signal, nodes_output_matrix, node_names)
    """
    # Create time vector
    dt = 0.01  # time step
    time_vector = np.arange(n_samples) * dt
    
    # Generate random input signal suitable for sin(x) task
    np.random.seed(42)  # for reproducibility
    input_min, input_max = input_range
    input_signal = np.random.uniform(input_min, input_max, n_samples)
    
    # Generate synthetic node outputs with nonlinear function approximation capabilities
    nodes_output = np.zeros((n_samples, n_nodes))
    
    for i in range(n_nodes):
        # Each node has different characteristics for function approximation
        # Parameters for nonlinear function approximation
        frequency_response = 0.5 + 2.0 * np.random.random()  # Frequency response 0.5-2.5
        phase_shift = 2 * np.pi * np.random.random()  # Random phase shift
        amplitude_scaling = 0.3 + 0.7 * np.random.random()  # Amplitude scaling
        nonlin_order = np.random.choice([1, 2, 3])  # Polynomial order for nonlinearity
        bias = -0.2 + 0.4 * np.random.random()  # Random bias
        memory_coeff = 0.05 + 0.15 * np.random.random()  # Small memory component
        
        # Create node response with function approximation characteristics
        node_signal = np.zeros(n_samples)
        
        for t in range(n_samples):
            current_input = input_signal[t]
            
            # Base response - different types of basis functions
            if i % 4 == 0:
                # Sinusoidal basis
                base_response = amplitude_scaling * np.sin(frequency_response * current_input + phase_shift)
            elif i % 4 == 1:
                # Cosine basis
                base_response = amplitude_scaling * np.cos(frequency_response * current_input + phase_shift)
            elif i % 4 == 2:
                # Polynomial basis
                base_response = amplitude_scaling * (current_input ** nonlin_order)
            else:
                # Exponential/sigmoid basis
                base_response = amplitude_scaling * np.tanh(frequency_response * current_input + bias)
            
            # Add small memory component
            if t > 0:
                memory_component = memory_coeff * node_signal[t-1]
                base_response += memory_component
            
            # Add bias and apply activation
            node_signal[t] = np.tanh(base_response + bias)
        
        # Add noise
        node_signal += noise_level * np.random.normal(0, 1, n_samples)
        nodes_output[:, i] = node_signal
    
    # Create node names
    node_names = [f'Node_{i+1}' for i in range(n_nodes)]
    
    return time_vector, input_signal, nodes_output, node_names

def main():
    """Main function to demonstrate Sin(x) evaluation with synthetic data."""
    
    logger.info("=== Sin(x) Task Evaluation with Synthetic Data ===")
    
    # Generate synthetic data
    logger.info("Generating synthetic reservoir data...")
    time_vector, input_signal, nodes_output, node_names = generate_synthetic_sinx_data(
        n_samples=4000,
        n_nodes=15,
        input_range=(-2, 2),
        noise_level=0.02
    )
    
    logger.info(f"Generated data:")
    logger.info(f"  - Time samples: {len(time_vector)}")
    logger.info(f"  - Input signal shape: {input_signal.shape}")
    logger.info(f"  - Input range: [{np.min(input_signal):.3f}, {np.max(input_signal):.3f}]")
    logger.info(f"  - Nodes output shape: {nodes_output.shape}")
    logger.info(f"  - Node names: {node_names[:5]}... (showing first 5)")
    
    # Create Sin(x) evaluator
    logger.info("Creating Sin(x) evaluator...")
    
    # Create plot configuration
    plot_config = SinxPlotConfig(
        show_plot=True,  # Set to False if running in headless environment
        plot_target_prediction=True
    )
    
    evaluator = SinxEvaluator(
        input_signal=input_signal,
        nodes_output=nodes_output,
        node_names=node_names,
        plot_config=plot_config
    )
    
    # Run Sin(x) evaluation
    logger.info("Running Sin(x) evaluation...")
    results = evaluator.run_evaluation(
        metric='NMSE',
        feature_selection_method='pca',
        num_features=10,  # Use top 10 components
        modeltype="Ridge",
        regression_alpha=0.1,
        train_ratio=0.8
    )
    
    # Display results
    logger.output(f"\n=== SIN(X) APPROXIMATION RESULTS ===")
    logger.output(f"SinX Accuracy (NMSE): {results['accuracy']:.4f}")
    logger.output(f"Metric used: {results['metric']}")
    logger.output(f"Selected features: {len(results['selected_features'])}")
    
    # Calculate some additional statistics
    y_true = results['y_test']
    y_pred = results['y_pred']
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    
    logger.output(f"\nAdditional Statistics:")
    logger.output(f"  MSE: {mse:.6f}")
    logger.output(f"  MAE: {mae:.6f}")
    logger.output(f"  Correlation: {correlation:.4f}")
    
    # Generate plots
    logger.info("Generating Sin(x) plots...")
    evaluator.plot_results(results)
    
    # Optional: Show input signal characteristics
    logger.info(f"\nInput signal characteristics:")
    logger.info(f"  Mean: {np.mean(input_signal):.4f}")
    logger.info(f"  Std: {np.std(input_signal):.4f}")
    logger.info(f"  Min: {np.min(input_signal):.4f}")
    logger.info(f"  Max: {np.max(input_signal):.4f}")
    
    # Optional: Show target characteristics
    target_signal = evaluator.target
    logger.info(f"\nTarget Sin(x) characteristics:")
    logger.info(f"  Mean: {np.mean(target_signal):.4f}")
    logger.info(f"  Std: {np.std(target_signal):.4f}")
    logger.info(f"  Min: {np.min(target_signal):.4f}")
    logger.info(f"  Max: {np.max(target_signal):.4f}")
    
    # Optional: Show node output characteristics
    logger.info(f"\nNode outputs characteristics:")
    logger.info(f"  Mean across all nodes: {np.mean(nodes_output):.4f}")
    logger.info(f"  Std across all nodes: {np.std(nodes_output):.4f}")
    logger.info(f"  Node with highest variance: {node_names[np.argmax(np.var(nodes_output, axis=0))]}")
    logger.info(f"  Node with lowest variance: {node_names[np.argmin(np.var(nodes_output, axis=0))]}")

if __name__ == "__main__":
    main() 