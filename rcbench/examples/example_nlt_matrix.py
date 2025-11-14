"""
Non-Linear Transformation (NLT) Task Evaluation with Synthetic Data Example

This example demonstrates how to perform NLT evaluation using artificially
generated data without using any dataset classes.

This approach is useful when:
1. You have your own data format and don't want to convert to the dataset classes
2. You want to test the NLT evaluation framework with synthetic data
3. You need to generate data with specific waveform transformation capabilities

The example creates:
- A time vector with regularly spaced samples
- An input signal with specific waveform patterns (sine, triangular)
- A matrix of synthetic node outputs with nonlinear transformation capabilities
- Each node exhibits different responses to waveform patterns

Author: Davide Pilati
Date: 2025
"""

import logging
import numpy as np
import matplotlib.pyplot as plt

from rcbench.tasks.nlt import NltEvaluator
from rcbench.visualization.plot_config import NLTPlotConfig
from rcbench.logger import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.INFO)

def generate_synthetic_nlt_data(n_samples=6000, n_nodes=22, input_amplitude=1.0, 
                               base_frequency=0.5, noise_level=0.02):
    """
    Generate synthetic reservoir computing data for testing NLT task.
    
    Args:
        n_samples (int): Number of time samples
        n_nodes (int): Number of reservoir nodes
        input_amplitude (float): Amplitude of input waveforms
        base_frequency (float): Base frequency for waveform generation
        noise_level (float): Level of noise to add to node outputs
    
    Returns:
        tuple: (time_vector, input_signal, nodes_output_matrix, node_names)
    """
    # Create time vector
    dt = 0.01  # time step
    time_vector = np.arange(n_samples) * dt
    
    # Generate complex input signal with multiple frequency components
    # This creates a rich waveform suitable for NLT analysis
    np.random.seed(42)  # for reproducibility
    
    # Base sinusoidal component
    input_signal = input_amplitude * np.sin(2 * np.pi * base_frequency * time_vector)
  
    
    
    # Add small amount of noise to make it more realistic
    input_signal += 0.05 * input_amplitude * np.random.normal(0, 1, n_samples)
    
    # Generate synthetic node outputs with nonlinear transformation capabilities
    nodes_output = np.zeros((n_samples, n_nodes))
    
    for i in range(n_nodes):
        # Each node has different nonlinear transformation characteristics
        # Parameters for different types of nonlinear responses
        sensitivity = 0.5 + 1.5 * np.random.random()  # Input sensitivity
        nonlin_type = i % 5  # Different types of nonlinearity
        time_constant = 0.01 + 0.05 * np.random.random()  # Time constant for dynamics
        bias = -0.3 + 0.6 * np.random.random()  # Random bias
        freq_selectivity = 0.5 + 2.0 * np.random.random()  # Frequency selectivity
        
        # Create node response with different nonlinear transformations
        node_signal = np.zeros(n_samples)
        internal_state = 0.0
        
        for t in range(n_samples):
            current_input = input_signal[t]
            
            # Different types of nonlinear transformations
            if nonlin_type == 0:
                # Polynomial nonlinearity
                transformed_input = sensitivity * (current_input + 0.3 * current_input**2 + 0.1 * current_input**3)
            elif nonlin_type == 1:
                # Exponential-like nonlinearity
                transformed_input = sensitivity * np.tanh(2 * current_input) * current_input
            elif nonlin_type == 2:
                # Frequency-dependent response
                phase = 2 * np.pi * freq_selectivity * time_vector[t]
                transformed_input = sensitivity * current_input * np.cos(phase)
            elif nonlin_type == 3:
                # Threshold-like nonlinearity
                threshold = 0.5 * input_amplitude
                transformed_input = sensitivity * current_input * (1 if np.abs(current_input) > threshold else 0.3)
            else:
                # Saturating nonlinearity
                transformed_input = sensitivity * np.sign(current_input) * np.sqrt(np.abs(current_input))
            
            # Apply first-order dynamics
            internal_state += (transformed_input - internal_state) / (time_constant * 1000 + 1)
            
            # Apply final activation and bias
            node_signal[t] = np.tanh(internal_state + bias)
        
        # Add noise
        node_signal += noise_level * np.random.normal(0, 1, n_samples)
        nodes_output[:, i] = node_signal
    
    # Create node names
    node_names = [f'Node_{i+1}' for i in range(n_nodes)]
    
    return time_vector, input_signal, nodes_output, node_names

def main():
    """Main function to demonstrate NLT evaluation with synthetic data."""
    
    logger.info("=== Non-Linear Transformation (NLT) Task Evaluation with Synthetic Data ===")
    
    # Generate synthetic data
    logger.info("Generating synthetic reservoir data...")
    time_vector, input_signal, nodes_output, node_names = generate_synthetic_nlt_data(
        n_samples=6000,
        n_nodes=20,
        input_amplitude=0.8,
        base_frequency=0.3,
        noise_level=0.015
    )
    
    logger.info(f"Generated data:")
    logger.info(f"  - Time samples: {len(time_vector)}")
    logger.info(f"  - Input signal shape: {input_signal.shape}")
    logger.info(f"  - Input range: [{np.min(input_signal):.3f}, {np.max(input_signal):.3f}]")
    logger.info(f"  - Nodes output shape: {nodes_output.shape}")
    logger.info(f"  - Node names: {node_names[:5]}... (showing first 5)")
    
    # Create NLT evaluator
    logger.info("Creating NLT evaluator...")
    
    # Create plot configuration
    plot_config = NLTPlotConfig(
        show_plot=True,  # Set to False if running in headless environment
        plot_target_prediction=True
    )
    
    evaluator = NltEvaluator(
        input_signal=input_signal,
        nodes_output=nodes_output,
        time_array=time_vector,
        waveform_type='sine',  # Can be 'sine' or 'triangular'
        node_names=node_names,
        plot_config=plot_config
    )
    
    # Get available targets
    available_targets = list(evaluator.targets.keys())
    logger.info(f"Available transformation targets: {available_targets}")
    
    # Run NLT evaluation for each available target
    all_results = {}
    for target_name in available_targets:
        logger.info(f"Running NLT evaluation for target: {target_name}")
        
        try:
            results = evaluator.run_evaluation(
                target_name=target_name,
                metric='NMSE',
                feature_selection_method='pca',
                num_features=12,  # Use top 12 components
                modeltype="Ridge",
                regression_alpha=0.1,
                train_ratio=0.8
            )
            
            all_results[target_name] = results
            
            # Display results for this target
            logger.output(f"\n=== NLT RESULTS FOR {target_name.upper()} ===")
            logger.output(f"Accuracy (NMSE): {results['accuracy']:.4f}")
            logger.output(f"Metric used: {results['metric']}")
            logger.output(f"Selected features: {len(results['selected_features'])}")
            
        except Exception as e:
            logger.error(f"Error evaluating {target_name}: {str(e)}")
    
    # Generate plots
    logger.info("Generating NLT plots...")
    evaluator.plot_results(all_results)
    
    # Optional: Show input signal characteristics
    logger.info(f"\nInput signal characteristics:")
    logger.info(f"  Mean: {np.mean(input_signal):.4f}")
    logger.info(f"  Std: {np.std(input_signal):.4f}")
    logger.info(f"  Min: {np.min(input_signal):.4f}")
    logger.info(f"  Max: {np.max(input_signal):.4f}")
    
    # Show frequency content of input
    from scipy.fft import fft, fftfreq
    fft_signal = fft(input_signal)
    frequencies = fftfreq(len(input_signal), d=0.01)
    power_spectrum = np.abs(fft_signal)**2
    
    # Find dominant frequencies
    positive_freq_mask = frequencies > 0
    dominant_freqs = frequencies[positive_freq_mask][np.argsort(power_spectrum[positive_freq_mask])[-3:]]
    logger.info(f"  Dominant frequencies: {dominant_freqs[::-1]} Hz")
    
    # Optional: Show node output characteristics
    logger.info(f"\nNode outputs characteristics:")
    logger.info(f"  Mean across all nodes: {np.mean(nodes_output):.4f}")
    logger.info(f"  Std across all nodes: {np.std(nodes_output):.4f}")
    logger.info(f"  Node with highest variance: {node_names[np.argmax(np.var(nodes_output, axis=0))]}")
    logger.info(f"  Node with lowest variance: {node_names[np.argmin(np.var(nodes_output, axis=0))]}")

if __name__ == "__main__":
    main() 