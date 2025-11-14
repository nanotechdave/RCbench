"""
Kernel Rank Task Evaluation with Synthetic Data Example

This example demonstrates how to perform Kernel Rank evaluation using artificially
generated data without using any dataset classes.

This approach is useful when:
1. You have your own data format and don't want to convert to the dataset classes
2. You want to test the Kernel Rank evaluation framework with synthetic data
3. You need to generate data with specific dimensionality characteristics for benchmarking

The example creates:
- A time vector with regularly spaced samples
- An input signal with rich dynamics suitable for kernel analysis
- A matrix of synthetic node outputs with diverse feature representations
- Each node contributes different aspects to the high-dimensional feature space

Author: Davide Pilati
Date: 2025
"""

import logging
import numpy as np
import matplotlib.pyplot as plt

from rcbench.tasks.kernelrank import KernelRankEvaluator
from rcbench.tasks.generalizationrank import GeneralizationRankEvaluator
from rcbench.logger import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.INFO)

def generate_synthetic_kernel_data(n_samples=4000, n_nodes=30, input_complexity=3, noise_level=0.02):
    """
    Generate synthetic reservoir computing data for testing Kernel Rank task.
    
    Args:
        n_samples (int): Number of time samples
        n_nodes (int): Number of reservoir nodes
        input_complexity (int): Complexity level of input signal (1-5)
        noise_level (float): Level of noise to add to node outputs
    
    Returns:
        tuple: (time_vector, input_signal, nodes_output_matrix, node_names)
    """
    # Create time vector
    dt = 0.01  # time step
    time_vector = np.arange(n_samples) * dt
    
    # Generate complex input signal with multiple components
    # Higher complexity creates richer dynamics for kernel analysis
    np.random.seed(42)  # for reproducibility
    
    # Base signal components
    frequencies = [0.5, 1.2, 2.1, 3.8, 5.5][:input_complexity]
    amplitudes = [1.0, 0.7, 0.5, 0.3, 0.2][:input_complexity]
    phases = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi][:input_complexity]
    
    input_signal = np.zeros(n_samples)
    for freq, amp, phase in zip(frequencies, amplitudes, phases):
        input_signal += amp * np.sin(2 * np.pi * freq * time_vector + phase)
    
    # Add nonlinear components
    input_signal += 0.3 * np.sin(2 * np.pi * 0.8 * time_vector) * np.cos(2 * np.pi * 2.5 * time_vector)
    
    # Add chaotic component
    lorenz_like = np.sin(2 * np.pi * 1.3 * time_vector) * np.exp(-0.1 * np.sin(2 * np.pi * 0.2 * time_vector))
    input_signal += 0.4 * lorenz_like
    
    # Normalize
    input_signal = input_signal / np.std(input_signal)
    
    # Generate synthetic node outputs with diverse feature representations
    nodes_output = np.zeros((n_samples, n_nodes))
    
    for i in range(n_nodes):
        # Each node represents different aspects of the feature space
        # Parameters for diverse feature generation
        feature_type = i % 6  # Different types of features
        time_scale = 0.01 + 0.1 * np.random.random()  # Different time scales
        nonlin_strength = 0.2 + 0.8 * np.random.random()  # Nonlinearity strength
        coupling_strength = 0.3 + 0.7 * np.random.random()  # Input coupling
        frequency_pref = 0.5 + 4.0 * np.random.random()  # Frequency preference
        
        # Create node response representing different feature dimensions
        node_signal = np.zeros(n_samples)
        internal_state = np.random.normal(0, 0.1)  # Random initial state
        
        for t in range(n_samples):
            current_input = input_signal[t]
            
            # Different feature extraction mechanisms
            if feature_type == 0:
                # Linear feature with memory
                feature = coupling_strength * current_input
                if t > 0:
                    feature += 0.7 * node_signal[t-1]
                    
            elif feature_type == 1:
                # Quadratic feature
                feature = coupling_strength * (current_input**2 - 0.5)
                
            elif feature_type == 2:
                # Frequency-selective feature
                phase = 2 * np.pi * frequency_pref * time_vector[t]
                feature = coupling_strength * current_input * np.cos(phase)
                
            elif feature_type == 3:
                # Nonlinear transformation
                feature = coupling_strength * np.tanh(nonlin_strength * current_input)
                
            elif feature_type == 4:
                # Cross-product feature (interaction)
                if t > 5:
                    past_input = input_signal[t-5]
                    feature = coupling_strength * current_input * past_input
                else:
                    feature = coupling_strength * current_input
                    
            else:
                # Edge detection feature
                if t > 0:
                    derivative = current_input - input_signal[t-1]
                    feature = coupling_strength * np.abs(derivative)
                else:
                    feature = 0
            
            # Apply dynamics
            internal_state += (feature - internal_state) / (time_scale * 1000 + 1)
            
            # Apply activation
            node_signal[t] = np.tanh(internal_state)
        
        # Add noise
        node_signal += noise_level * np.random.normal(0, 1, n_samples)
        nodes_output[:, i] = node_signal
    
    # Create node names
    node_names = [f'Node_{i+1}' for i in range(n_nodes)]
    
    return time_vector, input_signal, nodes_output, node_names

def main():
    """Main function to demonstrate Kernel Rank evaluation with synthetic data."""
    
    logger.info("=== Kernel Rank Task Evaluation with Synthetic Data ===")
    
    # Generate synthetic data
    logger.info("Generating synthetic reservoir data...")
    time_vector, input_signal, nodes_output, node_names = generate_synthetic_kernel_data(
        n_samples=4000,
        n_nodes=25,
        input_complexity=4,
        noise_level=0.015
    )
    
    logger.info(f"Generated data:")
    logger.info(f"  - Time samples: {len(time_vector)}")
    logger.info(f"  - Input signal shape: {input_signal.shape}")
    logger.info(f"  - Input range: [{np.min(input_signal):.3f}, {np.max(input_signal):.3f}]")
    logger.info(f"  - Nodes output shape: {nodes_output.shape}")
    logger.info(f"  - Node names: {node_names[:5]}... (showing first 5)")
    
    # Create Kernel Rank evaluator
    logger.info("Creating Kernel Rank evaluator...")
    
    kernel_evaluator = KernelRankEvaluator(
        input_signal=input_signal,
        nodes_output=nodes_output,
        node_names=node_names
    )
    
    # Run Kernel Rank evaluation
    logger.info("Running Kernel Rank evaluation...")
    kernel_results = kernel_evaluator.run_evaluation(
        feature_selection_method='pca',
        num_features=20,  # Use top 20 components
        kernel_type='rbf',
        gamma='scale',
        train_ratio=0.8
    )
    
    # Display Kernel Rank results
    logger.output(f"\n=== KERNEL RANK RESULTS ===")
    logger.output(f"Kernel Rank: {kernel_results['kernel_rank']}")
    logger.output(f"Kernel type: {kernel_results['kernel_type']}")
    logger.output(f"Selected features: {len(kernel_results['selected_features'])}")
    logger.output(f"Effective dimensionality: {kernel_results.get('effective_dimensionality', 'N/A')}")
    
    # Create Generalization Rank evaluator
    logger.info("Creating Generalization Rank evaluator...")
    
    gen_evaluator = GeneralizationRankEvaluator(
        input_signal=input_signal,
        nodes_output=nodes_output,
        node_names=node_names
    )
    
    # Run Generalization Rank evaluation
    logger.info("Running Generalization Rank evaluation...")
    gen_results = gen_evaluator.run_evaluation(
        feature_selection_method='pca',
        num_features=20,  # Use top 20 components
        kernel_type='rbf',
        gamma='scale',
        train_ratio=0.8
    )
    
    # Display Generalization Rank results
    logger.output(f"\n=== GENERALIZATION RANK RESULTS ===")
    logger.output(f"Generalization Rank: {gen_results['generalization_rank']}")
    logger.output(f"Kernel type: {gen_results['kernel_type']}")
    logger.output(f"Selected features: {len(gen_results['selected_features'])}")
    logger.output(f"Generalization capability: {gen_results.get('generalization_score', 'N/A')}")
    
    # Optional: Show input signal characteristics
    logger.info(f"\nInput signal characteristics:")
    logger.info(f"  Mean: {np.mean(input_signal):.4f}")
    logger.info(f"  Std: {np.std(input_signal):.4f}")
    logger.info(f"  Min: {np.min(input_signal):.4f}")
    logger.info(f"  Max: {np.max(input_signal):.4f}")
    
    # Show signal complexity metrics
    # Calculate approximate entropy as a measure of complexity
    def approximate_entropy(signal, m=2, r=None):
        """Calculate approximate entropy of a signal."""
        if r is None:
            r = 0.2 * np.std(signal)
        
        N = len(signal)
        patterns_m = []
        patterns_m1 = []
        
        # Generate patterns of length m and m+1
        for i in range(N - m + 1):
            patterns_m.append(signal[i:i+m])
        for i in range(N - m):
            patterns_m1.append(signal[i:i+m+1])
        
        # Count matches
        def count_matches(patterns, r):
            counts = []
            for i, pattern in enumerate(patterns):
                matches = 0
                for j, other_pattern in enumerate(patterns):
                    if np.max(np.abs(np.array(pattern) - np.array(other_pattern))) <= r:
                        matches += 1
                counts.append(matches)
            return counts
        
        counts_m = count_matches(patterns_m, r)
        counts_m1 = count_matches(patterns_m1, r)
        
        phi_m = np.mean([np.log(count/len(patterns_m)) for count in counts_m])
        phi_m1 = np.mean([np.log(count/len(patterns_m1)) for count in counts_m1])
        
        return phi_m - phi_m1
    
    try:
        complexity = approximate_entropy(input_signal)
        logger.info(f"  Approximate entropy (complexity): {complexity:.4f}")
    except:
        logger.info(f"  Approximate entropy: Could not calculate")
    
    # Optional: Show node output characteristics
    logger.info(f"\nNode outputs characteristics:")
    logger.info(f"  Mean across all nodes: {np.mean(nodes_output):.4f}")
    logger.info(f"  Std across all nodes: {np.std(nodes_output):.4f}")
    
    # Calculate effective rank of node outputs
    U, s, Vt = np.linalg.svd(nodes_output, full_matrices=False)
    effective_rank = np.sum(s > 0.01 * s[0])  # Count significant singular values
    logger.info(f"  Effective rank of node outputs: {effective_rank}")
    logger.info(f"  Condition number: {s[0]/s[-1]:.2f}")
    
    logger.info(f"  Node with highest variance: {node_names[np.argmax(np.var(nodes_output, axis=0))]}")
    logger.info(f"  Node with lowest variance: {node_names[np.argmin(np.var(nodes_output, axis=0))]}")

if __name__ == "__main__":
    main() 