"""
ReservoirPy Benchmark Example

This example demonstrates how to benchmark a simulated Echo State Network (ESN)
using the reservoirPy package with RCbench. It performs Memory Capacity (MC) 
and Sin(x) tasks on a uniformly random input signal.

Requirements:
    pip install reservoirpy

Author: Davide Pilati
Date: 2025
"""

import logging
import numpy as np

# ReservoirPy imports
try:
    from reservoirpy.nodes import Reservoir, Ridge as ReservoirPyRidge
except ImportError:
    raise ImportError(
        "reservoirpy is required for this example. "
        "Install it with: pip install reservoirpy"
    )

# RCbench imports
from rcbench import MemoryCapacityEvaluator, SinxEvaluator
from rcbench import MCPlotConfig, SinxPlotConfig
from rcbench.logger import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.INFO)


def create_reservoir(
    units: int = 100,
    lr: float = 0.3,
    sr: float = 0.9,
    input_scaling: float = 0.1,
    seed: int = 42
) -> Reservoir:
    """
    Create a ReservoirPy Echo State Network reservoir.
    
    Parameters:
    -----------
    units : int
        Number of reservoir neurons (nodes)
    lr : float
        Leak rate (between 0 and 1)
    sr : float
        Spectral radius of the reservoir weight matrix
    input_scaling : float
        Scaling factor for input weights
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    Reservoir
        A ReservoirPy Reservoir node
    """
    reservoir = Reservoir(
        units=units,
        lr=lr,
        sr=sr,
        input_scaling=input_scaling,
        seed=seed
    )
    return reservoir


def generate_random_input(
    n_samples: int = 3000,
    seed: int = 42
) -> np.ndarray:
    """
    Generate a uniformly distributed random input signal.
    
    Parameters:
    -----------
    n_samples : int
        Number of time steps
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    np.ndarray
        Random input signal with shape (n_samples, 1)
    """
    np.random.seed(seed)
    # Uniform distribution between -1 and 1
    input_signal = np.random.uniform(-1, 1, size=(n_samples, 1))
    return input_signal


def run_reservoir(reservoir: Reservoir, input_signal: np.ndarray) -> np.ndarray:
    """
    Run the reservoir with the given input signal.
    
    Parameters:
    -----------
    reservoir : Reservoir
        ReservoirPy Reservoir node
    input_signal : np.ndarray
        Input signal with shape (n_samples, 1)
        
    Returns:
    --------
    np.ndarray
        Reservoir states with shape (n_samples, n_units)
    """
    # Run reservoir and collect states
    states = reservoir.run(input_signal)
    
    return states


def benchmark_memory_capacity(
    input_signal: np.ndarray,
    reservoir_states: np.ndarray,
    node_names: list,
    max_delay: int = 30,
    train_ratio: float = 0.8,
    show_plots: bool = True
) -> dict:
    """
    Benchmark the reservoir's Memory Capacity using RCbench.
    
    Parameters:
    -----------
    input_signal : np.ndarray
        Input signal (1D array)
    reservoir_states : np.ndarray
        Reservoir node outputs with shape (n_samples, n_nodes)
    node_names : list
        Names of reservoir nodes
    max_delay : int
        Maximum delay to test for memory capacity
    train_ratio : float
        Ratio of data used for training
    show_plots : bool
        Whether to display plots
        
    Returns:
    --------
    dict
        Memory capacity results
    """
    logger.info("=" * 60)
    logger.info("MEMORY CAPACITY BENCHMARK")
    logger.info("=" * 60)
    
    # Create MC plot configuration
    plot_config = MCPlotConfig(
        save_dir=None,
        figsize=(10, 6),
        plot_mc_curve=True,
        plot_predictions=True,
        plot_total_mc=True,
        max_delays_to_plot=5,
        plot_input_signal=True,
        plot_output_responses=True,
        plot_nonlinearity=True,
        plot_frequency_analysis=False,  # No frequency analysis for random signal
        nonlinearity_plot_style='scatter',
        show_plot=show_plots,
        train_ratio=train_ratio
    )
    
    # Create Memory Capacity evaluator
    evaluator = MemoryCapacityEvaluator(
        input_signal=input_signal,
        nodes_output=reservoir_states,
        max_delay=max_delay,
        node_names=node_names,
        plot_config=plot_config
    )
    
    # Calculate total memory capacity
    results = evaluator.calculate_total_memory_capacity(
        feature_selection_method='pca',
        num_features='all',
        modeltype='Ridge',
        regression_alpha=0.1,
        train_ratio=train_ratio
    )
    
    # Generate plots
    if show_plots:
        evaluator.plot_results()
    
    # Log results
    logger.output(f"\nTotal Memory Capacity: {results['total_memory_capacity']:.4f}")
    logger.output(f"Theoretical Maximum (N nodes): {reservoir_states.shape[1]}")
    logger.output(f"MC/N ratio: {results['total_memory_capacity']/reservoir_states.shape[1]:.2%}")
    
    # Show per-delay results for first few delays
    logger.output("\nPer-delay Memory Capacity:")
    for delay, mc_value in list(results['delay_results'].items())[:10]:
        logger.output(f"  Delay {delay:2d}: MC = {mc_value:.4f}")
    
    return results


def benchmark_sinx(
    input_signal: np.ndarray,
    reservoir_states: np.ndarray,
    node_names: list,
    train_ratio: float = 0.8,
    show_plots: bool = True
) -> dict:
    """
    Benchmark the reservoir's ability to compute sin(x) using RCbench.
    
    Parameters:
    -----------
    input_signal : np.ndarray
        Input signal (1D array)
    reservoir_states : np.ndarray
        Reservoir node outputs with shape (n_samples, n_nodes)
    node_names : list
        Names of reservoir nodes
    train_ratio : float
        Ratio of data used for training
    show_plots : bool
        Whether to display plots
        
    Returns:
    --------
    dict
        Sin(x) task results
    """
    logger.info("\n" + "=" * 60)
    logger.info("SIN(X) APPROXIMATION BENCHMARK")
    logger.info("=" * 60)
    
    # Create Sinx plot configuration
    plot_config = SinxPlotConfig(
        save_dir=None,
        figsize=(10, 6),
        plot_input_signal=True,
        plot_output_responses=True,
        plot_nonlinearity=True,
        plot_frequency_analysis=False,
        plot_target_prediction=True,
        nonlinearity_plot_style='scatter',
        show_plot=show_plots,
        train_ratio=train_ratio,
        prediction_sample_count=500  # Show more samples in prediction plot
    )
    
    # Create Sin(x) evaluator
    evaluator = SinxEvaluator(
        input_signal=input_signal,
        nodes_output=reservoir_states,
        node_names=node_names,
        plot_config=plot_config
    )
    
    # Run evaluation
    results = evaluator.run_evaluation(
        metric='NMSE',
        feature_selection_method='pca',
        num_features='all',
        modeltype='Ridge',
        regression_alpha=0.1,
        train_ratio=train_ratio
    )
    
    # Generate plots
    if show_plots:
        evaluator.plot_results(existing_results=results)
    
    # Log results
    logger.output(f"\nSin(x) Approximation Results:")
    logger.output(f"  Metric: {results['metric']}")
    logger.output(f"  NMSE: {results['accuracy']:.6f}")
    logger.output(f"  RNMSE: {np.sqrt(results['accuracy']):.6f}")
    
    return results


def main():
    """
    Main function to run the ReservoirPy benchmark example.
    """
    logger.info("=" * 60)
    logger.info("RESERVOIRPY + RCBENCH BENCHMARK EXAMPLE")
    logger.info("=" * 60)
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    # Reservoir parameters
    N_UNITS = 100           # Number of reservoir neurons
    LEAK_RATE = 0.3         # Leak rate (memory vs reactivity trade-off)
    SPECTRAL_RADIUS = 0.9   # Spectral radius (edge of chaos)
    INPUT_SCALING = 0.1     # Input weight scaling
    
    # Signal parameters
    N_SAMPLES = 5000        # Number of time steps
    
    # Evaluation parameters
    MAX_DELAY = 30          # Maximum delay for MC task
    TRAIN_RATIO = 0.8       # Train/test split ratio
    SHOW_PLOTS = True       # Display plots
    
    SEED = 42               # Random seed for reproducibility
    
    # =========================================================================
    # CREATE RESERVOIR
    # =========================================================================
    
    logger.info(f"\nCreating reservoir with {N_UNITS} units...")
    logger.info(f"  Leak rate: {LEAK_RATE}")
    logger.info(f"  Spectral radius: {SPECTRAL_RADIUS}")
    logger.info(f"  Input scaling: {INPUT_SCALING}")
    
    reservoir = create_reservoir(
        units=N_UNITS,
        lr=LEAK_RATE,
        sr=SPECTRAL_RADIUS,
        input_scaling=INPUT_SCALING,
        seed=SEED
    )
    
    # =========================================================================
    # GENERATE INPUT SIGNAL
    # =========================================================================
    
    logger.info(f"\nGenerating random input signal ({N_SAMPLES} samples)...")
    input_signal_2d = generate_random_input(n_samples=N_SAMPLES, seed=SEED)
    
    # =========================================================================
    # RUN RESERVOIR
    # =========================================================================
    
    logger.info("Running reservoir...")
    reservoir_states = run_reservoir(reservoir, input_signal_2d)
    
    logger.info(f"  Input shape: {input_signal_2d.shape}")
    logger.info(f"  Reservoir states shape: {reservoir_states.shape}")
    
    # Flatten input for RCbench (expects 1D array)
    input_signal_1d = input_signal_2d.flatten()
    
    # Create node names
    node_names = [f'N{i}' for i in range(N_UNITS)]
    
    # =========================================================================
    # BENCHMARK: MEMORY CAPACITY
    # =========================================================================
    
    mc_results = benchmark_memory_capacity(
        input_signal=input_signal_1d,
        reservoir_states=reservoir_states,
        node_names=node_names,
        max_delay=MAX_DELAY,
        train_ratio=TRAIN_RATIO,
        show_plots=SHOW_PLOTS
    )
    
    # =========================================================================
    # BENCHMARK: SIN(X) APPROXIMATION
    # =========================================================================
    
    sinx_results = benchmark_sinx(
        input_signal=input_signal_1d,
        reservoir_states=reservoir_states,
        node_names=node_names,
        train_ratio=TRAIN_RATIO,
        show_plots=SHOW_PLOTS
    )
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 60)
    logger.output(f"\nReservoir Configuration:")
    logger.output(f"  Units: {N_UNITS}")
    logger.output(f"  Leak Rate: {LEAK_RATE}")
    logger.output(f"  Spectral Radius: {SPECTRAL_RADIUS}")
    logger.output(f"  Input Scaling: {INPUT_SCALING}")
    logger.output(f"\nResults:")
    logger.output(f"  Total Memory Capacity: {mc_results['total_memory_capacity']:.4f}")
    logger.output(f"  Sin(x) NMSE: {sinx_results['accuracy']:.6f}")
    
    return {
        'mc_results': mc_results,
        'sinx_results': sinx_results,
        'reservoir_config': {
            'units': N_UNITS,
            'leak_rate': LEAK_RATE,
            'spectral_radius': SPECTRAL_RADIUS,
            'input_scaling': INPUT_SCALING
        }
    }


if __name__ == "__main__":
    results = main()

