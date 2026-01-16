"""
Information Processing Capacity (IPC) Benchmark Example

This example demonstrates how to evaluate the total Information Processing Capacity
of a simulated Echo State Network (ESN) using the IPC framework from Dambre et al. (2012).

The IPC framework decomposes the reservoir's computational capacity into:
- Linear memory capacity (degree-1 Legendre polynomials)
- Nonlinear capacity (higher-degree polynomials and cross-terms)

Reference:
    Dambre, J., Verstraeten, D., Schrauwen, B. & Massar, S.
    "Information Processing Capacity of Dynamical Systems"
    Scientific Reports 2, 514 (2012). DOI: 10.1038/srep00514

Requirements:
    pip install reservoirpy

Author: Davide Pilati
Date: 2025
"""

import logging
import numpy as np

# ReservoirPy imports
try:
    from reservoirpy.nodes import Reservoir
except ImportError:
    raise ImportError(
        "reservoirpy is required for this example. "
        "Install it with: pip install reservoirpy"
    )

# RCbench imports
from rcbench import IPCEvaluator, MemoryCapacityEvaluator
from rcbench import IPCPlotConfig, MCPlotConfig
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
    n_samples: int = 5000,
    seed: int = 42
) -> np.ndarray:
    """
    Generate a uniformly distributed random input signal in [-1, 1].
    
    This is important for IPC as Legendre polynomials are orthonormal
    under the uniform distribution on [-1, 1].
    
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
    # Uniform distribution between -1 and 1 (required for Legendre basis)
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
    states = reservoir.run(input_signal)
    return states


def benchmark_ipc(
    input_signal: np.ndarray,
    reservoir_states: np.ndarray,
    node_names: list,
    max_delay: int = 10,
    max_degree: int = 3,
    train_ratio: float = 0.8,
    show_plots: bool = True
) -> dict:
    """
    Benchmark the reservoir's Information Processing Capacity using RCbench.
    
    Parameters:
    -----------
    input_signal : np.ndarray
        Input signal (1D array, uniform in [-1, 1])
    reservoir_states : np.ndarray
        Reservoir node outputs with shape (n_samples, n_nodes)
    node_names : list
        Names of reservoir nodes
    max_delay : int
        Maximum delay to consider
    max_degree : int
        Maximum polynomial degree per variable
    train_ratio : float
        Ratio of data used for training
    show_plots : bool
        Whether to display plots
        
    Returns:
    --------
    dict
        IPC results
    """
    logger.info("=" * 60)
    logger.info("INFORMATION PROCESSING CAPACITY BENCHMARK")
    logger.info("=" * 60)
    logger.info("\nBased on Dambre et al. (2012) 'Information Processing")
    logger.info("Capacity of Dynamical Systems', Scientific Reports 2, 514")
    
    # Create IPC plot configuration
    plot_config = IPCPlotConfig(
        save_dir=None,
        figsize=(10, 6),
        show_plot=show_plots,
        plot_capacity_by_degree=True,
        plot_tradeoff=True,
        plot_summary=True,
        plot_input_signal=False,
        plot_output_responses=False,
        plot_nonlinearity=False,
        plot_frequency_analysis=False,
        train_ratio=train_ratio
    )
    
    # Create IPC evaluator
    evaluator = IPCEvaluator(
        input_signal=input_signal,
        nodes_output=reservoir_states,
        max_delay=max_delay,
        max_degree=max_degree,
        include_cross_terms=True,  # Include product terms
        random_state=42,
        node_names=node_names,
        plot_config=plot_config
    )
    
    # Calculate total information processing capacity
    results = evaluator.calculate_total_capacity(
        feature_selection_method='pca',
        num_features='all',
        modeltype='Ridge',
        regression_alpha=0.1,
        train_ratio=train_ratio
    )
    
    # Generate plots
    if show_plots:
        evaluator.plot_results()
    
    # Get detailed analysis
    tradeoff = evaluator.get_memory_nonlinearity_tradeoff()
    summary = evaluator.summary()
    
    # Log detailed results
    logger.output(f"\n{'='*60}")
    logger.output("DETAILED IPC RESULTS")
    logger.output(f"{'='*60}")
    logger.output(f"\nTotal Capacity: {results['total_capacity']:.4f}")
    logger.output(f"Theoretical Maximum (N nodes): {results['theoretical_max']}")
    logger.output(f"Efficiency: {summary['efficiency']*100:.1f}%")
    logger.output(f"\nCapacity Decomposition:")
    logger.output(f"  Linear Memory (degree=1): {results['linear_memory_capacity']:.4f}")
    logger.output(f"  Nonlinear (degree>1): {results['nonlinear_capacity']:.4f}")
    logger.output(f"  Linear/Nonlinear Ratio: {summary['linear_nonlinear_ratio']:.2f}")
    
    logger.output(f"\nCapacity by Polynomial Degree:")
    for degree in sorted(results['capacity_by_degree'].keys()):
        cap = results['capacity_by_degree'][degree]
        logger.output(f"  Degree {degree}: {cap:.4f}")
    
    logger.output(f"\nCapacity by Delay (single-term functions):")
    for delay in sorted(results['capacity_by_delay'].keys())[:10]:
        data = results['capacity_by_delay'][delay]
        logger.output(f"  τ={delay:2d}: Linear={data['linear']:.3f}, "
                     f"Nonlinear={data['nonlinear']:.3f}, Total={data['total']:.3f}")
    
    return results


def benchmark_memory_capacity(
    input_signal: np.ndarray,
    reservoir_states: np.ndarray,
    node_names: list,
    max_delay: int = 30,
    train_ratio: float = 0.8,
    show_plots: bool = True
) -> dict:
    """
    Benchmark standard Memory Capacity for comparison.
    
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
    logger.info("\n" + "=" * 60)
    logger.info("STANDARD MEMORY CAPACITY BENCHMARK (for comparison)")
    logger.info("=" * 60)
    
    # Create MC plot configuration
    plot_config = MCPlotConfig(
        save_dir=None,
        figsize=(10, 6),
        plot_mc_curve=True,
        plot_predictions=False,
        plot_total_mc=True,
        max_delays_to_plot=5,
        plot_input_signal=False,
        plot_output_responses=False,
        plot_nonlinearity=False,
        plot_frequency_analysis=False,
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
    
    logger.output(f"\nStandard MC Results:")
    logger.output(f"  Total Memory Capacity: {results['total_memory_capacity']:.4f}")
    logger.output(f"  Theoretical Maximum: {reservoir_states.shape[1]}")
    logger.output(f"  MC/N ratio: {results['total_memory_capacity']/reservoir_states.shape[1]:.2%}")
    
    return results


def main():
    """
    Main function to run the IPC benchmark example.
    """
    logger.info("=" * 60)
    logger.info("INFORMATION PROCESSING CAPACITY (IPC) EXAMPLE")
    logger.info("=" * 60)
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    # Reservoir parameters
    N_UNITS = 16           # Number of reservoir neurons
    LEAK_RATE = 0.3         # Leak rate (memory vs reactivity trade-off)
    SPECTRAL_RADIUS = 0.9   # Spectral radius (edge of chaos)
    INPUT_SCALING = 0.1     # Input weight scaling
    
    # Signal parameters
    N_SAMPLES = 5000        # Number of time steps
    
    # IPC evaluation parameters
    MAX_DELAY_IPC = 10      # Maximum delay for IPC (keep small for speed)
    MAX_DEGREE = 3          # Maximum polynomial degree
    
    # Standard MC parameters (for comparison)
    MAX_DELAY_MC = 30       # Maximum delay for standard MC
    
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
    logger.info("  Distribution: Uniform[-1, 1] (required for Legendre basis)")
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
    # BENCHMARK: INFORMATION PROCESSING CAPACITY
    # =========================================================================
    
    ipc_results = benchmark_ipc(
        input_signal=input_signal_1d,
        reservoir_states=reservoir_states,
        node_names=node_names,
        max_delay=MAX_DELAY_IPC,
        max_degree=MAX_DEGREE,
        train_ratio=TRAIN_RATIO,
        show_plots=SHOW_PLOTS
    )
    
    # =========================================================================
    # BENCHMARK: STANDARD MEMORY CAPACITY (for comparison)
    # =========================================================================
    
    mc_results = benchmark_memory_capacity(
        input_signal=input_signal_1d,
        reservoir_states=reservoir_states,
        node_names=node_names,
        max_delay=MAX_DELAY_MC,
        train_ratio=TRAIN_RATIO,
        show_plots=SHOW_PLOTS
    )
    
    # =========================================================================
    # COMPARISON SUMMARY
    # =========================================================================
    
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON: IPC vs STANDARD MC")
    logger.info("=" * 60)
    
    logger.output(f"\nReservoir Configuration:")
    logger.output(f"  Units: {N_UNITS}")
    logger.output(f"  Leak Rate: {LEAK_RATE}")
    logger.output(f"  Spectral Radius: {SPECTRAL_RADIUS}")
    logger.output(f"  Input Scaling: {INPUT_SCALING}")
    
    logger.output(f"\nCapacity Comparison:")
    logger.output(f"  {'Metric':<35} {'Value':>10}")
    logger.output(f"  {'-'*45}")
    logger.output(f"  {'Theoretical Maximum (N)':<35} {N_UNITS:>10}")
    logger.output(f"  {'Standard MC (degree=1 only)':<35} {mc_results['total_memory_capacity']:>10.4f}")
    logger.output(f"  {'IPC Linear Memory (degree=1)':<35} {ipc_results['linear_memory_capacity']:>10.4f}")
    logger.output(f"  {'IPC Nonlinear (degree>1)':<35} {ipc_results['nonlinear_capacity']:>10.4f}")
    logger.output(f"  {'IPC Total Capacity':<35} {ipc_results['total_capacity']:>10.4f}")
    
    logger.output(f"\nKey Insight from Dambre et al. (2012):")
    logger.output(f"  The total capacity should approach N if the system has")
    logger.output(f"  fading memory and linearly independent state variables.")
    logger.output(f"  Current efficiency: {ipc_results['total_capacity']/N_UNITS*100:.1f}% of theoretical max")
    
    return {
        'ipc_results': ipc_results,
        'mc_results': mc_results,
        'reservoir_config': {
            'units': N_UNITS,
            'leak_rate': LEAK_RATE,
            'spectral_radius': SPECTRAL_RADIUS,
            'input_scaling': INPUT_SCALING
        }
    }


if __name__ == "__main__":
    results = main()

