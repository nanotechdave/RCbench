"""
PRCpy + RCbench Integration Example

This example demonstrates how to use PRCpy (Physical Reservoir Computing Python)
for data processing and RCbench for benchmarking. PRCpy is designed for processing
experimental data from physical reservoir computing systems.

Since PRCpy requires physical experimental data files, this example:
1. Shows how to generate synthetic data in PRCpy-compatible format
2. Demonstrates PRCpy's data processing pipeline
3. Uses RCbench for comprehensive benchmarking (MC and Sin(x) tasks)

Requirements:
    pip install prcpy

PRCpy Documentation: https://pypi.org/project/prcpy/

Author: Davide Pilati
Date: 2025
"""

import logging
import numpy as np
import os
import tempfile

# PRCpy imports
try:
    from prcpy.RC import Pipeline
    from prcpy.TrainingModels.RegressionModels import define_Ridge
    from prcpy.Maths.Target_functions import generate_square_wave
except ImportError:
    raise ImportError(
        "prcpy is required for this example. "
        "Install it with: pip install prcpy"
    )

# RCbench imports
from rcbench import MemoryCapacityEvaluator, SinxEvaluator
from rcbench import MCPlotConfig, SinxPlotConfig
from rcbench.logger import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.INFO)


def generate_synthetic_prc_data(
    output_dir: str,
    n_samples: int = 1000,
    n_readouts: int = 50,
    prefix: str = "scan",
    n_files: int = 1,
    seed: int = 42
) -> tuple:
    """
    Generate synthetic physical reservoir computing data files in PRCpy format.
    
    This simulates what you would get from a physical reservoir experiment,
    where each file contains frequency (input) and spectra (readout) data.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the synthetic data files
    n_samples : int
        Number of time samples per file
    n_readouts : int
        Number of readout channels (simulating physical readouts)
    prefix : str
        File prefix (PRCpy expects files matching this pattern)
    n_files : int
        Number of data files to generate
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (input_signal, readout_matrix) - The generated data
    """
    np.random.seed(seed)
    
    # Generate random input signal (simulating input frequency/voltage)
    input_signal = np.random.uniform(-1, 1, size=n_samples)
    
    # Generate synthetic readout responses
    # In a physical reservoir, readouts would be measurements from the device
    # Here we simulate nonlinear transformations of the input
    readout_matrix = np.zeros((n_samples, n_readouts))
    
    for i in range(n_readouts):
        # Each readout channel has different nonlinear characteristics
        delay = np.random.randint(0, 10)
        nonlinearity = np.random.uniform(0.5, 2.0)
        noise_level = np.random.uniform(0.01, 0.1)
        
        # Create delayed and nonlinearly transformed version of input
        delayed_input = np.roll(input_signal, delay)
        
        # Apply nonlinear transformation (simulating physical response)
        readout = np.tanh(nonlinearity * delayed_input)
        
        # Add some noise (simulating measurement noise)
        readout += np.random.normal(0, noise_level, size=n_samples)
        
        readout_matrix[:, i] = readout
    
    # Save data files in PRCpy-compatible format
    os.makedirs(output_dir, exist_ok=True)
    
    for file_idx in range(n_files):
        filename = f"{prefix}_{file_idx:04d}.txt"
        filepath = os.path.join(output_dir, filename)
        
        # PRCpy expects tab-separated data with headers
        # First column: Frequency (input), remaining columns: Spectra (readouts)
        header = "Frequency\t" + "\t".join([f"Ch{i}" for i in range(n_readouts)])
        
        # Combine input and readouts
        data = np.column_stack([input_signal, readout_matrix])
        
        np.savetxt(filepath, data, delimiter='\t', header=header, comments='')
    
    logger.info(f"Generated {n_files} synthetic data file(s) in {output_dir}")
    
    return input_signal, readout_matrix


def run_prcpy_pipeline(
    data_dir: str,
    prefix: str = "scan",
    target_type: str = "square_wave"
) -> dict:
    """
    Run PRCpy's RC pipeline on the data.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing data files
    prefix : str
        File prefix pattern
    target_type : str
        Type of target function ('square_wave')
        
    Returns:
    --------
    dict
        PRCpy results including nonlinearity and memory capacity
    """
    logger.info("=" * 60)
    logger.info("PRCPY PIPELINE")
    logger.info("=" * 60)
    
    # Define processing parameters (minimal processing for synthetic data)
    process_params = {
        "Xs": "Frequency",
        "Readouts": "Spectra",
        "remove_bg": False,
        "bg_fname": "",
        "smooth": False,
        "smooth_win": 51,
        "smooth_rank": 4,
        "cut_xs": False,
        "x1": 0,
        "x2": 100,
        "normalize_local": False,
        "normalize_global": True,
        "sample": False,
        "sample_rate": 1,
        "transpose": False
    }
    
    try:
        # Create PRCpy pipeline
        rc_pipeline = Pipeline(data_dir, prefix, process_params)
        
        # Get data length for target generation
        length = rc_pipeline.get_df_length()
        logger.info(f"Data length: {length} samples")
        
        # Generate target (square wave for transformation task)
        num_periods = 10
        target_values = generate_square_wave(length, num_periods)
        
        # Add target to pipeline
        rc_pipeline.define_target(target_values)
        
        # Define Ridge regression model
        model_params = {
            "alpha": 1e-3,
            "fit_intercept": True,
            "copy_X": True,
            "max_iter": None,
            "tol": 0.0001,
            "solver": "auto",
            "positive": False,
            "random_state": None,
        }
        model = define_Ridge(model_params)
        
        # Define RC parameters (tau=0 for transformation task)
        rc_params = {
            "model": model,
            "tau": 0,  # No delay for transformation
            "test_size": 0.2,
            "error_type": "MSE"
        }
        
        # Run RC
        rc_pipeline.run(rc_params)
        
        # Get results
        results = rc_pipeline.get_rc_results()
        
        # Get reservoir metrics
        rc_pipeline.define_input(target_values)
        nonlinearity = rc_pipeline.get_non_linearity()
        lmc, _ = rc_pipeline.get_linear_memory_capacity()
        
        logger.output(f"\nPRCpy Results:")
        logger.output(f"  Training Error (MSE): {results.get('train_error', 'N/A')}")
        logger.output(f"  Test Error (MSE): {results.get('test_error', 'N/A')}")
        logger.output(f"  Nonlinearity: {nonlinearity:.4f}")
        logger.output(f"  Linear Memory Capacity: {lmc:.4f}")
        
        return {
            'rc_results': results,
            'nonlinearity': nonlinearity,
            'linear_memory_capacity': lmc
        }
        
    except Exception as e:
        logger.warning(f"PRCpy pipeline error: {e}")
        logger.info("Continuing with RCbench benchmarking...")
        return None


def benchmark_with_rcbench(
    input_signal: np.ndarray,
    readout_matrix: np.ndarray,
    train_ratio: float = 0.8,
    show_plots: bool = True
) -> dict:
    """
    Benchmark the physical reservoir data using RCbench.
    
    Parameters:
    -----------
    input_signal : np.ndarray
        Input signal (1D array)
    readout_matrix : np.ndarray
        Readout matrix with shape (n_samples, n_readouts)
    train_ratio : float
        Ratio of data used for training
    show_plots : bool
        Whether to display plots
        
    Returns:
    --------
    dict
        Benchmark results from RCbench
    """
    # Create node names
    n_readouts = readout_matrix.shape[1]
    node_names = [f'Ch{i}' for i in range(n_readouts)]
    
    # =========================================================================
    # MEMORY CAPACITY BENCHMARK
    # =========================================================================
    
    logger.info("\n" + "=" * 60)
    logger.info("RCBENCH: MEMORY CAPACITY BENCHMARK")
    logger.info("=" * 60)
    
    mc_config = MCPlotConfig(
        save_dir=None,
        figsize=(10, 6),
        plot_mc_curve=True,
        plot_predictions=True,
        plot_total_mc=True,
        max_delays_to_plot=5,
        plot_input_signal=True,
        plot_output_responses=True,
        plot_nonlinearity=True,
        plot_frequency_analysis=False,
        nonlinearity_plot_style='scatter',
        show_plot=show_plots,
        train_ratio=train_ratio
    )
    
    mc_evaluator = MemoryCapacityEvaluator(
        input_signal=input_signal,
        nodes_output=readout_matrix,
        max_delay=20,
        node_names=node_names,
        plot_config=mc_config
    )
    
    mc_results = mc_evaluator.calculate_total_memory_capacity(
        feature_selection_method='pca',
        num_features='all',
        modeltype='Ridge',
        regression_alpha=0.1,
        train_ratio=train_ratio
    )
    
    if show_plots:
        mc_evaluator.plot_results()
    
    logger.output(f"\nRCbench Memory Capacity Results:")
    logger.output(f"  Total MC: {mc_results['total_memory_capacity']:.4f}")
    logger.output(f"  Theoretical Max (N={n_readouts}): {n_readouts}")
    logger.output(f"  MC/N ratio: {mc_results['total_memory_capacity']/n_readouts:.2%}")
    
    # =========================================================================
    # SIN(X) BENCHMARK
    # =========================================================================
    
    logger.info("\n" + "=" * 60)
    logger.info("RCBENCH: SIN(X) APPROXIMATION BENCHMARK")
    logger.info("=" * 60)
    
    sinx_config = SinxPlotConfig(
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
        prediction_sample_count=500
    )
    
    sinx_evaluator = SinxEvaluator(
        input_signal=input_signal,
        nodes_output=readout_matrix,
        node_names=node_names,
        plot_config=sinx_config
    )
    
    sinx_results = sinx_evaluator.run_evaluation(
        metric='NMSE',
        feature_selection_method='pca',
        num_features='all',
        modeltype='Ridge',
        regression_alpha=0.1,
        train_ratio=train_ratio
    )
    
    if show_plots:
        sinx_evaluator.plot_results(existing_results=sinx_results)
    
    logger.output(f"\nRCbench Sin(x) Results:")
    logger.output(f"  Metric: {sinx_results['metric']}")
    logger.output(f"  NMSE: {sinx_results['accuracy']:.6f}")
    logger.output(f"  RNMSE: {np.sqrt(sinx_results['accuracy']):.6f}")
    
    return {
        'mc_results': mc_results,
        'sinx_results': sinx_results
    }


def main():
    """
    Main function demonstrating PRCpy + RCbench integration.
    """
    logger.info("=" * 60)
    logger.info("PRCPY + RCBENCH INTEGRATION EXAMPLE")
    logger.info("=" * 60)
    logger.info("\nPRCpy is designed for processing physical reservoir data.")
    logger.info("This example uses synthetic data to demonstrate the workflow.")
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    N_SAMPLES = 2000        # Number of time samples
    N_READOUTS = 50         # Number of readout channels
    TRAIN_RATIO = 0.8       # Train/test split
    SHOW_PLOTS = True       # Display plots
    SEED = 42               # Random seed
    
    # =========================================================================
    # GENERATE SYNTHETIC DATA
    # =========================================================================
    
    logger.info(f"\nGenerating synthetic PRC data...")
    logger.info(f"  Samples: {N_SAMPLES}")
    logger.info(f"  Readout channels: {N_READOUTS}")
    
    # Create temporary directory for data files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate synthetic data
        input_signal, readout_matrix = generate_synthetic_prc_data(
            output_dir=temp_dir,
            n_samples=N_SAMPLES,
            n_readouts=N_READOUTS,
            prefix="scan",
            n_files=1,
            seed=SEED
        )
        
        logger.info(f"  Input shape: {input_signal.shape}")
        logger.info(f"  Readout matrix shape: {readout_matrix.shape}")
        
        # =====================================================================
        # RUN PRCPY PIPELINE (Optional - may require specific data format)
        # =====================================================================
        
        prcpy_results = run_prcpy_pipeline(temp_dir, prefix="scan")
        
        # =====================================================================
        # BENCHMARK WITH RCBENCH
        # =====================================================================
        
        rcbench_results = benchmark_with_rcbench(
            input_signal=input_signal,
            readout_matrix=readout_matrix,
            train_ratio=TRAIN_RATIO,
            show_plots=SHOW_PLOTS
        )
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 60)
    
    logger.output(f"\nData Configuration:")
    logger.output(f"  Samples: {N_SAMPLES}")
    logger.output(f"  Readout Channels: {N_READOUTS}")
    logger.output(f"  Train Ratio: {TRAIN_RATIO}")
    
    if prcpy_results:
        logger.output(f"\nPRCpy Metrics:")
        logger.output(f"  Nonlinearity: {prcpy_results['nonlinearity']:.4f}")
        logger.output(f"  Linear Memory Capacity: {prcpy_results['linear_memory_capacity']:.4f}")
    
    logger.output(f"\nRCbench Results:")
    logger.output(f"  Total Memory Capacity: {rcbench_results['mc_results']['total_memory_capacity']:.4f}")
    logger.output(f"  Sin(x) NMSE: {rcbench_results['sinx_results']['accuracy']:.6f}")
    
    return {
        'prcpy_results': prcpy_results,
        'rcbench_results': rcbench_results,
        'data_config': {
            'n_samples': N_SAMPLES,
            'n_readouts': N_READOUTS,
            'train_ratio': TRAIN_RATIO
        }
    }


if __name__ == "__main__":
    results = main()

