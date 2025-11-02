#!/usr/bin/env python3
"""
Test script for importing dataset_test.csv as ElecResDataset and performing NLT evaluation.

This script demonstrates how to:
1. Load a CSV dataset using ElecResDataset
2. Extract input signals and node outputs
3. Perform NLT (Non-Linear Transformation) evaluation
4. Display results for all available transformation targets
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd

from rcbench.measurements.dataset import ElecResDataset
from rcbench.tasks.nlt import NltEvaluator
from rcbench.visualization.plot_config import NLTPlotConfig
from rcbench.logger import get_logger

# Configure logging
logger = get_logger(__name__)
logger.setLevel(logging.INFO)

def main():
    """Main function to test dataset import and NLT evaluation."""
    
    logger.info("=== Testing dataset_test.csv Import and NLT Evaluation ===")
    
    # Define file paths
    BASE_DIR = Path(__file__).resolve().parent
    dataset_file = BASE_DIR / "test_files" / "dataset_test.csv"
    
    if not dataset_file.exists():
        logger.error(f"Dataset file not found: {dataset_file}")
        return
    
    logger.info(f"Loading dataset from: {dataset_file}")
    
    try:
        # Step 1: Load the dataset using ElecResDataset
        dataset = ElecResDataset(dataset_file)
        
        # Display dataset information
        logger.info(f"Dataset loaded successfully!")
        logger.info(f"Dataset shape: {dataset.dataframe.shape}")
        logger.info(f"Time column: {dataset.time_column}")
        logger.info(f"Measurement type: {dataset.measurement_type}")
        
        # Get node information
        logger.info(f"Input nodes: {dataset.input_nodes}")
        logger.info(f"Ground nodes: {dataset.ground_nodes}")
        logger.info(f"Computation nodes: {dataset.nodes}")
        logger.info(f"Total voltage columns: {len(dataset.voltage_columns)}")
        logger.info(f"Total current columns: {len(dataset.current_columns)}")
        
        # Step 2: Extract data for NLT evaluation
        logger.info("\n--- Extracting data for NLT evaluation ---")
        
        # Get time array
        time = dataset.time
        logger.info(f"Time array shape: {time.shape}")
        logger.info(f"Time range: [{time.min():.6f}, {time.max():.6f}] seconds")
        
        # Get input signal from the first input node
        if not dataset.input_nodes:
            logger.error("No input nodes found in the dataset!")
            return
            
        input_node = dataset.input_nodes[0]
        input_voltages = dataset.get_input_voltages()
        input_signal = input_voltages[input_node]
        
        logger.info(f"Using input node: {input_node}")
        logger.info(f"Input signal shape: {input_signal.shape}")
        logger.info(f"Input signal range: [{input_signal.min():.6f}, {input_signal.max():.6f}] V")
        
        # Get node outputs (computation nodes only)
        nodes_output = dataset.get_node_voltages()
        node_names = dataset.nodes
        
        logger.info(f"Node outputs shape: {nodes_output.shape}")
        logger.info(f"Number of computation nodes: {len(node_names)}")
        logger.info(f"Node names: {node_names}")
        
        # Step 3: Create NLT plot configuration
        logger.info("\n--- Configuring NLT evaluation ---")
        
        plot_config = NLTPlotConfig(
            save_dir=None,  # Don't save plots automatically
            
            # General reservoir property plots
            plot_input_signal=True,         # Plot the input signal
            plot_output_responses=True,     # Plot node responses
            plot_nonlinearity=True,         # Plot nonlinearity of nodes
            plot_frequency_analysis=True,   # Plot frequency analysis
            
            # Target-specific plots
            plot_target_prediction=True,    # Plot target vs prediction results
            
            # Plot styling options
            nonlinearity_plot_style='scatter',
            frequency_range=(0, 20),        # Limit frequency range for clearer visualization
            show_plot=False                 # Set to True if you want to display plots
        )
        
        # Step 4: Create NLT evaluator
        logger.info("Creating NLT evaluator...")
        
        evaluator = NltEvaluator(
            input_signal=input_signal,
            nodes_output=nodes_output,
            time_array=time,
            waveform_type='sine',  # Can be 'sine' or 'triangular'
            node_names=node_names,
            plot_config=plot_config
        )
        
        # Display available targets
        available_targets = list(evaluator.targets.keys())
        logger.info(f"Available transformation targets: {available_targets}")
        
        # Step 5: Run NLT evaluation for all targets
        logger.info("\n--- Running NLT evaluations ---")
        
        all_results = {}
        
        for target_name in available_targets:
            try:
                logger.info(f"Evaluating target: {target_name}")
                
                result = evaluator.run_evaluation(
                    target_name=target_name,
                    metric='NMSE',                    # Normalized Mean Squared Error
                    feature_selection_method='kbest', # K-best feature selection
                    num_features='all',               # Use all features
                    modeltype="Ridge",                # Ridge regression
                    regression_alpha=0.1,             # Regularization parameter
                    train_ratio=0.8,                  # 80% for training, 20% for testing
                    plot=False                        # Don't generate individual plots
                )
                
                all_results[target_name] = result
                
                # Display results
                logger.output(f"Target: {target_name}")
                logger.output(f"  - Metric: {result['metric']}")
                logger.output(f"  - Accuracy (NMSE): {result['accuracy']:.6f}")
                logger.output(f"  - Selected features: {result.get('selected_features_count', 'N/A')}")
                logger.output(f"  - Model type: {result.get('model_type', 'Ridge')}")
                
            except Exception as e:
                logger.error(f"Error evaluating {target_name}: {str(e)}")
                all_results[target_name] = {"error": str(e)}
        
        # Step 6: Summary of results
        logger.info("\n--- Summary of Results ---")
        
        successful_evaluations = {k: v for k, v in all_results.items() if "error" not in v}
        failed_evaluations = {k: v for k, v in all_results.items() if "error" in v}
        
        logger.output(f"Successful evaluations: {len(successful_evaluations)}/{len(all_results)}")
        
        if successful_evaluations:
            # Find best and worst performing targets
            accuracies = {k: v['accuracy'] for k, v in successful_evaluations.items()}
            best_target = min(accuracies, key=accuracies.get)  # Lower NMSE is better
            worst_target = max(accuracies, key=accuracies.get)
            
            logger.output(f"Best performing target: {best_target} (NMSE: {accuracies[best_target]:.6f})")
            logger.output(f"Worst performing target: {worst_target} (NMSE: {accuracies[worst_target]:.6f})")
            logger.output(f"Average NMSE: {np.mean(list(accuracies.values())):.6f}")
        
        if failed_evaluations:
            logger.error(f"Failed evaluations: {list(failed_evaluations.keys())}")
        
        # Optional: Generate comprehensive plots
        if plot_config.show_plot:
            logger.info("\n--- Generating comprehensive plots ---")
            evaluator.plot_results()
        
        logger.info("\n=== Test completed successfully! ===")
        
        return all_results
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise


def test_dataset_import_only():
    """Test function to only import and inspect the dataset without running NLT evaluation."""
    
    logger.info("=== Testing dataset_test.csv Import Only ===")
    
    BASE_DIR = Path(__file__).resolve().parent
    dataset_file = BASE_DIR / "test_files" / "dataset_test.csv"
    
    if not dataset_file.exists():
        logger.error(f"Dataset file not found: {dataset_file}")
        return None
    
    try:
        # Load dataset
        dataset = ElecResDataset(dataset_file)
        
        # Display comprehensive information
        logger.info(f"Dataset file: {dataset_file.name}")
        logger.info(f"Dataset shape: {dataset.dataframe.shape}")
        logger.info(f"Columns: {list(dataset.dataframe.columns)}")
        logger.info(f"Time range: [{dataset.time.min():.6f}, {dataset.time.max():.6f}] seconds")
        logger.info(f"Sampling frequency: ~{1/np.mean(np.diff(dataset.time)):.1f} Hz")
        
        # Node information
        logger.info(f"Input nodes: {dataset.input_nodes}")
        logger.info(f"Ground nodes: {dataset.ground_nodes}")
        logger.info(f"Computation nodes: {dataset.nodes}")
        
        # Data quality checks
        logger.info(f"Missing values: {dataset.dataframe.isnull().sum().sum()}")
        logger.info(f"Infinite values: {np.isinf(dataset.dataframe.select_dtypes(include=[np.number])).sum().sum()}")
        
        return dataset
        
    except Exception as e:
        logger.error(f"Error importing dataset: {str(e)}")
        raise


if __name__ == "__main__":
    # You can run either the full test or just the import test
    
    # Option 1: Full test with NLT evaluation
    results = main()
    
    # Option 2: Just test dataset import (uncomment to use instead)
    # dataset = test_dataset_import_only()
