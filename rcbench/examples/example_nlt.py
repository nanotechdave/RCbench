import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from rcbench.measurements.dataset import ReservoirDataset
from rcbench.tasks.nlt import NltEvaluator
from rcbench.visualization.plot_config import NLTPlotConfig
from rcbench.logger import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.INFO) #use 25 for output only, use logging.INFO for output and INFO 

BASE_DIR = Path(__file__).resolve().parent.parent

filenameNLT = "837_INRiMJanis_NWN_Pad99C_grid_SE_Custom Wave Measurement_2024_02_01.txt"

measurement_file_NLT = BASE_DIR.parent / "tests" / "test_files" / filenameNLT

# Load the data directly using the ReservoirDataset class
dataset = ReservoirDataset(measurement_file_NLT)

# Get information about the electrodes
electrodes_info = dataset.summary()
logger.info(f"Parsed Electrodes: {electrodes_info}")

# Get input and node voltages directly from the dataset
input_elec = electrodes_info['input_electrodes'][0]
input_signal = dataset.get_input_voltages()[input_elec]
time = dataset.time

# Get node voltages (only node electrodes, not input)
nodes_output = dataset.get_node_voltages()
electrode_names = electrodes_info['node_electrodes']

# Create NLT plot configuration
plot_config = NLTPlotConfig(
    save_dir=None,  # Save plots to this directory
    
    # General reservoir property plots
    plot_input_signal=True,         # Plot the input signal
    plot_output_responses=True,     # Plot node responses
    plot_nonlinearity=True,         # Plot nonlinearity of nodes
    plot_frequency_analysis=True,   # Plot frequency analysis
    
    # Target-specific plots
    plot_target_prediction=True,    # Plot target vs prediction results
    
    # Plot styling options
    nonlinearity_plot_style='scatter',
    frequency_range=(0, 20)         # Limit frequency range to 0-20 Hz for clearer visualization
)

# Run NLT evaluation with plot config
evaluatorNLT = NltEvaluator(
    input_signal=input_signal,
    nodes_output=nodes_output,
    time_array=time,
    waveform_type='sine',  # or 'triangular'
    electrode_names=electrode_names,
    plot_config=plot_config
)

# Run evaluation without plots first to compute all results
logger.info("Running evaluations for all targets...")
resultsNLT = {}
for target_name in evaluatorNLT.targets:
    try:
        result = evaluatorNLT.run_evaluation(
            target_name=target_name,
            metric='NMSE',
            feature_selection_method='pca',
            num_features='all',
            regression_alpha=0.01,
            train_ratio=0.8,
            plot=False,  # Don't plot during evaluation
        )
        resultsNLT[target_name] = result
        # Print results clearly
        logger.output(f"NLT Analysis for Target: '{target_name}'")
        logger.output(f"  - Metric: {result['metric']}")
        logger.output(f"  - Accuracy: {result['accuracy']:.5f}")
        logger.output(f"  - Selected Features Indices: {[electrode_names[i] for i in result['selected_features']]}")
        logger.output(f"  - Model Weights: {result['model'].coef_}\n")
    except Exception as e:
        logger.error(f"Error evaluating {target_name}: {str(e)}")

# Generate all plots in one go, including frequency analysis
# Pass the pre-computed results to avoid re-computing everything
logger.info("Generating comprehensive plots including frequency analysis...")
evaluatorNLT.plot_results(existing_results=resultsNLT)


    


    
   