import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from rcbench import ElecResDataset  # Explicitly use electrical functionality
from rcbench.tasks.nlt import NltEvaluator
from rcbench.visualization.plot_config import NLTPlotConfig
from rcbench.logger import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.INFO) #use 25 for output only, use logging.INFO for output and INFO 

BASE_DIR = Path(__file__).resolve().parent.parent

filenameNLT = "034_INRiMPXIe_NWN_Pad285T_SO_IVMeasurement_2025_09_05.txt"

measurement_file_NLT = BASE_DIR.parent / "tests" / "test_files" / filenameNLT

# Load the data directly using the ElecResDataset class
dataset = ElecResDataset(measurement_file_NLT)

# Get information about the nodes
nodes_info = dataset.summary()
logger.info(f"Parsed Nodes: {nodes_info}")

# Get input and node voltages directly from the dataset
input_node = nodes_info['input_nodes'][0]
input_signal = dataset.get_input_voltages()[input_node]
time = dataset.time

# Get node voltages (only computation nodes, not input)
nodes_output = dataset.get_node_voltages()
node_names = nodes_info['nodes']

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
    node_names=node_names,
    plot_config=plot_config
)

# Run evaluations for all targets
logger.info("Running evaluations for all targets...")
for target_name in evaluatorNLT.targets:
    try:
        result = evaluatorNLT.run_evaluation(
            target_name=target_name,
            metric='NMSE',
            feature_selection_method='kbest',
            num_features='all',
            modeltype="Ridge",  # Changed from model to modeltype
            regression_alpha=0.1,
            plot=False
        )
        logger.output(f"NLT Task: {target_name}")
        logger.output(f"  - Metric: {result['metric']}")
        logger.output(f"  - Accuracy: {result['accuracy']:.5f}")
    except Exception as e:
        logger.error(f"Error evaluating {target_name}: {str(e)}")

logger.info("Generating comprehensive plots including frequency analysis...")

# Generate plots for all targets 
evaluatorNLT.plot_results()


    


    
   