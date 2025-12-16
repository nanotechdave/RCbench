"""
Sin(x) Task Evaluation Example

This example demonstrates how to evaluate Sin(x) approximation using real experimental data.

Author: Davide Pilati
Date: 2025
"""

import logging
from pathlib import Path

from rcbench import ElecResDataset  # Explicitly use electrical functionality
from rcbench.tasks.sinx import SinxEvaluator
from rcbench.logger import get_logger
from rcbench.visualization.plot_config import SinxPlotConfig

logger = get_logger(__name__)
logger.setLevel(logging.INFO) #use 25 for output only, use logging.INFO for output and INFO 

# using a MC file because I need an experiment with a random input x to perform NLT to sin(x)
BASE_DIR = Path(__file__).resolve().parent.parent
filenameMC = "074_INRiMARC_NWN_Pad129M_gridSE_MemoryCapacity_2024_04_02.txt"

measurement_file_MC = BASE_DIR.parent / "tests" / "test_files" / filenameMC

# Load the data directly using the ElecResDataset class
dataset = ElecResDataset(measurement_file_MC)

# Get information about the nodes
nodes_info = dataset.summary()
logger.info(f"Parsed Nodes: {nodes_info}")

# Get input and node voltages directly from the dataset
input_voltages = dataset.get_input_voltages()
nodes_output = dataset.get_node_voltages()

primary_input_node = nodes_info['input_nodes'][0]
input_signal = input_voltages[primary_input_node]

# Get node names for the computation nodes
node_names = nodes_info['nodes']

# Create plot configuration with all plots enabled
plot_config = SinxPlotConfig(
    plot_input_signal=True,
    plot_output_responses=True,
    plot_nonlinearity=True,
    plot_frequency_analysis=True,
    plot_target_prediction=True,
    show_plot=True
)

evaluatorSinx = SinxEvaluator(
    input_signal=input_signal, 
    nodes_output=nodes_output,
    node_names=node_names,
    plot_config=plot_config
)

resultSinx = evaluatorSinx.run_evaluation(
    metric='NMSE',
    feature_selection_method='kbest',
    num_features='all',
    modeltype="Ridge",  # Changed from model to modeltype
    regression_alpha=0.1
)

logger.output(f"Sinx Accuracy ({resultSinx['metric']}): {resultSinx['accuracy']:.4f}\n")

# Generate all plots
evaluatorSinx.plot_results(existing_results=resultSinx)


    


    
   

