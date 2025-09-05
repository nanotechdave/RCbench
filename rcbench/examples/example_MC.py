import logging
from pathlib import Path

from rcbench import ElecResDataset
from rcbench.tasks.memorycapacity import MemoryCapacityEvaluator
from rcbench.visualization.plot_config import MCPlotConfig
from rcbench.logger import get_logger


logger = get_logger(__name__)
logger.setLevel(logging.INFO) #use 25 for output only, use logging.INFO for output and INFO 

BASE_DIR = Path(__file__).resolve().parent.parent
filenameMC = "074_INRiMARC_NWN_Pad129M_gridSE_MemoryCapacity_2024_04_02.txt"

measurement_file_MC = BASE_DIR.parent / "tests" / "test_files" / filenameMC

# Load the data directly using the ElecResDataset class
dataset = ElecResDataset(measurement_file_MC)
print(dataset.dataframe.head())
# Get information about the nodes
nodes_info = dataset.summary()
logger.info(f"Parsed Nodes: {nodes_info}")

# Get input and node voltages directly from the dataset
input_voltages = dataset.get_input_voltages()
nodes_output = dataset.get_node_voltages()

primary_input_node = nodes_info['input_nodes'][0]
input_signal = input_voltages[primary_input_node]

# Get node names from the dataset
node_names = nodes_info['nodes']
logger.info(f"Computation nodes: {node_names}")

# Create MC plot configuration
plot_config = MCPlotConfig(
    save_dir=None,
    figsize=(6, 4),  # Save plots to this directory
    # MC-specific plot options
    plot_mc_curve=True,          # Plot the memory capacity vs delay curve
    plot_predictions=True,        # Plot predictions for each delay
    plot_total_mc=True,           # Plot cumulative memory capacity
    max_delays_to_plot=5,         # Maximum number of delays to plot predictions for
    
    # General reservoir property plots
    plot_input_signal=True,       # Plot the input signal
    plot_output_responses=True,   # Plot node responses
    plot_nonlinearity=True,       # Plot nonlinearity of nodes
    plot_frequency_analysis=True, # Plot frequency analysis
    
    # Plot styling options
    nonlinearity_plot_style='scatter',
    frequency_range=(0, 50)       # Limit frequency range to 0-50 Hz
)

# Create evaluator with plot configuration
evaluatorMC = MemoryCapacityEvaluator(
    input_signal, 
    nodes_output, 
    max_delay=30,
    node_names=node_names,
    plot_config=plot_config
)


resultsMC = evaluatorMC.calculate_total_memory_capacity(
    feature_selection_method='pca',
    num_features='all',
    modeltype='Ridge',
    regression_alpha=0.1,
    train_ratio=0.8,
)

# Generate plots based on the plotter's configuration
evaluatorMC.plot_results()

logger.output(f"Total Memory Capacity: {resultsMC['total_memory_capacity']:.4f}\n")
logger.info(f"keys: {resultsMC.keys()}")

    


    
   