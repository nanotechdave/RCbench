import logging
from pathlib import Path

from rcbench.measurements.dataset import ReservoirDataset
from rcbench.tasks.memorycapacity import MemoryCapacityEvaluator
from rcbench.visualization.plot_config import MCPlotConfig
from rcbench.logger import get_logger


logger = get_logger(__name__)
logger.setLevel(logging.INFO) #use 25 for output only, use logging.INFO for output and INFO 

BASE_DIR = Path(__file__).resolve().parent.parent
filenameMC = "074_INRiMARC_NWN_Pad129M_gridSE_MemoryCapacity_2024_04_02.txt"

measurement_file_MC = BASE_DIR.parent / "tests" / "test_files" / filenameMC

# Load the data directly using the ReservoirDataset class
dataset = ReservoirDataset(measurement_file_MC)
print(dataset.dataframe.head())
# Get information about the electrodes
electrodes_info = dataset.summary()
logger.info(f"Parsed Electrodes: {electrodes_info}")

# Get input and node voltages directly from the dataset
input_voltages = dataset.get_input_voltages()
nodes_output = dataset.get_node_voltages()

primary_input_electrode = electrodes_info['input_electrodes'][0]
input_signal = input_voltages[primary_input_electrode]

# Get electrode names from the dataset
electrode_names = electrodes_info['node_electrodes']
logger.info(f"Node electrodes: {electrode_names}")

# Create MC plot configuration
plot_config = MCPlotConfig(
    save_dir=None,
    figsize=(10, 8),  # Save plots to this directory
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
    electrode_names=electrode_names,
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

    


    
   