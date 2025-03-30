import logging
from pathlib import Path

from rcbench.measurements.dataset import ReservoirDataset
from rcbench.tasks.memorycapacity import MemoryCapacityEvaluator, PlotConfig
from rcbench.logger import get_logger


logger = get_logger(__name__)
logger.setLevel(logging.INFO) #use 25 for output only, use logging.INFO for output and INFO 

BASE_DIR = Path(__file__).resolve().parent.parent
filenameMC = "074_INRiMARC_NWN_Pad129M_gridSE_MemoryCapacity_2024_04_02.txt"

measurement_file_MC = BASE_DIR.parent / "tests" / "test_files" / filenameMC

# Load the data directly using the ReservoirDataset class
dataset = ReservoirDataset(measurement_file_MC)

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

# Create plot configuration
plot_config = PlotConfig(
    save_dir=None,  # Save plots to this directory
    plot_mc_vs_delay=False,
    plot_feature_importance=True,
    plot_prediction_results=False,
    plot_cumulative_mc=True,
    plot_mc_heatmap=False
)

evaluatorMC = MemoryCapacityEvaluator(
    input_signal, 
    nodes_output, 
    max_delay=30,
    electrode_names=electrode_names
)

resultsMC = evaluatorMC.calculate_total_memory_capacity(
    feature_selection_method='pca',
    num_features='all',
    regression_alpha=0.1,
    train_ratio=0.8,
)
evaluatorMC.plot_results(plot_config)

logger.output(f"Total Memory Capacity: {resultsMC['total_memory_capacity']:.4f}\n")

    


    
   