import logging
from pathlib import Path

from rcbench.measurements.dataset import ReservoirDataset
from rcbench.tasks.sinx import SinxEvaluator
from rcbench.logger import get_logger
from rcbench.visualization.plot_config import SinxPlotConfig

logger = get_logger(__name__)
logger.setLevel(logging.INFO) #use 25 for output only, use logging.INFO for output and INFO 

# using a MC file because I need an experiment with a random input x to perform NLT to sin(x)
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

# Get electrode names for the node electrodes
electrode_names = electrodes_info['node_electrodes']

# Create plot configuration with all plots enabled
plot_config = SinxPlotConfig(
    plot_input_signal=True,
    plot_output_responses=True,
    plot_nonlinearity=True,
    plot_frequency_analysis=True,
    plot_target_prediction=True,
    show_plot=True
)

evaluatorSinX = SinxEvaluator(
    input_signal=input_signal, 
    nodes_output=nodes_output,
    electrode_names=electrode_names,
    plot_config=plot_config
)

resultSinX = evaluatorSinX.run_evaluation(
    metric='NMSE',
    feature_selection_method='kbest',
    num_features='all',
    model="Linear",
    regression_alpha=0.1
)

logger.output(f"SinX Accuracy ({resultSinX['metric']}): {resultSinX['accuracy']:.4f}\n")

# Generate all plots
evaluatorSinX.plot_results(existing_results=resultSinX)




    


    
   