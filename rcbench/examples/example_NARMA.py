import logging
from pathlib import Path

from rcbench.measurements.dataset import ReservoirDataset
from rcbench.tasks.narma import NarmaEvaluator
from rcbench.logger import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.INFO) #use 25 for output only, use logging.INFO for output and INFO 

# using a MC experimental measurement, compatible with narma evaluation
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

# Create list of electrode names for the node electrodes
electrode_names = electrodes_info['node_electrodes']

evaluatorNARMA = NarmaEvaluator(input_signal, 
                               nodes_output, 
                               electrode_names=electrode_names,  # Pass electrode names
                               order=2, 
                               alpha=0.4, 
                               beta=0.4, 
                               gamma=0.6, 
                               delta=0.1,
                               )
resultsNARMA2 = evaluatorNARMA.run_evaluation(metric='NMSE',
                                            feature_selection_method='pca',
                                            num_features='all',
                                            regression_alpha=0.01,
                                            train_ratio=0.8,
                                            plot=True
                                        )
logger.output(f"NARMA Analysis for order: {evaluatorNARMA.order}")
logger.output(f"  - Metric: {resultsNARMA2['metric']}")
logger.output(f"  - Accuracy: {resultsNARMA2['accuracy']:.5f}\n")


    


    
   