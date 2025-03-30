import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from rcbench.measurements.dataset import ReservoirDataset
from rcbench.tasks.nlt import NltEvaluator
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

# For NLT tasks, we can use input electrodes as state signals
# Since this dataset doesn't have designated node electrodes, we'll use the input electrodes
node_electrode_names = electrodes_info['input_electrodes']
input_voltages = dataset.get_input_voltages()

# Create a matrix of node voltages from the input electrodes
node_voltages = np.column_stack([input_voltages[elec] for elec in node_electrode_names])

# Run NLT evaluation
evaluatorNLT = NltEvaluator(
    input_signal=input_signal,
    nodes_output=node_voltages,
    time_array=time,
    waveform_type='sine',  # or 'triangular'
    electrode_names=node_electrode_names
)

# Run evaluation for each generated target waveform
resultsNLT = {}
for target_name in evaluatorNLT.targets:
    try:
        result = evaluatorNLT.run_evaluation(
            target_name=target_name,
            metric='NMSE',
            feature_selection_method='kbest',
            num_features='all',
            regression_alpha=0.01,
            train_ratio=0.8,
            plot=True,  # Enable plotting
        )
        resultsNLT[target_name] = result
        # Print results clearly
        logger.output(f"NLT Analysis for Target: '{target_name}'")
        logger.output(f"  - Metric: {result['metric']}")
        logger.output(f"  - Accuracy: {result['accuracy']:.5f}")
        logger.output(f"  - Selected Features Indices: {[node_electrode_names[i] for i in result['selected_features']]}")
        logger.output(f"  - Model Weights: {result['model'].coef_}\n")
    except Exception as e:
        logger.error(f"Error evaluating {target_name}: {str(e)}")

    


    
   