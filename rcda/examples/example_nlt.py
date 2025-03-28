import logging
from pathlib import Path

from rcda.measurements.loader import MeasurementLoader
from rcda.measurements.parser import MeasurementParser
from rcda.tasks.nlt import NltEvaluator
from rcda.logger import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.INFO) #use 25 for output only, use logging.INFO for output and INFO 

BASE_DIR = Path(__file__).resolve().parent.parent

filenameNLT = "837_INRiMJanis_NWN_Pad99C_grid_SE_Custom Wave Measurement_2024_02_01.txt"

measurement_file_NLT = BASE_DIR.parent / "tests" / "test_files" / filenameNLT

# Load and structure the measurement data
loaderNLT = MeasurementLoader(measurement_file_NLT)
datasetNLT = loaderNLT.get_dataset()

# Automatic parsing
parserNLT = MeasurementParser(datasetNLT)

electrodes_infoNLT = parserNLT.summary()
logger.info(f"Parsed Electrodes: {electrodes_infoNLT}")

input_elec = parserNLT.input_electrodes[0]
input_signal = parserNLT.get_input_voltages()[input_elec]
time = datasetNLT.time
nodes_output = parserNLT.get_node_voltages()

# Run NLT evaluation
evaluatorNLT = NltEvaluator(
    input_signal=input_signal,
    nodes_output=nodes_output,
    time_array=time,
    waveform_type='sine'  # or 'triangular'
)

# Run evaluation for each generated target waveform
resultsNLT = {}
for target_name in evaluatorNLT.targets:
    result = evaluatorNLT.run_evaluation(
        target_name=target_name,
        metric='NMSE',
        feature_selection_method='kbest',
        num_features='all',
        regression_alpha=0.01,
        train_ratio=0.8,
        plot=True,
    )
    resultsNLT[target_name] = result
    # Print results clearly
    logger.output(f"NLT Analysis for Target: '{target_name}'")
    logger.output(f"  - Metric: {result['metric']}")
    logger.output(f"  - Accuracy: {result['accuracy']:.5f}")
    logger.output(f"  - Selected Features Indices: {[electrodes_infoNLT['node_electrodes'][i] for i in result['selected_features']]}")
    logger.output(f"  - Model Weights: {result['model'].coef_}\n")

    


    
   