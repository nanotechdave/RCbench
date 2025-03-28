import logging
from pathlib import Path

from rcda.measurements.loader import MeasurementLoader
from rcda.measurements.parser import MeasurementParser
from rcda.tasks.memorycapacity import MemoryCapacityEvaluator
from rcda.logger import get_logger


logger = get_logger(__name__)
logger.setLevel(logging.INFO) #use 25 for output only, use logging.INFO for output and INFO 

BASE_DIR = Path(__file__).resolve().parent.parent
filenameMC = "074_INRiMARC_NWN_Pad129M_gridSE_MemoryCapacity_2024_04_02.txt"

measurement_file_MC = BASE_DIR.parent / "tests" / "test_files" / filenameMC

loaderMC = MeasurementLoader(measurement_file_MC)
datasetMC = loaderMC.get_dataset()

# Automatic parsing
parserMC = MeasurementParser(datasetMC)

electrodes_infoMC = parserMC.summary()
logger.info(f"Parsed Electrodes: {electrodes_infoMC}")

input_voltages = parserMC.get_input_voltages()
nodes_output = parserMC.get_node_voltages()

primary_input_electrode = electrodes_infoMC['input_electrodes'][0]
input_signal = input_voltages[primary_input_electrode]

evaluatorMC = MemoryCapacityEvaluator(input_signal, nodes_output, max_delay=30)

resultsMC = evaluatorMC.calculate_total_memory_capacity(
    feature_selection_method='kbest',
    num_features='all',
    regression_alpha=0.1,
    train_ratio=0.8
)

logger.output(f"Total Memory Capacity: {resultsMC['total_memory_capacity']:.4f}\n")

    


    
   