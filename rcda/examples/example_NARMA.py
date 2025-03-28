import logging
from pathlib import Path

from rcda.measurements.loader import MeasurementLoader
from rcda.measurements.parser import MeasurementParser
from rcda.tasks.narma import NarmaEvaluator
from rcda.logger import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.INFO) #use 25 for output only, use logging.INFO for output and INFO 

# using a MC experimental measurement, compatible with narma evaluation
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

evaluatorNARMA = NarmaEvaluator(input_signal, 
                                nodes_output, 
                                order=2, 
                                alpha=0.4, 
                                beta=0.4, 
                                gamma=0.6, 
                                delta=0.1,
                                )
resultsNARMA2 = evaluatorNARMA.run_evaluation()

logger.output(f"NARMA Analysis for order: {evaluatorNARMA.order}")
logger.output(f"  - Metric: {resultsNARMA2['metric']}")
logger.output(f"  - Accuracy: {resultsNARMA2['accuracy']:.5f}\n")


    


    
   