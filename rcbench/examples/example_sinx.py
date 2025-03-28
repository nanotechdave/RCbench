import logging
from pathlib import Path

from rcbench.measurements.loader import MeasurementLoader
from rcbench.measurements.parser import MeasurementParser
from rcbench.tasks.sinx import SinxEvaluator
from rcbench.logger import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.INFO) #use 25 for output only, use logging.INFO for output and INFO 

# using a MC file because I need an experiment with a random input x to perform NLT to sin(x)
BASE_DIR = Path(__file__).resolve().parent.parent
filenameMC = "074_INRiMARC_NWN_Pad129M_gridSE_MemoryCapacity_2024_04_02.txt"

measurement_file_MC = BASE_DIR.parent / "tests" / "test_files" / filenameMC

loaderSinx = MeasurementLoader(measurement_file_MC)
datasetSinx = loaderSinx.get_dataset()

# Automatic parsing
parserSinx = MeasurementParser(datasetSinx)

electrodes_infoSinx = parserSinx.summary()
logger.info(f"Parsed Electrodes: {electrodes_infoSinx}")

input_voltages = parserSinx.get_input_voltages()
nodes_output = parserSinx.get_node_voltages()

primary_input_electrode = electrodes_infoSinx['input_electrodes'][0]
input_signal = input_voltages[primary_input_electrode]

evaluatorSinX = SinxEvaluator(input_signal, nodes_output)
resultSinX = evaluatorSinX.run_evaluation(
    metric='NMSE',
    feature_selection_method='kbest',
    num_features='all',
    regression_alpha=0.1
)

logger.output(f"SinX Accuracy ({resultSinX['metric']}): {resultSinX['accuracy']:.4f}\n")




    


    
   