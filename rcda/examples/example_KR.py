import logging
from pathlib import Path

from rcda.measurements.loader import MeasurementLoader
from rcda.measurements.parser import MeasurementParser
from rcda.tasks.kernelrank import KernelRankEvaluator
from rcda.logger import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.INFO) #use 25 for output only, use logging.INFO for output and INFO 

# use MC measurement to evaluate kernel rank
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

# Computing kernel rank via kernel rank evaluator
evaluatorKernel = KernelRankEvaluator(nodes_output,threshold=1e-6)

resultsKernel = evaluatorKernel.run_evaluation()

logger.output(f"Kernel Analysis:")
logger.output(f"  - Rank: {resultsKernel['kernel_rank']}\n")



    
   