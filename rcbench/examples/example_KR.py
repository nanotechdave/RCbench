import logging
from pathlib import Path

from rcbench.measurements.dataset import ReservoirDataset
from rcbench.tasks.kernelrank import KernelRankEvaluator
from rcbench.tasks.generalizationrank import GeneralizationRankEvaluator
from rcbench.logger import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.INFO) #use 25 for output only, use logging.INFO for output and INFO 

# use MC measurement to evaluate kernel rank
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

# Computing kernel rank via kernel rank evaluator (using generalization rank for svd)

general = GeneralizationRankEvaluator(nodes_output)
resultgen = general.run_evaluation()

logger.output(f"Kernel Analysis:")
logger.output(f"  - Rank: {resultgen['generalization_rank']}\n")



    
   