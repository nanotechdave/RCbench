"""
Kernel Rank and Generalization Rank Evaluation Example

This example demonstrates how to evaluate Kernel Rank and Generalization Rank using real experimental data.

Author: Davide Pilati
Date: 2025
"""

import logging
from pathlib import Path

from rcbench import ElecResDataset  # Explicitly use electrical functionality
from rcbench.tasks.kernelrank import KernelRankEvaluator
from rcbench.tasks.generalizationrank import GeneralizationRankEvaluator
from rcbench.logger import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.INFO) #use 25 for output only, use logging.INFO for output and INFO 

# use MC measurement to evaluate kernel rank
BASE_DIR = Path(__file__).resolve().parent.parent
filenameMC = "074_INRiMARC_NWN_Pad129M_gridSE_MemoryCapacity_2024_04_02.txt"

measurement_file_MC = BASE_DIR.parent / "tests" / "test_files" / filenameMC

# Load the data directly using the ElecResDataset class
dataset = ElecResDataset(measurement_file_MC)

# Get information about the nodes
nodes_info = dataset.summary()
logger.info(f"Parsed Nodes: {nodes_info}")

# Get input and node voltages directly from the dataset
input_voltages = dataset.get_input_voltages()
nodes_output = dataset.get_node_voltages()

primary_input_node = nodes_info['input_nodes'][0]
input_signal = input_voltages[primary_input_node]

# Computing kernel rank via KernelRankEvaluator (now using SVD)
logger.output(f"Kernel Analysis:")

# Using KernelRankEvaluator with linear kernel
kr_evaluator = KernelRankEvaluator(nodes_output, kernel='linear', threshold=1e-6)
kr_results = kr_evaluator.run_evaluation()
logger.output(f"  - KernelRank (linear kernel): {kr_results['kernel_rank']}")
logger.output(f"  - First few singular values: {kr_results['singular_values'][:5]}")

# Using KernelRankEvaluator with RBF kernel
kr_evaluator_rbf = KernelRankEvaluator(nodes_output, kernel='rbf', sigma=1.0, threshold=1e-6)
kr_results_rbf = kr_evaluator_rbf.run_evaluation()
logger.output(f"  - KernelRank (RBF kernel): {kr_results_rbf['kernel_rank']}")

# Computing generalization rank for comparison
general = GeneralizationRankEvaluator(nodes_output, threshold=1e-6)
result_gen = general.run_evaluation()
logger.output(f"  - GeneralizationRank: {result_gen['generalization_rank']}")
logger.output(f"  - First few singular values: {result_gen['singular_values'][:5]}\n")

# Compare outputs
logger.output(f"Comparison between ranks:")
logger.output(f"  - KernelRank (linear): {kr_results['kernel_rank']}")
logger.output(f"  - KernelRank (RBF): {kr_results_rbf['kernel_rank']}")
logger.output(f"  - GeneralizationRank: {result_gen['generalization_rank']}")

    
   

