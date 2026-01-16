"""
Kernel Rank and Generalization Rank Evaluation Example

This example demonstrates how to evaluate Kernel Rank and Generalization Rank 
using real experimental data.

Based on Wringe et al. (2025) "Reservoir Computing Benchmarks: a tutorial 
review and critique" (arXiv:2405.06561), the Kernel Rank captures the combined
dynamics of the input signal and reservoir states.

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
logger.setLevel(logging.INFO)  # use 25 for output only, use logging.INFO for output and INFO 

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

# =============================================================================
# KERNEL RANK EVALUATION
# =============================================================================
logger.output(f"\nKernel Rank Analysis:")
logger.output(f"  - Data shape: {nodes_output.shape[0]} samples, {nodes_output.shape[1]} nodes")

# Method 1: Combined mode (RECOMMENDED) - includes input signal
# This captures the combined dynamics of input and reservoir as per Wringe et al. (2025)
logger.output(f"\n1. Combined Mode (input + nodes) - Recommended:")
kr_combined = KernelRankEvaluator(
    nodes_output=nodes_output, 
    input_signal=input_signal,  # Include input signal
    kernel='linear', 
    threshold=1e-6
)
kr_results_combined = kr_combined.run_evaluation()
logger.output(f"   - KernelRank (linear): {kr_results_combined['kernel_rank']}")
logger.output(f"   - Features used: {kr_results_combined['n_features']} (1 input + {nodes_output.shape[1]} nodes)")

# With RBF kernel
kr_combined_rbf = KernelRankEvaluator(
    nodes_output=nodes_output,
    input_signal=input_signal,
    kernel='rbf', 
    sigma=1.0, 
    threshold=1e-6
)
kr_results_combined_rbf = kr_combined_rbf.run_evaluation()
logger.output(f"   - KernelRank (RBF): {kr_results_combined_rbf['kernel_rank']}")

# Method 2: Nodes-only mode (backward compatible)
logger.output(f"\n2. Nodes-Only Mode (backward compatible):")
kr_nodes_only = KernelRankEvaluator(
    nodes_output=nodes_output, 
    kernel='linear', 
    threshold=1e-6
)
kr_results_nodes = kr_nodes_only.run_evaluation()
logger.output(f"   - KernelRank (linear): {kr_results_nodes['kernel_rank']}")
logger.output(f"   - Features used: {kr_results_nodes['n_features']} nodes")

# With RBF kernel
kr_nodes_rbf = KernelRankEvaluator(
    nodes_output=nodes_output, 
    kernel='rbf', 
    sigma=1.0, 
    threshold=1e-6
)
kr_results_nodes_rbf = kr_nodes_rbf.run_evaluation()
logger.output(f"   - KernelRank (RBF): {kr_results_nodes_rbf['kernel_rank']}")

# =============================================================================
# GENERALIZATION RANK EVALUATION
# =============================================================================
logger.output(f"\n3. Generalization Rank (for comparison):")
general = GeneralizationRankEvaluator(nodes_output, threshold=1e-6)
result_gen = general.run_evaluation()
logger.output(f"   - GeneralizationRank: {result_gen['generalization_rank']}")
logger.output(f"   - First 5 singular values: {result_gen['singular_values'][:5].round(4)}")

# =============================================================================
# SUMMARY COMPARISON
# =============================================================================
logger.output(f"\n" + "=" * 50)
logger.output(f"SUMMARY COMPARISON")
logger.output(f"=" * 50)
logger.output(f"  Combined (input+nodes), linear: {kr_results_combined['kernel_rank']}")
logger.output(f"  Combined (input+nodes), RBF:    {kr_results_combined_rbf['kernel_rank']}")
logger.output(f"  Nodes-only, linear:             {kr_results_nodes['kernel_rank']}")
logger.output(f"  Nodes-only, RBF:                {kr_results_nodes_rbf['kernel_rank']}")
logger.output(f"  Generalization Rank:            {result_gen['generalization_rank']}")
logger.output(f"\nNote: Combined mode (input+nodes) is recommended per Wringe et al. (2025)")
