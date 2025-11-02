"""
PCA Analysis for Memory Capacity Evaluation

This example demonstrates how Memory Capacity varies as a function of the number of PCA features.
It creates two main plots:
1. PCA loadings for each node across different components
2. Total Memory Capacity as a function of the number of selected PCA features

Author: RCbench team
Date: 2024
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

from rcbench import ElecResDataset
from rcbench.tasks.memorycapacity import MemoryCapacityEvaluator
from rcbench.visualization.plot_config import MCPlotConfig
from rcbench.logger import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.INFO)

def analyze_pca_memory_capacity(input_signal: np.ndarray, 
                               nodes_output: np.ndarray, 
                               node_names: List[str],
                               feature_range: range = range(2, 15),
                               max_delay: int = 25) -> Tuple[Dict, Dict]:
    """
    Analyze Memory Capacity as a function of PCA features.
    
    Args:
        input_signal: Input signal array
        nodes_output: Node output matrix
        node_names: List of node names
        feature_range: Range of PCA features to test
        max_delay: Maximum delay for MC evaluation
    
    Returns:
        Tuple of (mc_results, pca_results) dictionaries
    """
    
    mc_results = {}
    pca_results = {}
    
    logger.info(f"Starting PCA analysis for {len(feature_range)} different feature counts...")
    
    for i, num_features in enumerate(feature_range):
        logger.info(f"[{i+1}/{len(feature_range)}] Evaluating with {num_features} PCA features...")
        
        # Create plot config - disable plots for batch processing
        plot_config = MCPlotConfig(show_plot=False)
        
        # Create evaluator
        evaluator = MemoryCapacityEvaluator(
            input_signal=input_signal,
            nodes_output=nodes_output,
            max_delay=max_delay,
            node_names=node_names,
            plot_config=plot_config
        )
        
        # Run MC evaluation with specific number of PCA features
        results = evaluator.calculate_total_memory_capacity(
            feature_selection_method='pca',
            num_features=num_features,
            modeltype='Ridge',
            regression_alpha=0.1,
            train_ratio=0.8
        )
        
        # Store MC results
        mc_results[num_features] = {
            'total_mc': results['total_memory_capacity'],
            'delay_results': results['delay_results'],
            'selected_features': evaluator.selected_feature_names
        }
        
        # Access PCA results from the feature selector
        pca_selector = evaluator.feature_selector
        if pca_selector.pca is not None:
            pca_results[num_features] = {
                'components': pca_selector.pca.components_.copy(),
                'explained_variance_ratio': pca_selector.pca.explained_variance_ratio_.copy(),
                'feature_importance': pca_selector.get_feature_importance().copy(),
                'selected_indices': pca_selector.selected_features.copy(),
                'selected_names': pca_selector.selected_feature_names.copy()
            }
        
        logger.info(f"  Total MC: {results['total_memory_capacity']:.4f}")
        logger.info(f"  Selected features: {evaluator.selected_feature_names}")
    
    logger.info("PCA analysis completed!")
    return mc_results, pca_results

def plot_pca_loadings_improved(pca_results: Dict, node_names: List[str], max_components: int = 5):
    """
    Plot improved PCA loadings visualization that immediately shows node importance.
    
    Args:
        pca_results: Dictionary containing PCA analysis results
        node_names: List of node names
        max_components: Maximum number of components to consider
    """
    
    # Choose a representative feature count for plotting loadings
    # Use the maximum feature count available
    max_features = max(pca_results.keys())
    components = pca_results[max_features]['components']
    explained_variance = pca_results[max_features]['explained_variance_ratio']
    
    # Limit to max_components
    n_components = min(max_components, components.shape[0], len(node_names))
    
    # Calculate weighted average of absolute loadings (weighted by explained variance)
    weighted_importance = np.zeros(len(node_names))
    for i in range(n_components):
        weighted_importance += np.abs(components[i]) * explained_variance[i]
    
    # Create single plot visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort nodes by importance for better visualization
    importance_order = np.argsort(weighted_importance)[::-1]
    sorted_names = [node_names[i] for i in importance_order]
    sorted_importance = weighted_importance[importance_order]
    
    # Color code: high importance = blue, low importance = red
    colors = plt.cm.RdYlBu(np.linspace(0.2, 0.8, len(node_names)))
    
    bars = ax.bar(range(len(node_names)), sorted_importance, color=colors, 
                  alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_title('Overall Node Importance\n(Weighted by Explained Variance)', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Nodes (sorted by importance)', fontsize=14)
    ax.set_ylabel('Weighted Importance Score', fontsize=14)
    ax.set_xticks(range(len(node_names)))
    ax.set_xticklabels(sorted_names, rotation=45, ha='right', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels for top nodes
    for i, (bar, importance) in enumerate(zip(bars[:5], sorted_importance[:5])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(sorted_importance),
                f'{importance:.3f}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary of most important nodes
    print("\n" + "="*60)
    print("NODE IMPORTANCE SUMMARY")
    print("="*60)
    print("Ranking based on weighted importance (explained variance weighted):")
    for i, (name, score) in enumerate(zip(sorted_names[:8], sorted_importance[:8])):
        print(f"{i+1:2d}. {name:<15} | Score: {score:.4f}")
    print("="*60)

def plot_node_importance_radar(pca_results: Dict, node_names: List[str], top_nodes: int = 8):
    """
    Create a radar plot showing the top most important nodes.
    
    Args:
        pca_results: Dictionary containing PCA analysis results
        node_names: List of node names
        top_nodes: Number of top nodes to show in radar plot
    """
    
    max_features = max(pca_results.keys())
    components = pca_results[max_features]['components']
    explained_variance = pca_results[max_features]['explained_variance_ratio']
    
    # Calculate weighted importance
    n_components = min(5, components.shape[0], len(node_names))
    weighted_importance = np.zeros(len(node_names))
    for i in range(n_components):
        weighted_importance += np.abs(components[i]) * explained_variance[i]
    
    # Get top nodes
    top_indices = np.argsort(weighted_importance)[::-1][:top_nodes]
    top_names = [node_names[i] for i in top_indices]
    top_scores = weighted_importance[top_indices]
    
    # Normalize scores to [0, 1]
    top_scores_norm = top_scores / np.max(top_scores)
    
    # Create radar plot
    angles = np.linspace(0, 2*np.pi, top_nodes, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    top_scores_norm = np.concatenate([top_scores_norm, [top_scores_norm[0]]])
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot the scores
    ax.plot(angles, top_scores_norm, 'o-', linewidth=2, color='navy', markersize=8)
    ax.fill(angles, top_scores_norm, alpha=0.25, color='lightblue')
    
    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(top_names, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True)
    
    # Add value labels
    for angle, score, name in zip(angles[:-1], top_scores_norm[:-1], top_names):
        ax.text(angle, score + 0.05, f'{score:.2f}', ha='center', va='center',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.set_title(f'Top {top_nodes} Most Important Nodes\n(Radar Plot)', 
                size=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()

def plot_pca_loadings(pca_results: Dict, node_names: List[str], max_components: int = 5):
    """
    Master function that creates all improved PCA loading visualizations.
    """
    logger.info("Creating improved PCA loadings visualizations...")
    
    # Main comprehensive plot
    plot_pca_loadings_improved(pca_results, node_names, max_components)
    
    # Bonus radar plot for top nodes
    plot_node_importance_radar(pca_results, node_names, top_nodes=8)

def plot_mc_vs_features(mc_results: Dict, pca_results: Dict):
    """
    Plot Total Memory Capacity as a function of number of PCA features.
    
    Args:
        mc_results: Dictionary containing MC analysis results
        pca_results: Dictionary containing PCA analysis results
    """
    
    feature_counts = sorted(mc_results.keys())
    total_mcs = [mc_results[n]['total_mc'] for n in feature_counts]
    
    # Create single plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Memory Capacity vs Number of Features
    ax.plot(feature_counts, total_mcs, 'o-', linewidth=2, markersize=8, color='navy', 
            markerfacecolor='lightblue', markeredgecolor='navy', markeredgewidth=2)
    ax.set_xlabel('Number of PCA Features', fontsize=14)
    ax.set_ylabel('Total Memory Capacity', fontsize=14)
    ax.set_title('Memory Capacity vs Number of PCA Features', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(feature_counts)
    
    # Add value labels
    for x, y in zip(feature_counts, total_mcs):
        ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', 
                    fontsize=11, fontweight='bold', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def plot_feature_selection_heatmap(mc_results: Dict, node_names: List[str]):
    """
    Plot a heatmap showing which features are selected for different numbers of PCA components.
    
    Args:
        mc_results: Dictionary containing MC analysis results
        node_names: List of node names
    """
    
    feature_counts = sorted(mc_results.keys())
    n_nodes = len(node_names)
    
    # Create selection matrix
    selection_matrix = np.zeros((len(feature_counts), n_nodes))
    
    for i, n_features in enumerate(feature_counts):
        selected_names = mc_results[n_features]['selected_features']
        for j, node_name in enumerate(node_names):
            if node_name in selected_names:
                selection_matrix[i, j] = 1
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(selection_matrix, cmap='RdYlBu_r', aspect='auto', interpolation='nearest')
    
    # Set ticks and labels
    ax.set_xticks(range(n_nodes))
    ax.set_xticklabels(node_names, rotation=45, ha='right')
    ax.set_yticks(range(len(feature_counts)))
    ax.set_yticklabels([f'{n} features' for n in feature_counts])
    
    # Add title and labels
    ax.set_title('Feature Selection Pattern Across Different PCA Components', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Node Names', fontsize=12)
    ax.set_ylabel('Number of PCA Features', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Selected (1) / Not Selected (0)', fontsize=10)
    cbar.set_ticks([0, 1])
    
    # Add grid
    ax.set_xticks(np.arange(n_nodes) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(feature_counts)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
    
    plt.tight_layout()
    plt.show()

def plot_electrode_grid_heatmap(pca_results: Dict, node_names: List[str], max_components: int = 5):
    """
    Plot a heatmap of electrodes arranged in a 4x4 grid layout, colored by importance.
    
    Grid layout (after relabeling):
    13, 14, 15, 16
     9, 10, 11, 12
     5,  6,  7,  8
     1,  2,  3,  4
    
    Args:
        pca_results: Dictionary containing PCA analysis results
        node_names: List of node names (should be relabeled)
        max_components: Maximum number of components to consider
    """
    
    # Calculate weighted importance (same as in the main plot)
    max_features = max(pca_results.keys())
    components = pca_results[max_features]['components']
    explained_variance = pca_results[max_features]['explained_variance_ratio']
    
    # Limit to max_components
    n_components = min(max_components, components.shape[0], len(node_names))
    
    # Calculate weighted average of absolute loadings (weighted by explained variance)
    weighted_importance = np.zeros(len(node_names))
    for i in range(n_components):
        weighted_importance += np.abs(components[i]) * explained_variance[i]
    
    # Create mapping from node names to importance scores
    node_importance_map = {}
    for name, importance in zip(node_names, weighted_importance):
        node_importance_map[name] = importance
    
    # Define the 4x4 grid layout
    grid_layout = [
        [13, 14, 15, 16],
        [9, 10, 11, 12],
        [5, 6, 7, 8],
        [1, 2, 3, 4]
    ]
    
    # Create the importance matrix for the heatmap
    importance_matrix = np.zeros((4, 4))
    label_matrix = [[''] * 4 for _ in range(4)]
    
    for row in range(4):
        for col in range(4):
            electrode_num = grid_layout[row][col]
            node_name = f'Node_{electrode_num}'
            
            if node_name in node_importance_map:
                importance_matrix[row, col] = node_importance_map[node_name]
                label_matrix[row][col] = str(electrode_num)
            else:
                importance_matrix[row, col] = 0  # Default value for missing nodes
                label_matrix[row][col] = str(electrode_num)
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Use the same colormap approach as the first plot
    # Map importance values to the same color range (0.2-0.8) used in the first plot
    min_importance = np.min(importance_matrix[importance_matrix > 0])  # Exclude zeros if any
    max_importance = np.max(importance_matrix)
    
    # Normalize importance values to range [0.2, 0.8] like the first plot
    normalized_importance = 0.2 + 0.6 * (importance_matrix - min_importance) / (max_importance - min_importance)
    
    # Use the same colormap as the first plot (blue = high, red = low)
    cmap = plt.cm.RdYlBu
    
    # Create the image with the normalized color mapping
    im = ax.imshow(normalized_importance, cmap=cmap, vmin=0.2, vmax=0.8, aspect='equal', interpolation='nearest')
    
    # Set ticks and labels
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(['Col 1', 'Col 2', 'Col 3', 'Col 4'])
    ax.set_yticklabels(['Row 1', 'Row 2', 'Row 3', 'Row 4'])
    
    # Add electrode numbers as text annotations
    for row in range(4):
        for col in range(4):
            electrode_num = grid_layout[row][col]
            importance_val = importance_matrix[row, col]
            
            # Choose text color based on background intensity
            text_color = 'white' if importance_val > np.max(importance_matrix) * 0.6 else 'black'
            
            # Add electrode number
            ax.text(col, row, str(electrode_num), ha='center', va='center',
                   fontsize=16, fontweight='bold', color=text_color)
            
            # Add importance value below electrode number
            ax.text(col, row + 0.25, f'{importance_val:.3f}', ha='center', va='center',
                   fontsize=10, fontweight='bold', color=text_color)
    
    # Customize the plot
    ax.set_title('Electrode Grid - Node Importance Heatmap\n(Weighted by Explained Variance)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Column Position', fontsize=12)
    ax.set_ylabel('Row Position', fontsize=12)
    
    # Add colorbar with actual importance values
    import matplotlib.colors as mcolors
    
    # Create a colorbar that shows actual importance values
    norm = mcolors.Normalize(vmin=min_importance, vmax=max_importance)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Weighted Importance Score', fontsize=12)
    
    # Remove tick marks for cleaner look
    ax.tick_params(which='both', length=0)
    
    # Add grid lines
    for i in range(5):
        ax.axhline(i - 0.5, color='white', linewidth=2)
        ax.axvline(i - 0.5, color='white', linewidth=2)
    
    plt.tight_layout()
    plt.show()
    
    # Print grid layout information
    print("\n" + "="*50)
    print("ELECTRODE GRID LAYOUT")
    print("="*50)
    print("Grid arrangement (with importance scores):")
    for row in range(4):
        row_str = ""
        for col in range(4):
            electrode_num = grid_layout[row][col]
            importance_val = importance_matrix[row, col]
            row_str += f"{electrode_num:2d}({importance_val:.3f})  "
        print(f"Row {row + 1}: {row_str}")
    print("="*50)

def main():
    """Main function to run the PCA Memory Capacity analysis."""
    
    logger.info("="*80)
    logger.info("PCA MEMORY CAPACITY ANALYSIS")
    logger.info("="*80)
    
    # Load the dataset
    BASE_DIR = Path(__file__).resolve().parent.parent
    filenameMC = "074_INRiMARC_NWN_Pad129M_gridSE_MemoryCapacity_2024_04_02.txt"
    measurement_file_MC = BASE_DIR.parent / "tests" / "test_files" / filenameMC
    
    logger.info(f"Loading dataset: {filenameMC}")
    dataset = ElecResDataset(measurement_file_MC)
    
    # Get data and information
    nodes_info = dataset.summary()
    input_voltages = dataset.get_input_voltages()
    nodes_output = dataset.get_node_voltages()
    
    primary_input_node = nodes_info['input_nodes'][0]
    input_signal = input_voltages[primary_input_node]
    node_names = nodes_info['nodes']
    
    # Apply custom node relabeling
    # Change from: 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
    # Change to:   1,2,4,6,16,3,5,7,15,12,8,9,14,13,11,10
    new_node_labels = [1,2,4,6,16,3,5,7,15,12,8,9,14,13,11,10]
    
    # Create relabeled node names
    if len(node_names) <= len(new_node_labels):
        relabeled_node_names = [f'Node_{new_node_labels[i]}' for i in range(len(node_names))]
        logger.info(f"Applied custom node relabeling:")
        logger.info(f"  Original: {node_names}")
        logger.info(f"  Relabeled: {relabeled_node_names}")
        node_names = relabeled_node_names
    else:
        logger.warning(f"Dataset has {len(node_names)} nodes but only {len(new_node_labels)} relabeling mappings provided. Using original names.")
    
    logger.info(f"Dataset loaded:")
    logger.info(f"  - Nodes: {len(node_names)}")
    logger.info(f"  - Data points: {len(input_signal)}")
    logger.info(f"  - Node names: {node_names}")
    
    # Define feature range to analyze
    max_available_features = min(14, len(node_names))
    feature_range = range(2, max_available_features + 1)
    
    logger.info(f"Analyzing PCA features from {min(feature_range)} to {max(feature_range)}")
    
    # Run PCA analysis
    mc_results, pca_results = analyze_pca_memory_capacity(
        input_signal=input_signal,
        nodes_output=nodes_output,
        node_names=node_names,
        feature_range=feature_range,
        max_delay=25
    )
    
    # Display results summary
    logger.output("\n" + "="*60)
    logger.output("RESULTS SUMMARY")
    logger.output("="*60)
    
    for n_features in sorted(mc_results.keys()):
        total_mc = mc_results[n_features]['total_mc']
        selected_features = mc_results[n_features]['selected_features']
        
        # Get explained variance if available
        if n_features in pca_results:
            explained_var = np.sum(pca_results[n_features]['explained_variance_ratio'][:n_features]) * 100
            logger.output(f"{n_features:2d} features: MC = {total_mc:6.4f}, "
                         f"Explained Var = {explained_var:5.1f}%, "
                         f"Selected: {', '.join(selected_features[:3])}{'...' if len(selected_features) > 3 else ''}")
        else:
            logger.output(f"{n_features:2d} features: MC = {total_mc:6.4f}, "
                         f"Selected: {', '.join(selected_features[:3])}{'...' if len(selected_features) > 3 else ''}")
    
    # Find optimal number of features
    feature_counts = list(mc_results.keys())
    total_mcs = [mc_results[n]['total_mc'] for n in feature_counts]
    optimal_idx = np.argmax(total_mcs)
    optimal_features = feature_counts[optimal_idx]
    optimal_mc = total_mcs[optimal_idx]
    
    logger.output(f"\nOptimal configuration:")
    logger.output(f"  - Number of features: {optimal_features}")
    logger.output(f"  - Memory Capacity: {optimal_mc:.4f}")
    logger.output(f"  - Selected nodes: {mc_results[optimal_features]['selected_features']}")
    
    # Generate plots
    logger.info("\nGenerating plots...")
    
    # Plot 1: PCA loadings
    logger.info("Creating PCA loadings plot...")
    plot_pca_loadings(pca_results, node_names, max_components=5)
    
    # Plot 2: MC vs Features
    logger.info("Creating MC vs Features plot...")
    plot_mc_vs_features(mc_results, pca_results)
    
    # Plot 3: Feature selection heatmap
    logger.info("Creating feature selection heatmap...")
    plot_feature_selection_heatmap(mc_results, node_names)

    # Plot 4: Electrode grid heatmap
    logger.info("Creating electrode grid heatmap...")
    plot_electrode_grid_heatmap(pca_results, node_names, max_components=5)
    
    logger.info("\n" + "="*80)
    logger.info("PCA ANALYSIS COMPLETE!")
    logger.info("="*80)
    logger.info("Three plots have been generated:")
    logger.info("1. PCA loadings for each node across components")
    logger.info("2. Memory Capacity vs Number of PCA features")
    logger.info("3. Feature selection pattern heatmap")
    logger.info("4. Electrode grid heatmap")

if __name__ == "__main__":
    main() 