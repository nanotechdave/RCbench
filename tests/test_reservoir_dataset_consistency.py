import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from rcbench.measurements.dataset import ReservoirDataset
from rcbench.tasks.featureselector import FeatureSelector


@pytest.fixture
def reservoir_dataset():
    """Load measurement data for testing using the ReservoirDataset class."""
    BASE_DIR = Path(__file__).resolve().parent.parent
    filename = "074_INRiMARC_NWN_Pad129M_gridSE_MemoryCapacity_2024_04_02.txt"
    measurement_file = BASE_DIR / "tests" / "test_files" / filename
    
    # Load the data directly using the ReservoirDataset class
    dataset = ReservoirDataset(measurement_file)
    return dataset


def test_node_data_consistency(reservoir_dataset):
    """
    Test that all nodes in feature selection point to the same data as their
    corresponding columns in the raw dataframe.
    
    This test verifies data integrity through the entire pipeline:
    1. Raw data from file
    2. ReservoirDataset processing
    3. Feature selection
    """
    # Get the raw dataframe directly from the dataset
    raw_df = reservoir_dataset.dataframe
    print(f"Raw dataframe shape: {raw_df.shape}")
    
    # Get node information
    nodes_info = reservoir_dataset.summary()
    nodes = nodes_info['nodes']
    
    # Skip if no nodes found
    if not nodes:
        print("ERROR: No computation nodes found in dataset")
        pytest.skip("No computation nodes found")
    
    print(f"Testing data consistency for {len(nodes)} nodes: {nodes}")
    
    # Get the node output matrix
    nodes_output = reservoir_dataset.get_node_voltages()
    print(f"Node output matrix shape: {nodes_output.shape}")
    
    # Setup for feature selection
    input_voltages = reservoir_dataset.get_input_voltages()
    primary_input_node = nodes_info['input_nodes'][0]
    input_signal = input_voltages[primary_input_node]
    
    # Dummy target for feature selection
    y = np.sin(input_signal)
    
    # Run feature selection
    feature_selector = FeatureSelector(random_state=42)
    X_selected, selected_indices, selected_names = feature_selector.select_features(
        X=nodes_output,
        y=y,
        node_names=nodes,
        method='pca',
        num_features='all'
    )
    
    print(f"Selected node names: {selected_names}")
    
    # Track results
    failures = []
    successes = []
    skipped = []
    
    # Check each node
    for node in nodes:
        column = f'{node}_V[V]'
        
        # Check if column exists in raw dataframe
        if column not in raw_df.columns:
            print(f"⚠️ Skipping node {node}: Column {column} not found in raw dataframe")
            skipped.append(node)
            continue
        
        # Get node index in nodes list
        node_idx = nodes.index(node)
        
        # Get raw values from the dataframe (limit to first 10 values for readability)
        raw_values = raw_df[column].values[:10]
        
        # Get values from nodes_output
        node_values = nodes_output[:10, node_idx]
        
        # Check if node is in selected features
        if node not in selected_names:
            print(f"⚠️ Skipping node {node}: Not selected by feature selection")
            skipped.append(node)
            continue
        
        # Get index in selected features
        selected_idx = selected_names.index(node)
        
        # Get values from selected features
        selected_values = X_selected[:10, selected_idx]
        
        try:
            # Verify raw dataframe values match node output values
            assert np.allclose(raw_values, node_values, rtol=1e-5, atol=1e-5)
            
            # Verify raw values match selected feature values
            assert np.allclose(raw_values, selected_values, rtol=1e-5, atol=1e-5)
            
            # Success!
            successes.append(node)
            print(f"✅ Node {node}: Data consistent across raw dataframe, nodes output, and feature selection")
            
        except AssertionError as e:
            failures.append(node)
            print(f"❌ Node {node}: Data mismatch detected")
            print(f"  Raw values: {raw_values}")
            print(f"  Node values: {node_values}")
            print(f"  Selected values: {selected_values}")
    
    # Final report
    print(f"\n=== NODE DATA CONSISTENCY TEST RESULTS ===")
    print(f"✅ {len(successes)}/{len(nodes)} nodes verified successful")
    if skipped:
        print(f"⚠️ {len(skipped)}/{len(nodes)} nodes skipped: {skipped}")
    if failures:
        print(f"❌ {len(failures)}/{len(nodes)} nodes failed: {failures}")
    
    # Final assertion to make the test pass/fail
    assert not failures, f"Data mismatch found for nodes: {failures}"
    
    # If we got here, all checks passed!
    print("\n✅ VERIFICATION SUCCESSFUL: All checked nodes have consistent data across all stages")


def test_specific_node_consistency(reservoir_dataset):
    """
    Test that a specific node '10' in feature selection points to the same data as '10_V[V]' column
    in the raw dataframe.
    """
    # Get the raw dataframe directly from the dataset
    raw_df = reservoir_dataset.dataframe
    
    # Check if the target column exists
    target_node = '10'
    target_column = f'{target_node}_V[V]'
    
    if target_column not in raw_df.columns:
        all_voltage_columns = [col for col in raw_df.columns if '_V[V]' in col]
        print(f"Available voltage columns: {all_voltage_columns}")
        pytest.skip(f"Target column {target_column} not found in raw dataframe")
    
    # Get raw values from the dataframe
    raw_values = raw_df[target_column].values[:20]  # Get first 20 values
    print(f"Raw values from DataFrame['{target_column}'] (first 20):\n{raw_values}")
    
    # Get node information
    nodes_info = reservoir_dataset.summary()
    nodes = nodes_info['nodes']
    
    # Verify target node is in nodes
    if target_node not in nodes:
        pytest.skip(f"Target node {target_node} not found in nodes")
    
    # Get the node output matrix
    nodes_output = reservoir_dataset.get_node_voltages()
    
    # Get the index of target node in nodes
    target_idx = nodes.index(target_node)
    
    # Extract the values for this node from the node_output matrix
    node_values = nodes_output[:20, target_idx]
    print(f"Node values from nodes_output[:, {target_idx}] (first 20):\n{node_values}")
    
    # Verify raw dataframe values match node output values
    assert np.allclose(raw_values, node_values, rtol=1e-5, atol=1e-5), \
        f"Data mismatch between raw dataframe and node output for node {target_node}"
    
    # Now run feature selection
    input_voltages = reservoir_dataset.get_input_voltages()
    primary_input_node = nodes_info['input_nodes'][0]
    input_signal = input_voltages[primary_input_node]
    
    # Dummy target for feature selection
    y = np.sin(input_signal)
    
    # Run feature selection
    feature_selector = FeatureSelector(random_state=42)
    X_selected, selected_indices, selected_names = feature_selector.select_features(
        X=nodes_output,
        y=y,
        node_names=nodes,
        method='pca',
        num_features='all'
    )
    
    print(f"Selected node names: {selected_names}")
    
    # Verify node '10' is in the selected features
    if target_node not in selected_names:
        pytest.skip(f"Target node {target_node} was not selected by feature selection")
    
    # Get the index of target node in selected_names
    selected_idx = selected_names.index(target_node)
    
    # Get values from selected features
    selected_values = X_selected[:20, selected_idx]
    print(f"Selected feature values (first 20):\n{selected_values}")
    
    # Final verification that raw dataframe values match selected feature values
    assert np.allclose(raw_values, selected_values, rtol=1e-5, atol=1e-5), \
        f"Data mismatch between raw dataframe and selected feature values for node {target_node}"
    
    # Get values directly from individual node method
    individual_values = reservoir_dataset.get_node_voltage(target_node)[:20]
    print(f"Individual node values (first 20):\n{individual_values}")
    
    # Verify direct access method matches raw values
    assert np.allclose(raw_values, individual_values, rtol=1e-5, atol=1e-5), \
        f"Data mismatch between raw dataframe and individual node values for node {target_node}"
    
    print("\n✅ VERIFICATION SUCCESSFUL: Node '10' data is consistent across all access methods")


def test_all_access_methods_consistent(reservoir_dataset):
    """
    Test that all methods of accessing node data return consistent results.
    """
    # Get node information
    nodes_info = reservoir_dataset.summary()
    nodes = nodes_info['nodes']
    
    # Skip if no nodes found
    if not nodes:
        pytest.skip("No computation nodes found")
    
    # Get the node output matrix
    nodes_output = reservoir_dataset.get_node_voltages()
    
    # Track results
    failures = []
    successes = []
    
    # Check each node
    for node in nodes:
        # Get node index in nodes list
        node_idx = nodes.index(node)
        
        # Get values from nodes_output matrix
        matrix_values = nodes_output[:10, node_idx]
        
        # Get values directly using get_node_voltage method
        try:
            direct_values = reservoir_dataset.get_node_voltage(node)[:10]
            
            # Verify both methods return the same data
            assert np.allclose(matrix_values, direct_values, rtol=1e-5, atol=1e-5)
            
            # Success!
            successes.append(node)
            
        except AssertionError as e:
            failures.append(node)
            print(f"❌ Node {node}: Data mismatch detected")
            print(f"  Matrix values: {matrix_values}")
            print(f"  Direct values: {direct_values}")
    
    # Final report
    print(f"\n=== NODE ACCESS METHODS CONSISTENCY RESULTS ===")
    print(f"✅ {len(successes)}/{len(nodes)} nodes verified consistent")
    if failures:
        print(f"❌ {len(failures)}/{len(nodes)} nodes failed: {failures}")
    
    # Final assertion to make the test pass/fail
    assert not failures, f"Data mismatch found for nodes: {failures}"
    
    print("\n✅ VERIFICATION SUCCESSFUL: All node access methods return consistent data") 