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


def test_electrode_data_consistency(reservoir_dataset):
    """
    Test that all electrodes in feature selection point to the same data as their
    corresponding columns in the raw dataframe.
    
    This test verifies data integrity through the entire pipeline:
    1. Raw data from file
    2. ReservoirDataset processing
    3. Feature selection
    """
    # Get the raw dataframe directly from the dataset
    raw_df = reservoir_dataset.dataframe
    print(f"Raw dataframe shape: {raw_df.shape}")
    
    # Get electrode information
    electrodes_info = reservoir_dataset.summary()
    node_electrodes = electrodes_info['node_electrodes']
    
    # Skip if no electrodes found
    if not node_electrodes:
        print("ERROR: No node electrodes found in dataset")
        pytest.skip("No node electrodes found")
    
    print(f"Testing data consistency for {len(node_electrodes)} electrodes: {node_electrodes}")
    
    # Get the node output matrix
    nodes_output = reservoir_dataset.get_node_voltages()
    print(f"Node output matrix shape: {nodes_output.shape}")
    
    # Setup for feature selection
    input_voltages = reservoir_dataset.get_input_voltages()
    primary_input_electrode = electrodes_info['input_electrodes'][0]
    input_signal = input_voltages[primary_input_electrode]
    
    # Dummy target for feature selection
    y = np.sin(input_signal)
    
    # Run feature selection
    feature_selector = FeatureSelector(random_state=42)
    X_selected, selected_indices, selected_names = feature_selector.select_features(
        X=nodes_output,
        y=y,
        electrode_names=node_electrodes,
        method='pca',
        num_features='all'
    )
    
    print(f"Selected electrode names: {selected_names}")
    
    # Track results
    failures = []
    successes = []
    skipped = []
    
    # Check each electrode
    for electrode in node_electrodes:
        column = f'{electrode}_V[V]'
        
        # Check if column exists in raw dataframe
        if column not in raw_df.columns:
            print(f"⚠️ Skipping electrode {electrode}: Column {column} not found in raw dataframe")
            skipped.append(electrode)
            continue
        
        # Get electrode index in node_electrodes list
        node_idx = node_electrodes.index(electrode)
        
        # Get raw values from the dataframe (limit to first 10 values for readability)
        raw_values = raw_df[column].values[:10]
        
        # Get values from nodes_output
        node_values = nodes_output[:10, node_idx]
        
        # Check if electrode is in selected features
        if electrode not in selected_names:
            print(f"⚠️ Skipping electrode {electrode}: Not selected by feature selection")
            skipped.append(electrode)
            continue
        
        # Get index in selected features
        selected_idx = selected_names.index(electrode)
        
        # Get values from selected features
        selected_values = X_selected[:10, selected_idx]
        
        try:
            # Verify raw dataframe values match node output values
            assert np.allclose(raw_values, node_values, rtol=1e-5, atol=1e-5)
            
            # Verify raw values match selected feature values
            assert np.allclose(raw_values, selected_values, rtol=1e-5, atol=1e-5)
            
            # Success!
            successes.append(electrode)
            print(f"✅ Electrode {electrode}: Data consistent across raw dataframe, nodes output, and feature selection")
            
        except AssertionError as e:
            failures.append(electrode)
            print(f"❌ Electrode {electrode}: Data mismatch detected")
            print(f"  Raw values: {raw_values}")
            print(f"  Node values: {node_values}")
            print(f"  Selected values: {selected_values}")
    
    # Final report
    print(f"\n=== ELECTRODE DATA CONSISTENCY TEST RESULTS ===")
    print(f"✅ {len(successes)}/{len(node_electrodes)} electrodes verified successful")
    if skipped:
        print(f"⚠️ {len(skipped)}/{len(node_electrodes)} electrodes skipped: {skipped}")
    if failures:
        print(f"❌ {len(failures)}/{len(node_electrodes)} electrodes failed: {failures}")
    
    # Final assertion to make the test pass/fail
    assert not failures, f"Data mismatch found for electrodes: {failures}"
    
    # If we got here, all checks passed!
    print("\n✅ VERIFICATION SUCCESSFUL: All checked electrodes have consistent data across all stages")


def test_specific_electrode_consistency(reservoir_dataset):
    """
    Test that a specific electrode '10' in feature selection points to the same data as '10_V[V]' column
    in the raw dataframe.
    """
    # Get the raw dataframe directly from the dataset
    raw_df = reservoir_dataset.dataframe
    
    # Check if the target column exists
    target_electrode = '10'
    target_column = f'{target_electrode}_V[V]'
    
    if target_column not in raw_df.columns:
        all_voltage_columns = [col for col in raw_df.columns if '_V[V]' in col]
        print(f"Available voltage columns: {all_voltage_columns}")
        pytest.skip(f"Target column {target_column} not found in raw dataframe")
    
    # Get raw values from the dataframe
    raw_values = raw_df[target_column].values[:20]  # Get first 20 values
    print(f"Raw values from DataFrame['{target_column}'] (first 20):\n{raw_values}")
    
    # Get electrode information
    electrodes_info = reservoir_dataset.summary()
    node_electrodes = electrodes_info['node_electrodes']
    
    # Verify target electrode is in node_electrodes
    if target_electrode not in node_electrodes:
        pytest.skip(f"Target electrode {target_electrode} not found in node_electrodes")
    
    # Get the node output matrix
    nodes_output = reservoir_dataset.get_node_voltages()
    
    # Get the index of target electrode in node_electrodes
    target_idx = node_electrodes.index(target_electrode)
    
    # Extract the values for this electrode from the node_output matrix
    node_values = nodes_output[:20, target_idx]
    print(f"Node values from nodes_output[:, {target_idx}] (first 20):\n{node_values}")
    
    # Verify raw dataframe values match node output values
    assert np.allclose(raw_values, node_values, rtol=1e-5, atol=1e-5), \
        f"Data mismatch between raw dataframe and node output for electrode {target_electrode}"
    
    # Now run feature selection
    input_voltages = reservoir_dataset.get_input_voltages()
    primary_input_electrode = electrodes_info['input_electrodes'][0]
    input_signal = input_voltages[primary_input_electrode]
    
    # Dummy target for feature selection
    y = np.sin(input_signal)
    
    # Run feature selection
    feature_selector = FeatureSelector(random_state=42)
    X_selected, selected_indices, selected_names = feature_selector.select_features(
        X=nodes_output,
        y=y,
        electrode_names=node_electrodes,
        method='pca',
        num_features='all'
    )
    
    print(f"Selected electrode names: {selected_names}")
    
    # Verify electrode '10' is in the selected features
    if target_electrode not in selected_names:
        pytest.skip(f"Target electrode {target_electrode} was not selected by feature selection")
    
    # Get the index of target electrode in selected_names
    selected_idx = selected_names.index(target_electrode)
    
    # Get values from selected features
    selected_values = X_selected[:20, selected_idx]
    print(f"Selected feature values (first 20):\n{selected_values}")
    
    # Final verification that raw dataframe values match selected feature values
    assert np.allclose(raw_values, selected_values, rtol=1e-5, atol=1e-5), \
        f"Data mismatch between raw dataframe and selected feature values for electrode {target_electrode}"
    
    # Get values directly from individual electrode method
    individual_values = reservoir_dataset.get_node_voltage(target_electrode)[:20]
    print(f"Individual electrode values (first 20):\n{individual_values}")
    
    # Verify direct access method matches raw values
    assert np.allclose(raw_values, individual_values, rtol=1e-5, atol=1e-5), \
        f"Data mismatch between raw dataframe and individual electrode values for electrode {target_electrode}"
    
    print("\n✅ VERIFICATION SUCCESSFUL: Electrode '10' data is consistent across all access methods")


def test_all_access_methods_consistent(reservoir_dataset):
    """
    Test that all methods of accessing electrode data return consistent results.
    """
    # Get electrode information
    electrodes_info = reservoir_dataset.summary()
    node_electrodes = electrodes_info['node_electrodes']
    
    # Skip if no electrodes found
    if not node_electrodes:
        pytest.skip("No node electrodes found")
    
    # Get the node output matrix
    nodes_output = reservoir_dataset.get_node_voltages()
    
    # Track results
    failures = []
    successes = []
    
    # Check each electrode
    for electrode in node_electrodes:
        # Get electrode index in node_electrodes list
        node_idx = node_electrodes.index(electrode)
        
        # Get values from nodes_output matrix
        matrix_values = nodes_output[:10, node_idx]
        
        # Get values directly using get_node_voltage method
        try:
            direct_values = reservoir_dataset.get_node_voltage(electrode)[:10]
            
            # Verify both methods return the same data
            assert np.allclose(matrix_values, direct_values, rtol=1e-5, atol=1e-5)
            
            # Success!
            successes.append(electrode)
            
        except AssertionError as e:
            failures.append(electrode)
            print(f"❌ Electrode {electrode}: Data mismatch detected")
            print(f"  Matrix values: {matrix_values}")
            print(f"  Direct values: {direct_values}")
    
    # Final report
    print(f"\n=== ELECTRODE ACCESS METHODS CONSISTENCY RESULTS ===")
    print(f"✅ {len(successes)}/{len(node_electrodes)} electrodes verified consistent")
    if failures:
        print(f"❌ {len(failures)}/{len(node_electrodes)} electrodes failed: {failures}")
    
    # Final assertion to make the test pass/fail
    assert not failures, f"Data mismatch found for electrodes: {failures}"
    
    print("\n✅ VERIFICATION SUCCESSFUL: All electrode access methods return consistent data") 