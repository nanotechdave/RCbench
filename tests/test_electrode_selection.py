import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from rcbench.measurements.loader import MeasurementLoader
from rcbench.measurements.parser import MeasurementParser
from rcbench.tasks.memorycapacity import MemoryCapacityEvaluator
from rcbench.tasks.featureselector import FeatureSelector


@pytest.fixture
def measurement_data():
    """Load measurement data for testing."""
    BASE_DIR = Path(__file__).resolve().parent.parent
    filename = "074_INRiMARC_NWN_Pad129M_gridSE_MemoryCapacity_2024_04_02.txt"
    measurement_file = BASE_DIR / "tests" / "test_files" / filename
    
    loader = MeasurementLoader(measurement_file)
    dataset = loader.get_dataset()
    return dataset


@pytest.fixture
def parsed_data(measurement_data):
    """Parse measurement data."""
    parser = MeasurementParser(measurement_data)
    return parser


def test_electrode_names_consistency(parsed_data):
    """Test that electrode names are consistent between parser and feature selection."""
    # Get data from parser
    electrodes_info = parsed_data.summary()
    node_electrodes = electrodes_info['node_electrodes']
    
    # Check that node_electrodes is not empty
    assert len(node_electrodes) > 0, "No node electrodes found in parser output"
    
    # Print node electrodes for debug purposes
    print(f"Node electrodes from parser: {node_electrodes}")
    
    # Get input and node outputs
    input_voltages = parsed_data.get_input_voltages()
    nodes_output = parsed_data.get_node_voltages()
    
    # Get input signal
    primary_input_electrode = electrodes_info['input_electrodes'][0]
    input_signal = input_voltages[primary_input_electrode]
    
    # Create dummy target for testing
    y = np.sin(input_signal)
    
    # Initialize feature selector
    feature_selector = FeatureSelector(random_state=42)
    
    # Perform feature selection
    X_selected, selected_indices, selected_names = feature_selector.select_features(
        X=nodes_output,
        y=y,
        electrode_names=node_electrodes,
        method='pca',
        num_features='all'
    )
    
    # Print feature selection results for debugging
    print(f"Selected indices: {selected_indices}")
    print(f"Selected names: {selected_names}")
    
    # Verify all selected electrodes are in the original node_electrodes list
    for name in selected_names:
        assert name in node_electrodes, f"Selected electrode {name} not found in node_electrodes"
    
    # Create a mapping of indices to electrode names
    electrode_map = {i: name for i, name in enumerate(node_electrodes)}
    
    # Verify indices match electrode names
    for idx, name in zip(selected_indices, selected_names):
        assert electrode_map[idx] == name, f"Mismatch: index {idx} maps to {electrode_map[idx]}, not {name}"


def test_memorycapacity_evaluator_electrode_selection(parsed_data):
    """Test that MemoryCapacityEvaluator selects the correct electrodes."""
    # Get data from parser
    electrodes_info = parsed_data.summary()
    node_electrodes = electrodes_info['node_electrodes']
    
    # Get input and node outputs
    input_voltages = parsed_data.get_input_voltages()
    nodes_output = parsed_data.get_node_voltages()
    
    # Get input signal
    primary_input_electrode = electrodes_info['input_electrodes'][0]
    input_signal = input_voltages[primary_input_electrode]
    
    # Create MC evaluator
    evaluator = MemoryCapacityEvaluator(
        input_signal,
        nodes_output,
        max_delay=5,  # Use a small value for testing
        random_state=42,
        electrode_names=node_electrodes
    )
    
    # Run memory capacity calculation
    results = evaluator.calculate_total_memory_capacity(
        feature_selection_method='pca',
        num_features=len(node_electrodes),  # Select all electrodes
        regression_alpha=0.1,
        train_ratio=0.8
    )
    
    # Check that selected feature names match a subset of node_electrodes
    assert all(name in node_electrodes for name in evaluator.selected_feature_names), \
        "Selected feature names don't match node electrodes"
    
    # Check that number of selected features matches num_features
    assert len(evaluator.selected_feature_names) == len(node_electrodes), \
        f"Expected {len(node_electrodes)} selected features, got {len(evaluator.selected_feature_names)}"
    
    # Check that indices and names correspond
    for idx, name in zip(evaluator.selected_features, evaluator.selected_feature_names):
        assert node_electrodes[idx] == name, \
            f"Selected index {idx} should map to {node_electrodes[idx]}, not {name}"


def test_importance_values_match_electrodes(parsed_data):
    """Test that feature importance values match the correct electrodes."""
    # Get data from parser
    electrodes_info = parsed_data.summary()
    node_electrodes = electrodes_info['node_electrodes']
    
    # Get input and node outputs
    input_voltages = parsed_data.get_input_voltages()
    nodes_output = parsed_data.get_node_voltages()
    
    # Get input signal
    primary_input_electrode = electrodes_info['input_electrodes'][0]
    input_signal = input_voltages[primary_input_electrode]
    
    # Create MC evaluator
    evaluator = MemoryCapacityEvaluator(
        input_signal,
        nodes_output,
        max_delay=5,  # Use a small value for testing
        random_state=42,
        electrode_names=node_electrodes
    )
    
    # Run memory capacity calculation
    results = evaluator.calculate_total_memory_capacity(
        feature_selection_method='pca',
        num_features=len(node_electrodes),  # Select all electrodes
        regression_alpha=0.1,
        train_ratio=0.8
    )
    
    # Get feature importance
    feature_importance = evaluator.feature_selector.get_feature_importance()
    
    # Check that feature importance Series has the correct index
    assert all(name in feature_importance.index for name in node_electrodes), \
        "Not all electrode names found in feature importance index"
    
    # Get the selected electrode names and their importance scores
    selected_names = evaluator.selected_feature_names
    importance_scores = np.array([feature_importance[name] for name in selected_names])
    
    # Verify scores are in descending order (highest importance first)
    assert np.all(np.diff(importance_scores) <= 0), \
        "Importance scores are not in descending order"
    
    # Create a DataFrame with electrode names and importance scores for debugging
    importance_df = pd.DataFrame({
        'electrode': selected_names,
        'importance': importance_scores
    })
    print("\nElectrode importance scores:")
    print(importance_df)
    
    # Verify consistency by running a second time
    second_evaluator = MemoryCapacityEvaluator(
        input_signal,
        nodes_output,
        max_delay=5,
        random_state=42,
        electrode_names=node_electrodes
    )
    
    second_evaluator.calculate_total_memory_capacity(
        feature_selection_method='pca',
        num_features=len(node_electrodes),
        regression_alpha=0.1,
        train_ratio=0.8
    )
    
    # Compare selected electrodes to ensure consistency
    assert evaluator.selected_feature_names == second_evaluator.selected_feature_names, \
        "Selected electrodes are not consistent between runs"


def test_selected_columns_match_electrode_data(parsed_data):
    """Test that selected columns match the actual electrode data."""
    # Get data from parser
    electrodes_info = parsed_data.summary()
    node_electrodes = electrodes_info['node_electrodes']
    
    # Get input and node outputs
    input_voltages = parsed_data.get_input_voltages()
    nodes_output = parsed_data.get_node_voltages()
    
    # Get input signal
    primary_input_electrode = electrodes_info['input_electrodes'][0]
    input_signal = input_voltages[primary_input_electrode]
    
    # Create dummy target for testing
    y = np.sin(input_signal)
    
    # Initialize feature selector
    feature_selector = FeatureSelector(random_state=42)
    
    # Perform feature selection
    X_selected, selected_indices, selected_names = feature_selector.select_features(
        X=nodes_output,
        y=y,
        electrode_names=node_electrodes,
        method='pca',
        num_features=5  # Just select a few top electrodes
    )
    
    # Print for debugging
    print(f"Original electrode names: {node_electrodes}")
    print(f"Selected indices: {selected_indices}")
    print(f"Selected names: {selected_names}")
    
    # Create a DataFrame with the original data
    full_df = pd.DataFrame(nodes_output, columns=node_electrodes)
    
    # Create a DataFrame with just the selected columns using indices
    selected_df = pd.DataFrame(X_selected, columns=selected_names)
    
    # Verify data matches by comparing sample values
    for i, col_name in enumerate(selected_names):
        col_idx = selected_indices[i]
        
        # Get first 5 values from both DataFrames
        orig_values = full_df[col_name].values[:5]
        selected_values = selected_df[col_name].values[:5]
        
        # Print for debugging
        print(f"\nElectrode {col_name} (index {col_idx}):")
        print(f"Original data (first 5): {orig_values}")
        print(f"Selected data (first 5): {selected_values}")
        
        # Check that values match
        assert np.allclose(orig_values, selected_values), \
            f"Data mismatch for electrode {col_name} (index {col_idx})"


def test_raw_measurement_data_matches_selected_electrodes():
    """Test that electrode data matches the raw measurement file columns."""
    # Load the measurement data
    BASE_DIR = Path(__file__).resolve().parent.parent
    filename = "074_INRiMARC_NWN_Pad129M_gridSE_MemoryCapacity_2024_04_02.txt"
    measurement_file = BASE_DIR / "tests" / "test_files" / filename
    
    # Load and parse the data
    loader = MeasurementLoader(measurement_file)
    dataset = loader.get_dataset()
    parser = MeasurementParser(dataset)
    
    # Get electrode information from parser
    electrodes_info = parser.summary()
    node_electrodes = electrodes_info['node_electrodes']
    
    print(f"Node electrodes: {node_electrodes}")
    
    # Get the specific electrode we want to check (electrode '10')
    target_electrode = '10'
    
    # Check if target electrode is in node_electrodes
    if target_electrode not in node_electrodes:
        # If not, use any electrode that is available
        target_electrode = node_electrodes[0] if node_electrodes else None
        if not target_electrode:
            pytest.skip("No node electrodes available for testing")
            
    print(f"Target electrode: {target_electrode}")
    
    # Get input and node outputs from parser
    input_voltages = parser.get_input_voltages()
    nodes_output = parser.get_node_voltages()
    
    # Get individual node voltage arrays directly
    # This uses parser's direct method to get a specific node's voltage
    node_voltages = {}
    for node in node_electrodes:
        try:
            node_voltages[node] = parser.get_node_voltage(node)
        except Exception as e:
            print(f"Could not get voltage for node {node}: {e}")
    
    # Get index of target electrode in node_electrodes list
    target_idx = node_electrodes.index(target_electrode)
    
    # Get values directly from the parser for our target electrode
    target_voltage = node_voltages.get(target_electrode, None)
    if target_voltage is None:
        pytest.skip(f"Could not get voltage data for electrode {target_electrode}")
    
    # Get values from nodes_output (matrix of all node readings)
    nodes_values = nodes_output[:10, target_idx]
    
    # Get raw values from the direct node reading
    raw_values = target_voltage[:10]
    
    print(f"Raw values from direct access (first 10): {raw_values}")
    print(f"Node values from nodes_output matrix (first 10): {nodes_values}")
    
    # Verify the values match between direct node access and nodes_output matrix
    assert np.allclose(raw_values, nodes_values, rtol=1e-5, atol=1e-5), \
        f"Data mismatch between raw node voltage and nodes_output for electrode {target_electrode}"
    
    # Now perform feature selection and verify the selected electrode data matches
    # Get input signal
    primary_input_electrode = electrodes_info['input_electrodes'][0]
    input_signal = input_voltages[primary_input_electrode]
    
    # Create a target for feature selection
    y = np.sin(input_signal)
    
    # Run feature selection
    feature_selector = FeatureSelector(random_state=42)
    
    # Run feature selection to get selected indices and names
    X_selected, selected_indices, selected_names = feature_selector.select_features(
        X=nodes_output,
        y=y,
        electrode_names=node_electrodes,
        method='pca',
        num_features='all'
    )
    
    print(f"Selected electrode names: {selected_names}")
    
    # Check if target electrode was selected
    if target_electrode in selected_names:
        # Get the index of the target electrode in the selected features
        selected_idx = selected_names.index(target_electrode)
        
        # Get values from selected features
        selected_values = X_selected[:10, selected_idx]
        
        print(f"Selected feature values (first 10): {selected_values}")
        
        # Verify both raw and selected values match
        assert np.allclose(raw_values, selected_values, rtol=1e-5, atol=1e-5), \
            f"Data mismatch between raw values and selected feature values for electrode {target_electrode}"
        
        # Double check that selected values match nodes_output
        assert np.allclose(nodes_values, selected_values, rtol=1e-5, atol=1e-5), \
            f"Data mismatch between nodes_output and selected feature values for electrode {target_electrode}"
    else:
        # If electrode 10 wasn't selected, check any electrode that was selected
        if selected_names and selected_indices:
            test_electrode = selected_names[0]
            test_idx = selected_indices[0]
            node_idx = node_electrodes.index(test_electrode)
            
            # Get raw values for this electrode
            test_voltage = node_voltages.get(test_electrode, None)
            if test_voltage is not None:
                test_raw_values = test_voltage[:10]
                test_nodes_values = nodes_output[:10, node_idx]
                test_selected_values = X_selected[:10, 0]  # First selected feature
                
                print(f"\nTesting with alternative electrode {test_electrode}:")
                print(f"Raw values (first 10): {test_raw_values}")
                print(f"Nodes values (first 10): {test_nodes_values}")
                print(f"Selected values (first 10): {test_selected_values}")
                
                # Verify all values match
                assert np.allclose(test_raw_values, test_nodes_values, rtol=1e-5, atol=1e-5), \
                    f"Data mismatch between raw node voltage and nodes_output for electrode {test_electrode}"
                
                assert np.allclose(test_nodes_values, test_selected_values, rtol=1e-5, atol=1e-5), \
                    f"Data mismatch between nodes_output and selected values for electrode {test_electrode}"
        else:
            pytest.skip("No electrodes were selected during feature selection")


def test_electrode_data_consistency_with_raw_dataframe():
    """
    Test that electrode '10' in feature selection points to the same data as '10_V[V]' column
    in the raw dataframe.
    """
    # Load the measurement data
    BASE_DIR = Path(__file__).resolve().parent.parent
    filename = "074_INRiMARC_NWN_Pad129M_gridSE_MemoryCapacity_2024_04_02.txt"
    measurement_file = BASE_DIR / "tests" / "test_files" / filename
    
    # Load the data
    loader = MeasurementLoader(measurement_file)
    dataset = loader.get_dataset()
    
    # Get the raw dataframe directly from the dataset
    raw_df = dataset.dataframe
    
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
    
    # Now parse the data normally
    parser = MeasurementParser(dataset)
    electrodes_info = parser.summary()
    node_electrodes = electrodes_info['node_electrodes']
    
    # Verify target electrode is in node_electrodes
    if target_electrode not in node_electrodes:
        pytest.skip(f"Target electrode {target_electrode} not found in node_electrodes")
    
    # Get the node output matrix
    nodes_output = parser.get_node_voltages()
    
    # Get the index of target electrode in node_electrodes
    target_idx = node_electrodes.index(target_electrode)
    
    # Extract the values for this electrode from the node_output matrix
    node_values = nodes_output[:20, target_idx]
    print(f"Node values from nodes_output[:, {target_idx}] (first 20):\n{node_values}")
    
    # Verify raw dataframe values match parser's node output values
    assert np.allclose(raw_values, node_values, rtol=1e-5, atol=1e-5), \
        f"Data mismatch between raw dataframe and parser's node output for electrode {target_electrode}"
    
    # Now run feature selection
    input_voltages = parser.get_input_voltages()
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
    
    print("\n✅ VERIFICATION SUCCESSFUL: Electrode '10' in feature selection points to the same data as '10_V[V]' column in raw dataframe")


def test_all_electrodes_data_consistency():
    """
    Test that all electrodes in feature selection point to the same data as their
    corresponding columns in the raw dataframe.
    """
    # Load the measurement data
    BASE_DIR = Path(__file__).resolve().parent.parent
    filename = "074_INRiMARC_NWN_Pad129M_gridSE_MemoryCapacity_2024_04_02.txt"
    measurement_file = BASE_DIR / "tests" / "test_files" / filename
    
    # Load the data
    loader = MeasurementLoader(measurement_file)
    dataset = loader.get_dataset()
    
    # Get the raw dataframe directly from the dataset
    raw_df = dataset.dataframe
    
    # Parse the data
    parser = MeasurementParser(dataset)
    electrodes_info = parser.summary()
    node_electrodes = electrodes_info['node_electrodes']
    
    # Skip if no electrodes found
    if not node_electrodes:
        pytest.skip("No node electrodes found")
    
    print(f"Testing data consistency for {len(node_electrodes)} electrodes: {node_electrodes}")
    
    # Get the node output matrix
    nodes_output = parser.get_node_voltages()
    
    # Setup for feature selection
    input_voltages = parser.get_input_voltages()
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
            # Verify raw dataframe values match parser's node output values
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