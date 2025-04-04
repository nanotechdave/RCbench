import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from rcbench.measurements.loader import MeasurementLoader
from rcbench.measurements.parser import MeasurementParser
from rcbench.tasks.memorycapacity import MemoryCapacityEvaluator
from rcbench.tasks.featureselector import FeatureSelector
from rcbench.measurements.dataset import ReservoirDataset


@pytest.fixture
def reservoir_dataset():
    """Load measurement data for testing using the ReservoirDataset class."""
    BASE_DIR = Path(__file__).resolve().parent.parent
    filename = "074_INRiMARC_NWN_Pad129M_gridSE_MemoryCapacity_2024_04_02.txt"
    measurement_file = BASE_DIR / "tests" / "test_files" / filename
    
    # Load the data directly using the ReservoirDataset class
    dataset = ReservoirDataset(measurement_file)
    return dataset


@pytest.fixture
def measurement_data():
    """Legacy fixture for backward compatibility - will be deprecated."""
    BASE_DIR = Path(__file__).resolve().parent.parent
    filename = "074_INRiMARC_NWN_Pad129M_gridSE_MemoryCapacity_2024_04_02.txt"
    measurement_file = BASE_DIR / "tests" / "test_files" / filename
    
    loader = MeasurementLoader(measurement_file)
    dataset = loader.get_dataset()
    return dataset


@pytest.fixture
def parsed_data(measurement_data):
    """Legacy fixture for backward compatibility - will be deprecated."""
    # Create a dictionary to hold the parser data using the updated static methods
    parser_data = {
        'dataframe': measurement_data.dataframe,
        'electrodes': MeasurementParser.identify_electrodes(measurement_data.dataframe)
    }
    
    # Add methods to get data from the dataset
    parser_data['get_input_voltages'] = lambda: MeasurementParser.get_input_voltages(
        measurement_data.dataframe, parser_data['electrodes']['input_electrodes'])
    
    parser_data['get_node_voltages'] = lambda: MeasurementParser.get_node_voltages(
        measurement_data.dataframe, parser_data['electrodes']['node_electrodes'])
    
    parser_data['summary'] = lambda: parser_data['electrodes']
    
    return parser_data


def test_electrode_names_consistency(parsed_data):
    """Test that electrode names are consistent between parser and feature selection."""
    # Get data from parser
    electrodes_info = parsed_data['summary']()
    node_electrodes = electrodes_info['node_electrodes']
    
    # Check that node_electrodes is not empty
    assert len(node_electrodes) > 0, "No node electrodes found in parser output"
    
    # Print node electrodes for debug purposes
    print(f"Node electrodes from parser: {node_electrodes}")
    
    # Get input and node outputs
    input_voltages = parsed_data['get_input_voltages']()
    nodes_output = parsed_data['get_node_voltages']()
    
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
    electrodes_info = parsed_data['summary']()
    node_electrodes = electrodes_info['node_electrodes']
    
    # Get input and node outputs
    input_voltages = parsed_data['get_input_voltages']()
    nodes_output = parsed_data['get_node_voltages']()
    
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
    electrodes_info = parsed_data['summary']()
    node_electrodes = electrodes_info['node_electrodes']
    
    # Get input and node outputs
    input_voltages = parsed_data['get_input_voltages']()
    nodes_output = parsed_data['get_node_voltages']()
    
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
    electrodes_info = parsed_data['summary']()
    node_electrodes = electrodes_info['node_electrodes']
    
    # Get input and node outputs
    input_voltages = parsed_data['get_input_voltages']()
    nodes_output = parsed_data['get_node_voltages']()
    
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
    
    # Load raw data directly from file
    raw_df = pd.read_csv(measurement_file, sep='\s+', engine='python')
    
    # Get data from target column
    target_electrode = '10'
    target_column = f'{target_electrode}_V[V]'
    
    if target_column not in raw_df.columns:
        print(f"Available columns: {[col for col in raw_df.columns if '_V[V]' in col]}")
        pytest.skip(f"Target column {target_column} not found in raw DataFrame")
    
    # Get the raw target data (our baseline for comparison)
    target = raw_df[target_column].values
    print(f"Raw target data (first 10): {target[:10]}")

    # Load and parse the data using the standard pipeline
    loader = MeasurementLoader(measurement_file)
    dataset = loader.get_dataset()
    
    # Get electrode information using the parser
    electrodes_info = MeasurementParser.identify_electrodes(dataset.dataframe)
    node_electrodes = electrodes_info['node_electrodes']
    
    print(f"Node electrodes: {node_electrodes}")
    
    # Check if target electrode is in node_electrodes
    if target_electrode not in node_electrodes:
        # If not, use any electrode that is available
        target_electrode = node_electrodes[0] if node_electrodes else None
        if not target_electrode:
            pytest.skip("No node electrodes available for testing")
            
    print(f"Target electrode: {target_electrode}")
    
    # Get input and node outputs using static methods
    input_voltages = MeasurementParser.get_input_voltages(dataset.dataframe, electrodes_info['input_electrodes'])
    nodes_output = MeasurementParser.get_node_voltages(dataset.dataframe, node_electrodes)
    
    # Get individual node voltage arrays directly from the dataframe
    node_voltages = {}
    for node in node_electrodes:
        col = f'{node}_V[V]'
        if col in dataset.dataframe.columns:
            node_voltages[node] = dataset.dataframe[col].values
        else:
            print(f"Could not get voltage for node {node}: Column {col} not found")
    
    # Get index of target electrode in node_electrodes list
    target_idx = node_electrodes.index(target_electrode)
    
    # Get values directly from the dataframe for our target electrode
    target_voltage = node_voltages.get(target_electrode, None)
    if target_voltage is None:
        pytest.skip(f"Could not get voltage data for electrode {target_electrode}")
    
    # Get values from nodes_output (matrix of all node readings)
    nodes_values = nodes_output[:10, target_idx]
    
    # Get raw values from the direct node reading
    raw_values = target_voltage[:10]
    
    print(f"Raw values from dataframe (first 10): {raw_values}")
    print(f"Node values from nodes_output matrix (first 10): {nodes_values}")
    print(f"Original target values from raw file (first 10): {target[:10]}")
    
    # Verify that raw data from the file matches node output for the target electrode
    assert np.allclose(target[:10], nodes_values, rtol=1e-5, atol=1e-5), \
        f"Data mismatch between raw file data and nodes_output for electrode {target_electrode}"
    
    # Verify that raw values from dataframe match nodes_output
    assert np.allclose(raw_values, nodes_values, rtol=1e-5, atol=1e-5), \
        f"Data mismatch between dataframe and nodes_output for electrode {target_electrode}"
    
    # Now run feature selection
    y = np.sin(input_voltages[electrodes_info['input_electrodes'][0]])  # Use sine of input as target
    
    # Run feature selection
    feature_selector = FeatureSelector(random_state=42)
    X_selected, selected_indices, selected_names = feature_selector.select_features(
        X=nodes_output,
        y=y,
        electrode_names=node_electrodes,
        method='pca',
        num_features='all'
    )
    
    print(f"Selected indices: {selected_indices}")
    print(f"Selected names: {selected_names}")
    
    # Check if target electrode was selected
    if target_electrode in selected_names:
        # Get the index of the target electrode in the selected features
        selected_idx = selected_names.index(target_electrode)
        
        # Get values from selected features
        selected_values = X_selected[:10, selected_idx]
        
        print(f"Selected feature values (first 10): {selected_values}")
        
        # Verify raw file data matches selected values
        assert np.allclose(target[:10], selected_values, rtol=1e-5, atol=1e-5), \
            f"Data mismatch between raw file data and selected feature values for electrode {target_electrode}"
        
        # Verify that raw values match selected feature values
        assert np.allclose(raw_values, selected_values, rtol=1e-5, atol=1e-5), \
            f"Data mismatch between dataframe and selected feature values for electrode {target_electrode}"
        
        # Double check that nodes_output matches selected values
        assert np.allclose(nodes_values, selected_values, rtol=1e-5, atol=1e-5), \
            f"Data mismatch between nodes_output and selected feature values for electrode {target_electrode}"
            
        print(f"\n✅ VERIFIED: Original raw data for '{target_column}' matches node voltage for electrode '{target_electrode}' through all processing stages")
    else:
        # If electrode 10 wasn't selected, check any electrode that was selected
        if selected_names and selected_indices:
            test_electrode = selected_names[0]
            test_idx = selected_indices[0]
            node_idx = node_electrodes.index(test_electrode)
            
            # Get the raw data for this test electrode
            test_column = f'{test_electrode}_V[V]'
            test_target = raw_df[test_column].values if test_column in raw_df.columns else None
            
            if test_target is not None:
                # Get raw values for this electrode
                test_voltage = node_voltages.get(test_electrode, None)
                if test_voltage is not None:
                    test_raw_values = test_voltage[:10]
                    test_nodes_values = nodes_output[:10, node_idx]
                    test_selected_values = X_selected[:10, 0]  # First selected feature
                    
                    print(f"\nTesting with alternative electrode {test_electrode}:")
                    print(f"Raw file values (first 10): {test_target[:10]}")
                    print(f"Parser dataframe values (first 10): {test_raw_values}")
                    print(f"Nodes values (first 10): {test_nodes_values}")
                    print(f"Selected values (first 10): {test_selected_values}")
                    
                    # Verify raw file data matches nodes_output 
                    assert np.allclose(test_target[:10], test_nodes_values, rtol=1e-5, atol=1e-5), \
                        f"Data mismatch between raw file data and nodes_output for electrode {test_electrode}"
                    
                    # Verify raw parser dataframe values match nodes_output
                    assert np.allclose(test_raw_values, test_nodes_values, rtol=1e-5, atol=1e-5), \
                        f"Data mismatch between parser dataframe and nodes_output for electrode {test_electrode}"
                    
                    # Verify nodes_output matches selected values
                    assert np.allclose(test_nodes_values, test_selected_values, rtol=1e-5, atol=1e-5), \
                        f"Data mismatch between nodes_output and selected values for electrode {test_electrode}"
                        
                    print(f"\n✅ VERIFIED: Original raw data for '{test_column}' matches node voltage for electrode '{test_electrode}' through all processing stages")
            else:
                pytest.skip(f"Could not get raw data for alternative electrode {test_electrode}")
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
    
    # Get electrode information using the static method
    electrodes_info = MeasurementParser.identify_electrodes(dataset.dataframe)
    node_electrodes = electrodes_info['node_electrodes']
    
    # Verify target electrode is in node_electrodes
    if target_electrode not in node_electrodes:
        pytest.skip(f"Target electrode {target_electrode} not found in node_electrodes")
    
    # Get the node output matrix using the static method
    nodes_output = MeasurementParser.get_node_voltages(dataset.dataframe, node_electrodes)
    
    # Get the index of target electrode in node_electrodes
    target_idx = node_electrodes.index(target_electrode)
    
    # Extract the values for this electrode from the node_output matrix
    node_values = nodes_output[:20, target_idx]
    print(f"Node values from nodes_output[:, {target_idx}] (first 20):\n{node_values}")
    
    # Verify raw dataframe values match parser's node output values
    assert np.allclose(raw_values, node_values, rtol=1e-5, atol=1e-5), \
        f"Data mismatch between raw dataframe and parser's node output for electrode {target_electrode}"
    
    # Now run feature selection
    input_voltages = MeasurementParser.get_input_voltages(dataset.dataframe, electrodes_info['input_electrodes'])
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
    
    print(f"Looking for test file at: {measurement_file}")
    print(f"File exists: {measurement_file.exists()}")
    
    # Load the data directly using the ReservoirDataset class
    dataset = ReservoirDataset(measurement_file)
    
    # Get the raw dataframe directly from the dataset
    raw_df = dataset.dataframe
    print(f"Raw dataframe shape: {raw_df.shape}")
    print(f"Raw dataframe columns (first 5): {list(raw_df.columns)[:5]}")
    
    # Get electrode information
    electrodes_info = dataset.summary()
    node_electrodes = electrodes_info['node_electrodes']
    
    # Skip if no electrodes found
    if not node_electrodes:
        print("ERROR: No node electrodes found in dataset output")
        pytest.skip("No node electrodes found")
    
    print(f"Testing data consistency for {len(node_electrodes)} electrodes: {node_electrodes}")
    
    # Get the node output matrix
    nodes_output = dataset.get_node_voltages()
    print(f"Node output matrix shape: {nodes_output.shape}")
    
    # Setup for feature selection
    input_voltages = dataset.get_input_voltages()
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
            print(f"Available columns (first 10): {list(raw_df.columns)[:10]}")
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
            print(f"Selected names: {selected_names}")
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


def test_measurement_file_exists():
    """
    Simple test to verify the test data file can be found.
    This helps debug issues with VSCode test discovery.
    """
    BASE_DIR = Path(__file__).resolve().parent.parent
    filename = "074_INRiMARC_NWN_Pad129M_gridSE_MemoryCapacity_2024_04_02.txt"
    measurement_file = BASE_DIR / "tests" / "test_files" / filename
    
    print(f"Test file path: {measurement_file}")
    print(f"Current working directory: {Path.cwd()}")
    print(f"__file__ is: {__file__}")
    print(f"BASE_DIR is: {BASE_DIR}")
    
    assert measurement_file.exists(), f"Test file not found at {measurement_file}"


def get_node_voltage(self, node: str) -> np.ndarray:
    """Get voltage data for a specific node."""
    if node not in self.node_electrodes:
        raise ValueError(f"Node {node} not found in node_electrodes")
    col = f'{node}_V[V]'
    return self.dataframe[col].values 