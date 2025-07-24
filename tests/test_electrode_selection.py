import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from rcbench.measurements.loader import MeasurementLoader
from rcbench.measurements.parser import MeasurementParser
from rcbench.tasks.memorycapacity import MemoryCapacityEvaluator
from rcbench.tasks.featureselector import FeatureSelector
from rcbench.measurements.dataset import ElecResDataset


@pytest.fixture
def reservoir_dataset():
    """Load measurement data for testing using the ElecResDataset class."""
    BASE_DIR = Path(__file__).resolve().parent.parent
    filename = "074_INRiMARC_NWN_Pad129M_gridSE_MemoryCapacity_2024_04_02.txt"
    measurement_file = BASE_DIR / "tests" / "test_files" / filename
    
    # Load the data directly using the ElecResDataset class
    dataset = ElecResDataset(measurement_file)
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
        'nodes': MeasurementParser.identify_nodes(measurement_data.dataframe)
    }
    
    # Add methods to get data from the dataset
    parser_data['get_input_voltages'] = lambda: MeasurementParser.get_input_voltages(
        measurement_data.dataframe, parser_data['nodes']['input_nodes'])
    
    parser_data['get_node_voltages'] = lambda: MeasurementParser.get_node_voltages(
        measurement_data.dataframe, parser_data['nodes']['nodes'])
    
    parser_data['summary'] = lambda: parser_data['nodes']
    
    return parser_data


def test_node_names_consistency(parsed_data):
    """Test that node names are consistent between parser and feature selection."""
    # Get data from parser
    nodes_info = parsed_data['summary']()
    nodes = nodes_info['nodes']
    
    # Check that nodes is not empty
    assert len(nodes) > 0, "No computation nodes found in parser output"
    
    # Print nodes for debug purposes
    print(f"Computation nodes from parser: {nodes}")
    
    # Get input and node outputs
    input_voltages = parsed_data['get_input_voltages']()
    nodes_output = parsed_data['get_node_voltages']()
    
    # Get input signal
    primary_input_node = nodes_info['input_nodes'][0]
    input_signal = input_voltages[primary_input_node]
    
    # Create dummy target for testing
    y = np.sin(input_signal)
    
    # Initialize feature selector
    feature_selector = FeatureSelector(random_state=42)
    
    # Perform feature selection
    X_selected, selected_indices, selected_names = feature_selector.select_features(
        X=nodes_output,
        y=y,
        node_names=nodes,
        method='pca',
        num_features='all'
    )
    
    # Print feature selection results for debugging
    print(f"Selected indices: {selected_indices}")
    print(f"Selected names: {selected_names}")
    
    # Verify all selected nodes are in the original nodes list
    for name in selected_names:
        assert name in nodes, f"Selected node {name} not found in nodes"
    
    # Create a mapping of indices to node names
    node_map = {i: name for i, name in enumerate(nodes)}
    
    # Verify indices match node names
    for idx, name in zip(selected_indices, selected_names):
        assert node_map[idx] == name, f"Mismatch: index {idx} maps to {node_map[idx]}, not {name}"


def test_memorycapacity_evaluator_node_selection(parsed_data):
    """Test that MemoryCapacityEvaluator selects the correct nodes."""
    # Get data from parser
    nodes_info = parsed_data['summary']()
    nodes = nodes_info['nodes']

    # Get input and node outputs
    input_voltages = parsed_data['get_input_voltages']()
    nodes_output = parsed_data['get_node_voltages']()

    # Get input signal
    primary_input_node = nodes_info['input_nodes'][0]
    input_signal = input_voltages[primary_input_node]
    
    # Create MC evaluator
    evaluator = MemoryCapacityEvaluator(
        input_signal,
        nodes_output,
        max_delay=5,  # Use a small value for testing
        random_state=42,
        node_names=nodes
    )
    
    # Run memory capacity calculation
    results = evaluator.calculate_total_memory_capacity(
        feature_selection_method='pca',
        num_features=len(nodes),  # Select all nodes
        regression_alpha=0.1,
        train_ratio=0.8
    )
    
    # Check that selected feature names match a subset of nodes
    assert all(name in nodes for name in evaluator.selected_feature_names), \
        "Selected feature names don't match nodes"
    
    # Check that number of selected features matches num_features
    assert len(evaluator.selected_feature_names) == len(nodes), \
        f"Expected {len(nodes)} selected features, got {len(evaluator.selected_feature_names)}"
    
    # Check that indices and names correspond
    for idx, name in zip(evaluator.selected_features, evaluator.selected_feature_names):
        assert nodes[idx] == name, \
            f"Selected index {idx} should map to {nodes[idx]}, not {name}"


def test_importance_values_match_nodes(parsed_data):
    """Test that feature importance values match the correct nodes."""
    # Get data from parser
    nodes_info = parsed_data['summary']()
    nodes = nodes_info['nodes']

    # Get input and node outputs
    input_voltages = parsed_data['get_input_voltages']()
    nodes_output = parsed_data['get_node_voltages']()

    # Get input signal
    primary_input_node = nodes_info['input_nodes'][0]
    input_signal = input_voltages[primary_input_node]
    
    # Create MC evaluator
    evaluator = MemoryCapacityEvaluator(
        input_signal,
        nodes_output,
        max_delay=5,  # Use a small value for testing
        random_state=42,
        node_names=nodes
    )
    
    # Run memory capacity calculation
    results = evaluator.calculate_total_memory_capacity(
        feature_selection_method='pca',
        num_features=len(nodes),  # Select all nodes
        regression_alpha=0.1,
        train_ratio=0.8
    )
    
    # Get feature importance
    feature_importance = evaluator.feature_selector.get_feature_importance()
    
    # Check that feature importance Series has the correct index
    assert all(name in feature_importance.index for name in nodes), \
        "Not all node names found in feature importance index"
    
    # Get the selected node names and their importance scores
    selected_names = evaluator.selected_feature_names
    importance_scores = np.array([feature_importance[name] for name in selected_names])
    
    # Verify scores are in descending order (highest importance first)
    assert np.all(np.diff(importance_scores) <= 0), \
        "Importance scores are not in descending order"
    
    # Create a DataFrame with node names and importance scores for debugging
    importance_df = pd.DataFrame({
        'node': selected_names,
        'importance': importance_scores
    })
    print("\nNode importance scores:")
    print(importance_df)
    
    # Verify consistency by running a second time
    second_evaluator = MemoryCapacityEvaluator(
        input_signal,
        nodes_output,
        max_delay=5,
        random_state=42,
        node_names=nodes
    )
    
    second_evaluator.calculate_total_memory_capacity(
        feature_selection_method='pca',
        num_features=len(nodes),
        regression_alpha=0.1,
        train_ratio=0.8
    )
    
    # Compare selected nodes to ensure consistency
    assert evaluator.selected_feature_names == second_evaluator.selected_feature_names, \
        "Selected nodes are not consistent between runs"


def test_selected_columns_match_node_data(parsed_data):
    """Test that selected columns match the actual node data."""
    # Get data from parser
    nodes_info = parsed_data['summary']()
    nodes = nodes_info['nodes']
    
    # Get input and node outputs
    input_voltages = parsed_data['get_input_voltages']()
    nodes_output = parsed_data['get_node_voltages']()
    
    # Get input signal
    primary_input_node = nodes_info['input_nodes'][0]
    input_signal = input_voltages[primary_input_node]
    
    # Create dummy target for testing
    y = np.sin(input_signal)
    
    # Initialize feature selector
    feature_selector = FeatureSelector(random_state=42)
    
    # Perform feature selection
    X_selected, selected_indices, selected_names = feature_selector.select_features(
        X=nodes_output,
        y=y,
        node_names=nodes,
        method='pca',
        num_features=5  # Just select a few top nodes
    )
    
    # Print for debugging
    print(f"Original node names: {nodes}")
    print(f"Selected indices: {selected_indices}")
    print(f"Selected names: {selected_names}")
    
    # Create a DataFrame with the original data
    full_df = pd.DataFrame(nodes_output, columns=nodes)
    
    # Create a DataFrame with just the selected columns using indices
    selected_df = pd.DataFrame(X_selected, columns=selected_names)
    
    # Verify data matches by comparing sample values
    for i, col_name in enumerate(selected_names):
        col_idx = selected_indices[i]
        
        # Get first 5 values from both DataFrames
        orig_values = full_df[col_name].values[:5]
        selected_values = selected_df[col_name].values[:5]
        
        # Print for debugging
        print(f"\nNode {col_name} (index {col_idx}):")
        print(f"Original data (first 5): {orig_values}")
        print(f"Selected data (first 5): {selected_values}")
        
        # Check that values match
        assert np.allclose(orig_values, selected_values), \
            f"Data mismatch for node {col_name} (index {col_idx})"


def test_raw_measurement_data_matches_selected_nodes():
    """Test that node data matches the raw measurement file columns."""
    # Load the measurement data
    BASE_DIR = Path(__file__).resolve().parent.parent
    filename = "074_INRiMARC_NWN_Pad129M_gridSE_MemoryCapacity_2024_04_02.txt"
    measurement_file = BASE_DIR / "tests" / "test_files" / filename
    
    # Load raw data directly from file
    raw_df = pd.read_csv(measurement_file, sep=r'\s+', engine='python')
    
    # Get data from target column
    target_node = '10'
    target_column = f'{target_node}_V[V]'
    
    if target_column not in raw_df.columns:
        print(f"Available columns: {[col for col in raw_df.columns if '_V[V]' in col]}")
        pytest.skip(f"Target column {target_column} not found in raw DataFrame")
    
    # Get the raw target data (our baseline for comparison)
    target = raw_df[target_column].values
    print(f"Raw target data (first 10): {target[:10]}")

    # Load and parse the data using the standard pipeline
    loader = MeasurementLoader(measurement_file)
    dataset = loader.get_dataset()
    
    # Get node information using the parser
    nodes_info = MeasurementParser.identify_nodes(dataset.dataframe)
    nodes = nodes_info['nodes']
    
    print(f"Computation nodes: {nodes}")
    
    # Check if target node is in nodes
    if target_node not in nodes:
        # If not, use any node that is available
        target_node = nodes[0] if nodes else None
        if not target_node:
            pytest.skip("No computation nodes available for testing")
            
    print(f"Target node: {target_node}")
    
    # Get input and node outputs using static methods
    input_voltages = MeasurementParser.get_input_voltages(dataset.dataframe, nodes_info['input_nodes'])
    nodes_output = MeasurementParser.get_node_voltages(dataset.dataframe, nodes)
    
    # Get individual node voltage arrays directly from the dataframe
    node_voltages = {}
    for node in nodes:
        col = f'{node}_V[V]'
        if col in dataset.dataframe.columns:
            node_voltages[node] = dataset.dataframe[col].values
        else:
            print(f"Could not get voltage for node {node}: Column {col} not found")
    
    # Get index of target node in nodes list
    target_idx = nodes.index(target_node)
    
    # Get values directly from the dataframe for our target node
    target_voltage = node_voltages.get(target_node, None)
    if target_voltage is None:
        pytest.skip(f"Could not get voltage data for node {target_node}")
    
    # Get values from nodes_output (matrix of all node readings)
    nodes_values = nodes_output[:10, target_idx]
    
    # Get raw values from the direct node reading
    raw_values = target_voltage[:10]
    
    print(f"Raw values from dataframe (first 10): {raw_values}")
    print(f"Node values from nodes_output matrix (first 10): {nodes_values}")
    print(f"Original target values from raw file (first 10): {target[:10]}")
    
    # Verify that raw data from the file matches node output for the target node
    assert np.allclose(target[:10], nodes_values, rtol=1e-5, atol=1e-5), \
        f"Data mismatch between raw file data and nodes_output for node {target_node}"
    
    # Verify that raw values from dataframe match nodes_output
    assert np.allclose(raw_values, nodes_values, rtol=1e-5, atol=1e-5), \
        f"Data mismatch between dataframe and nodes_output for node {target_node}"
    
    # Now run feature selection
    y = np.sin(input_voltages[nodes_info['input_nodes'][0]])  # Use sine of input as target
    
    # Run feature selection
    feature_selector = FeatureSelector(random_state=42)
    X_selected, selected_indices, selected_names = feature_selector.select_features(
        X=nodes_output,
        y=y,
        node_names=nodes,
        method='pca',
        num_features='all'
    )
    
    print(f"Selected indices: {selected_indices}")
    print(f"Selected names: {selected_names}")
    
    # Check if target node was selected
    if target_node in selected_names:
        # Get the index of the target node in the selected features
        selected_idx = selected_names.index(target_node)
        
        # Get values from selected features
        selected_values = X_selected[:10, selected_idx]
        
        print(f"Selected feature values (first 10): {selected_values}")
        
        # Verify raw file data matches selected values
        assert np.allclose(target[:10], selected_values, rtol=1e-5, atol=1e-5), \
            f"Data mismatch between raw file data and selected feature values for node {target_node}"
        
        # Verify that raw values match selected feature values
        assert np.allclose(raw_values, selected_values, rtol=1e-5, atol=1e-5), \
            f"Data mismatch between dataframe and selected feature values for node {target_node}"
        
        # Double check that nodes_output matches selected values
        assert np.allclose(nodes_values, selected_values, rtol=1e-5, atol=1e-5), \
            f"Data mismatch between nodes_output and selected feature values for node {target_node}"
        
        print(f"\n✅ VERIFIED: Original raw data for '{target_column}' matches node voltage for node '{target_node}' through all processing stages")
    else:
        # If node 10 wasn't selected, check any node that was selected
        if selected_names and selected_indices:
            test_node = selected_names[0]
            test_idx = selected_indices[0]
            node_idx = nodes.index(test_node)
            
            # Get the raw data for this test node
            test_column = f'{test_node}_V[V]'
            test_target = raw_df[test_column].values if test_column in raw_df.columns else None
            
            if test_target is not None:
                # Get raw values for this node
                test_voltage = node_voltages.get(test_node, None)
                if test_voltage is not None:
                    test_raw_values = test_voltage[:10]
                    test_nodes_values = nodes_output[:10, node_idx]
                    test_selected_values = X_selected[:10, 0]  # First selected feature
                    
                    print(f"\nTesting with alternative node {test_node}:")
                    print(f"Raw file values (first 10): {test_target[:10]}")
                    print(f"Parser dataframe values (first 10): {test_raw_values}")
                    print(f"Nodes values (first 10): {test_nodes_values}")
                    print(f"Selected values (first 10): {test_selected_values}")
                    
                    # Verify raw file data matches nodes_output 
                    assert np.allclose(test_target[:10], test_nodes_values, rtol=1e-5, atol=1e-5), \
                        f"Data mismatch between raw file data and nodes_output for node {test_node}"
                    
                    # Verify raw parser dataframe values match nodes_output
                    assert np.allclose(test_raw_values, test_nodes_values, rtol=1e-5, atol=1e-5), \
                        f"Data mismatch between parser dataframe and nodes_output for node {test_node}"
                    
                    # Verify nodes_output matches selected values
                    assert np.allclose(test_nodes_values, test_selected_values, rtol=1e-5, atol=1e-5), \
                        f"Data mismatch between nodes_output and selected values for node {test_node}"
                        
                    print(f"\n✅ VERIFIED: Original raw data for '{test_column}' matches node voltage for node '{test_node}' through all processing stages")
            else:
                pytest.skip(f"Could not get raw data for alternative node {test_node}")
        else:
            pytest.skip("No nodes were selected during feature selection")


def test_node_data_consistency_with_raw_dataframe():
    """
    Test that node '10' in feature selection points to the same data as '10_V[V]' column
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
    target_node = '10'
    target_column = f'{target_node}_V[V]'
    
    if target_column not in raw_df.columns:
        all_voltage_columns = [col for col in raw_df.columns if '_V[V]' in col]
        print(f"Available voltage columns: {all_voltage_columns}")
        pytest.skip(f"Target column {target_column} not found in raw dataframe")
    
    # Get raw values from the dataframe
    raw_values = raw_df[target_column].values[:20]  # Get first 20 values
    print(f"Raw values from DataFrame['{target_column}'] (first 20):\n{raw_values}")
    
    # Get node information using the static method
    nodes_info = MeasurementParser.identify_nodes(dataset.dataframe)
    nodes = nodes_info['nodes']
    
    # Verify target node is in nodes
    if target_node not in nodes:
        pytest.skip(f"Target node {target_node} not found in nodes")
    
    # Get the node output matrix using the static method
    nodes_output = MeasurementParser.get_node_voltages(dataset.dataframe, nodes)
    
    # Get the index of target node in nodes
    target_idx = nodes.index(target_node)
    
    # Extract the values for this electrode from the node_output matrix
    node_values = nodes_output[:20, target_idx]
    print(f"Node values from nodes_output[:, {target_idx}] (first 20):\n{node_values}")
    
    # Verify raw dataframe values match parser's node output values
    assert np.allclose(raw_values, node_values, rtol=1e-5, atol=1e-5), \
        f"Data mismatch between raw dataframe and parser's node output for node {target_node}"
    
    # Now run feature selection
    input_voltages = MeasurementParser.get_input_voltages(dataset.dataframe, nodes_info['input_nodes'])
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
    
    print("\n✅ VERIFICATION SUCCESSFUL: Node '10' in feature selection points to the same data as '10_V[V]' column in raw dataframe")


def test_all_nodes_data_consistency():
    """
    Test that all nodes in feature selection point to the same data as their
    corresponding columns in the raw dataframe.
    """
    # Load the measurement data
    BASE_DIR = Path(__file__).resolve().parent.parent
    filename = "074_INRiMARC_NWN_Pad129M_gridSE_MemoryCapacity_2024_04_02.txt"
    measurement_file = BASE_DIR / "tests" / "test_files" / filename
    
    print(f"Looking for test file at: {measurement_file}")
    print(f"File exists: {measurement_file.exists()}")
    
    # Load the data directly using the ElecResDataset class
    dataset = ElecResDataset(measurement_file)
    
    # Get the raw dataframe directly from the dataset
    raw_df = dataset.dataframe
    print(f"Raw dataframe shape: {raw_df.shape}")
    print(f"Raw dataframe columns (first 5): {list(raw_df.columns)[:5]}")
    
    # Get node information
    nodes_info = dataset.summary()
    nodes = nodes_info['nodes']
    
    # Skip if no nodes found
    if not nodes:
        print("ERROR: No computation nodes found in dataset output")
        pytest.skip("No computation nodes found")
    
    print(f"Testing data consistency for {len(nodes)} nodes: {nodes}")
    
    # Get the node output matrix
    nodes_output = dataset.get_node_voltages()
    print(f"Node output matrix shape: {nodes_output.shape}")
    
    # Setup for feature selection
    input_voltages = dataset.get_input_voltages()
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
            print(f"Available columns (first 10): {list(raw_df.columns)[:10]}")
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
            print(f"Selected names: {selected_names}")
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