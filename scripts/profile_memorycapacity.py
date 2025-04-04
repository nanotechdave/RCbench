import cProfile
import pstats
from pathlib import Path
from rcbench.measurements.loader import MeasurementLoader
from rcbench.measurements.parser import MeasurementParser
from rcbench.tasks.memorycapacity import MemoryCapacityEvaluator, PlotConfig
from rcbench.logger import get_logger

logger = get_logger(__name__)

def run_memory_capacity():
    # Load and parse your measurement data
    BASE_DIR = Path(__file__).resolve().parent
    filename = "074_INRiMARC_NWN_Pad129M_gridSE_MemoryCapacity_2024_04_02.txt"
    measurement_file = BASE_DIR.parent / "tests" / "test_files" / filename
    
    # Load and parse data
    dataset = MeasurementLoader(measurement_file).get_dataset()
    parser = MeasurementParser(dataset)
    
    # Extract necessary signals
    input_signal = parser.get_input_voltages()[parser.input_electrodes[0]]
    nodes_output = parser.get_node_voltages()
    
    # Initialize the Memory Capacity evaluator
    evaluator = MemoryCapacityEvaluator(
        input_signal=input_signal,
        nodes_output=nodes_output,
        max_delay=30
    )
    
    # Create plot configuration
    plot_config = PlotConfig(
        save_dir=None,
        plot_mc_vs_delay=True,
        plot_feature_importance=True,
        plot_prediction_results=True,
        plot_cumulative_mc=True,
        plot_mc_heatmap=True
    )
    
    # Run evaluation
    results = evaluator.calculate_total_memory_capacity(
        feature_selection_method='pca',
        num_features=6,
        regression_alpha=0.1,
        train_ratio=0.8
    )
    
    # Generate plots
    evaluator.plot_results(plot_config)
    
    logger.output(f"Total Memory Capacity: {results['total_memory_capacity']:.4f}")

if __name__ == "__main__":
    # Create a Profile object
    profiler = cProfile.Profile()
    
    # Start profiling
    profiler.enable()
    
    # Run the main function
    run_memory_capacity()
    
    # Stop profiling
    profiler.disable()
    
    # Create Stats object and sort by cumulative time
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    
    # Print the top 20 time-consuming functions
    print("\nTop 20 time-consuming functions:")
    stats.print_stats(20)
    
    # Print the top 20 time-consuming functions in the rcbench package
    print("\nTop 20 time-consuming functions in rcbench:")
    stats.print_stats(20, 'rcbench')
