"""
RCbench Evaluator Performance Benchmark

This script benchmarks the execution time of all RCbench evaluators
using synthetic reservoir data generated with ReservoirPy.

The benchmark measures:
- Memory Capacity (MC)
- Sin(x) Approximation
- Nonlinear Transformation (NLT)
- NARMA-2
- Nonlinear Memory
- Information Processing Capacity (IPC)
- Kernel Rank
- Generalization Rank

Requirements:
    pip install reservoirpy tabulate

Author: Davide Pilati
Date: 2025
"""

import time
import functools
import numpy as np
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
import logging

# ReservoirPy imports
try:
    from reservoirpy.nodes import Reservoir
except ImportError:
    raise ImportError(
        "reservoirpy is required for this benchmark. "
        "Install it with: pip install reservoirpy"
    )

# RCbench imports
from rcbench import (
    MemoryCapacityEvaluator,
    SinxEvaluator,
    NltEvaluator,
    NarmaEvaluator,
    NonlinearMemoryEvaluator,
    IPCEvaluator,
    KernelRankEvaluator,
    GeneralizationRankEvaluator,
)
from rcbench import (
    MCPlotConfig,
    SinxPlotConfig,
    NLTPlotConfig,
    NarmaPlotConfig,
    NonlinearMemoryPlotConfig,
    IPCPlotConfig,
)
from rcbench.logger import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.WARNING)  # Reduce verbosity during benchmarking


# =============================================================================
# TIMING UTILITIES
# =============================================================================

@dataclass
class TimingResult:
    """Store timing results for a benchmark run."""
    evaluator_name: str
    method_name: str
    execution_time: float
    n_samples: int
    n_nodes: int
    additional_info: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"{self.evaluator_name}.{self.method_name}: {self.execution_time:.4f}s"


class BenchmarkTimer:
    """Context manager and decorator for timing code execution."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed: float = 0.0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
    
    @staticmethod
    def time_function(func: Callable) -> Callable:
        """Decorator to time a function execution."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            return result, elapsed
        return wrapper


def run_timed(func: Callable, *args, **kwargs) -> tuple:
    """Run a function and return (result, execution_time)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


# =============================================================================
# DATA GENERATION
# =============================================================================

def create_reservoir(
    units: int = 50,
    lr: float = 0.3,
    sr: float = 0.9,
    input_scaling: float = 0.1,
    seed: int = 42
) -> Reservoir:
    """Create a ReservoirPy Echo State Network."""
    return Reservoir(
        units=units,
        lr=lr,
        sr=sr,
        input_scaling=input_scaling,
        seed=seed
    )


def generate_test_data(
    n_samples: int = 3000,
    n_nodes: int = 50,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic test data for benchmarking.
    
    Returns:
    --------
    dict with:
        - 'random_input': Uniform random input in [-1, 1]
        - 'sine_input': Sinusoidal input
        - 'reservoir_states': Reservoir node outputs
        - 'node_names': List of node names
    """
    np.random.seed(seed)
    
    # Generate input signals
    # 1. Random white noise (uniform distribution for IPC Legendre basis)
    random_input = np.random.uniform(-1, 1, size=(n_samples, 1))
    
    # 2. Sinusoidal input (for frequency-based tasks)
    t = np.linspace(0, 10 * np.pi, n_samples)
    sine_input = np.sin(t).reshape(-1, 1)
    
    # Create and run reservoir
    reservoir = create_reservoir(units=n_nodes, seed=seed)
    reservoir_states = reservoir.run(random_input)
    
    # Also run with sine input for comparison
    reservoir2 = create_reservoir(units=n_nodes, seed=seed + 1)
    reservoir_states_sine = reservoir2.run(sine_input)
    
    # Node names
    node_names = [f'N{i}' for i in range(n_nodes)]
    
    return {
        'random_input': random_input.flatten(),
        'sine_input': sine_input.flatten(),
        'reservoir_states_random': reservoir_states,
        'reservoir_states_sine': reservoir_states_sine,
        'node_names': node_names,
        'n_samples': n_samples,
        'n_nodes': n_nodes
    }


# =============================================================================
# INDIVIDUAL BENCHMARKS
# =============================================================================

def benchmark_memory_capacity(data: Dict[str, np.ndarray]) -> TimingResult:
    """Benchmark MemoryCapacityEvaluator."""
    config = MCPlotConfig(show_plot=False)
    
    evaluator = MemoryCapacityEvaluator(
        input_signal=data['random_input'],
        nodes_output=data['reservoir_states_random'],
        max_delay=20,
        node_names=data['node_names'],
        plot_config=config
    )
    
    result, elapsed = run_timed(
        evaluator.calculate_total_memory_capacity,
        feature_selection_method='pca',
        num_features='all',
        modeltype='Ridge',
        regression_alpha=0.1,
        train_ratio=0.8
    )
    
    return TimingResult(
        evaluator_name='MemoryCapacityEvaluator',
        method_name='calculate_total_memory_capacity',
        execution_time=elapsed,
        n_samples=data['n_samples'],
        n_nodes=data['n_nodes'],
        additional_info={
            'max_delay': 20,
            'total_mc': result['total_memory_capacity']
        }
    )


def benchmark_sinx(data: Dict[str, np.ndarray]) -> TimingResult:
    """Benchmark SinxEvaluator."""
    config = SinxPlotConfig(show_plot=False)
    
    evaluator = SinxEvaluator(
        input_signal=data['random_input'],
        nodes_output=data['reservoir_states_random'],
        node_names=data['node_names'],
        plot_config=config
    )
    
    result, elapsed = run_timed(
        evaluator.run_evaluation,
        metric='NMSE',
        feature_selection_method='pca',
        num_features='all',
        modeltype='Ridge',
        regression_alpha=0.1,
        train_ratio=0.8
    )
    
    return TimingResult(
        evaluator_name='SinxEvaluator',
        method_name='run_evaluation',
        execution_time=elapsed,
        n_samples=data['n_samples'],
        n_nodes=data['n_nodes'],
        additional_info={
            'metric': 'NMSE',
            'accuracy': result['accuracy']
        }
    )


def benchmark_nlt(data: Dict[str, np.ndarray]) -> List[TimingResult]:
    """Benchmark NltEvaluator for all targets."""
    config = NLTPlotConfig(show_plot=False)
    
    # NLT requires a time array
    time_array = np.linspace(0, 10 * np.pi, data['n_samples'])
    
    evaluator = NltEvaluator(
        input_signal=data['sine_input'],  # Use sine for NLT
        nodes_output=data['reservoir_states_sine'],
        time_array=time_array,
        node_names=data['node_names'],
        plot_config=config
    )
    
    results = []
    
    # Benchmark each target
    for target_name in evaluator.targets:
        result, elapsed = run_timed(
            evaluator.run_evaluation,
            target_name=target_name,
            metric='NMSE',
            feature_selection_method='pca',
            num_features='all',
            modeltype='Ridge',
            regression_alpha=0.1,
            train_ratio=0.8,
            plot=False
        )
        
        results.append(TimingResult(
            evaluator_name='NltEvaluator',
            method_name=f'run_evaluation({target_name})',
            execution_time=elapsed,
            n_samples=data['n_samples'],
            n_nodes=data['n_nodes'],
            additional_info={
                'target': target_name,
                'accuracy': result['accuracy']
            }
        ))
    
    return results


def benchmark_narma(data: Dict[str, np.ndarray]) -> TimingResult:
    """Benchmark NarmaEvaluator."""
    config = NarmaPlotConfig(show_plot=False)
    
    # NARMA needs input in [0, 0.5] range
    narma_input = (data['random_input'] + 1) / 4  # Scale from [-1,1] to [0, 0.5]
    
    evaluator = NarmaEvaluator(
        input_signal=narma_input,
        nodes_output=data['reservoir_states_random'],
        order=2,  # NARMA-2 task
        node_names=data['node_names'],
        plot_config=config
    )
    
    result, elapsed = run_timed(
        evaluator.run_evaluation,
        metric='NMSE',
        feature_selection_method='pca',
        num_features='all',
        modeltype='Ridge',
        regression_alpha=0.1,
        train_ratio=0.8
    )
    
    return TimingResult(
        evaluator_name='NarmaEvaluator',
        method_name='run_evaluation',
        execution_time=elapsed,
        n_samples=data['n_samples'],
        n_nodes=data['n_nodes'],
        additional_info={
            'order': 2,
            'accuracy': result['accuracy']
        }
    )


def benchmark_nonlinear_memory(data: Dict[str, np.ndarray]) -> TimingResult:
    """Benchmark NonlinearMemoryEvaluator."""
    config = NonlinearMemoryPlotConfig(show_plot=False)
    
    evaluator = NonlinearMemoryEvaluator(
        input_signal=data['random_input'],
        nodes_output=data['reservoir_states_random'],
        tau_values=[1, 2, 3, 4, 5],  # Reduced for speed
        nu_values=[0.1, 1.0, 3.0],   # Reduced for speed
        node_names=data['node_names'],
        plot_config=config
    )
    
    result, elapsed = run_timed(
        evaluator.run_parameter_sweep,
        feature_selection_method='pca',
        num_features='all',
        modeltype='Ridge',
        regression_alpha=0.1,
        train_ratio=0.8,
        metric='NMSE'
    )
    
    return TimingResult(
        evaluator_name='NonlinearMemoryEvaluator',
        method_name='run_parameter_sweep',
        execution_time=elapsed,
        n_samples=data['n_samples'],
        n_nodes=data['n_nodes'],
        additional_info={
            'tau_values': [1, 2, 3, 4, 5],
            'nu_values': [0.1, 1.0, 3.0],
            'num_combinations': 15
        }
    )


def benchmark_ipc(data: Dict[str, np.ndarray]) -> TimingResult:
    """Benchmark IPCEvaluator."""
    config = IPCPlotConfig(show_plot=False)
    
    evaluator = IPCEvaluator(
        input_signal=data['random_input'],
        nodes_output=data['reservoir_states_random'],
        max_delay=5,     # Reduced for speed
        max_degree=2,    # Reduced for speed
        include_cross_terms=True,
        node_names=data['node_names'],
        plot_config=config
    )
    
    result, elapsed = run_timed(
        evaluator.calculate_total_capacity,
        feature_selection_method='pca',
        num_features='all',
        modeltype='Ridge',
        regression_alpha=0.1,
        train_ratio=0.8
    )
    
    return TimingResult(
        evaluator_name='IPCEvaluator',
        method_name='calculate_total_capacity',
        execution_time=elapsed,
        n_samples=data['n_samples'],
        n_nodes=data['n_nodes'],
        additional_info={
            'max_delay': 5,
            'max_degree': 2,
            'total_capacity': result['total_capacity'],
            'num_basis_functions': result['num_basis_functions']
        }
    )


def benchmark_kernel_rank(data: Dict[str, np.ndarray]) -> TimingResult:
    """Benchmark KernelRankEvaluator."""
    # KernelRankEvaluator only needs nodes_output and kernel parameters
    evaluator = KernelRankEvaluator(
        nodes_output=data['reservoir_states_random'],
        kernel='rbf',
        sigma=1.0,
        threshold=0.01
    )
    
    result, elapsed = run_timed(evaluator.run_evaluation)
    
    return TimingResult(
        evaluator_name='KernelRankEvaluator',
        method_name='run_evaluation',
        execution_time=elapsed,
        n_samples=data['n_samples'],
        n_nodes=data['n_nodes'],
        additional_info={
            'kernel': 'rbf',
            'rank': result.get('kernel_rank', 'N/A')
        }
    )


def benchmark_generalization_rank(data: Dict[str, np.ndarray]) -> TimingResult:
    """Benchmark GeneralizationRankEvaluator."""
    # GeneralizationRankEvaluator takes states (reservoir output) and threshold
    evaluator = GeneralizationRankEvaluator(
        states=data['reservoir_states_random'],
        threshold=0.01
    )
    
    result, elapsed = run_timed(evaluator.run_evaluation)
    
    return TimingResult(
        evaluator_name='GeneralizationRankEvaluator',
        method_name='run_evaluation',
        execution_time=elapsed,
        n_samples=data['n_samples'],
        n_nodes=data['n_nodes'],
        additional_info={
            'generalization_rank': result.get('generalization_rank', 'N/A')
        }
    )


# =============================================================================
# MAIN BENCHMARK RUNNER
# =============================================================================

def run_all_benchmarks(
    n_samples: int = 3000,
    n_nodes: int = 50,
    n_runs: int = 3,
    seed: int = 42
) -> Dict[str, List[TimingResult]]:
    """
    Run all benchmarks and collect timing results.
    
    Parameters:
    -----------
    n_samples : int
        Number of time samples in the dataset
    n_nodes : int
        Number of reservoir nodes
    n_runs : int
        Number of times to run each benchmark (for averaging)
    seed : int
        Random seed
        
    Returns:
    --------
    dict
        Dictionary mapping evaluator names to lists of TimingResult objects
    """
    print("=" * 70)
    print("RCBENCH EVALUATOR PERFORMANCE BENCHMARK")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Samples: {n_samples}")
    print(f"  Nodes: {n_nodes}")
    print(f"  Runs per evaluator: {n_runs}")
    print()
    
    # Generate data once
    print("Generating synthetic reservoir data...")
    data = generate_test_data(n_samples=n_samples, n_nodes=n_nodes, seed=seed)
    print(f"  Input shape: {data['random_input'].shape}")
    print(f"  Reservoir states shape: {data['reservoir_states_random'].shape}")
    print()
    
    all_results: Dict[str, List[TimingResult]] = {}
    
    benchmarks = [
        ("Memory Capacity", benchmark_memory_capacity),
        ("Sin(x)", benchmark_sinx),
        ("NLT", benchmark_nlt),
        ("NARMA-2", benchmark_narma),
        ("Nonlinear Memory", benchmark_nonlinear_memory),
        ("IPC", benchmark_ipc),
        ("Kernel Rank", benchmark_kernel_rank),
        ("Generalization Rank", benchmark_generalization_rank),
    ]
    
    for name, benchmark_func in benchmarks:
        print(f"Benchmarking {name}...", end=" ", flush=True)
        
        try:
            run_times = []
            
            for run in range(n_runs):
                result = benchmark_func(data)
                
                # Handle list results (e.g., NLT with multiple targets)
                if isinstance(result, list):
                    for r in result:
                        if r.evaluator_name not in all_results:
                            all_results[r.evaluator_name] = []
                        all_results[r.evaluator_name].append(r)
                        run_times.append(r.execution_time)
                else:
                    if result.evaluator_name not in all_results:
                        all_results[result.evaluator_name] = []
                    all_results[result.evaluator_name].append(result)
                    run_times.append(result.execution_time)
            
            avg_time = np.mean(run_times)
            print(f"Done! (avg: {avg_time:.4f}s)")
            
        except Exception as e:
            print(f"FAILED: {e}")
    
    return all_results


def print_summary(results: Dict[str, List[TimingResult]]) -> None:
    """Print a summary table of all benchmark results."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    # Aggregate results by evaluator
    summary_data = []
    
    for evaluator_name, timing_results in results.items():
        times = [r.execution_time for r in timing_results]
        avg_time = np.mean(times)
        std_time = np.std(times) if len(times) > 1 else 0.0
        min_time = np.min(times)
        max_time = np.max(times)
        
        # Get method name (may vary for NLT)
        methods = set(r.method_name for r in timing_results)
        method_str = list(methods)[0] if len(methods) == 1 else "multiple"
        
        summary_data.append({
            'Evaluator': evaluator_name,
            'Method': method_str,
            'Avg Time (s)': avg_time,
            'Std (s)': std_time,
            'Min (s)': min_time,
            'Max (s)': max_time,
            'Runs': len(times)
        })
    
    # Sort by average time
    summary_data.sort(key=lambda x: x['Avg Time (s)'])
    
    # Try to use tabulate for nice formatting
    try:
        from tabulate import tabulate
        
        headers = ['Evaluator', 'Method', 'Avg (s)', 'Std (s)', 'Min (s)', 'Max (s)', 'Runs']
        rows = [
            [
                d['Evaluator'],
                d['Method'][:30] + '...' if len(d['Method']) > 30 else d['Method'],
                f"{d['Avg Time (s)']:.4f}",
                f"{d['Std (s)']:.4f}",
                f"{d['Min (s)']:.4f}",
                f"{d['Max (s)']:.4f}",
                d['Runs']
            ]
            for d in summary_data
        ]
        
        print("\n" + tabulate(rows, headers=headers, tablefmt='grid'))
        
    except ImportError:
        # Fallback to simple formatting
        print(f"\n{'Evaluator':<35} {'Avg (s)':<12} {'Std (s)':<12} {'Runs':<6}")
        print("-" * 70)
        for d in summary_data:
            print(f"{d['Evaluator']:<35} {d['Avg Time (s)']:<12.4f} {d['Std (s)']:<12.4f} {d['Runs']:<6}")
    
    # Print total time
    total_avg = sum(d['Avg Time (s)'] for d in summary_data)
    print(f"\nTotal average execution time: {total_avg:.4f}s")
    
    # Ranking
    print("\n" + "-" * 70)
    print("SPEED RANKING (fastest to slowest):")
    print("-" * 70)
    for i, d in enumerate(summary_data, 1):
        print(f"  {i}. {d['Evaluator']}: {d['Avg Time (s)']:.4f}s")


def print_detailed_results(results: Dict[str, List[TimingResult]]) -> None:
    """Print detailed results including additional info."""
    print("\n" + "=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)
    
    for evaluator_name, timing_results in sorted(results.items()):
        print(f"\n{evaluator_name}:")
        print("-" * 50)
        
        for result in timing_results[:3]:  # Show first 3 runs
            print(f"  Method: {result.method_name}")
            print(f"  Time: {result.execution_time:.4f}s")
            print(f"  Data: {result.n_samples} samples, {result.n_nodes} nodes")
            if result.additional_info:
                for key, value in result.additional_info.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.6f}")
                    else:
                        print(f"  {key}: {value}")
            print()


def main():
    """Run the complete benchmark suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description='RCbench Evaluator Performance Benchmark')
    parser.add_argument('--samples', type=int, default=3000, help='Number of samples')
    parser.add_argument('--nodes', type=int, default=50, help='Number of reservoir nodes')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs per benchmark')
    parser.add_argument('--detailed', action='store_true', help='Show detailed results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Run benchmarks
    results = run_all_benchmarks(
        n_samples=args.samples,
        n_nodes=args.nodes,
        n_runs=args.runs,
        seed=args.seed
    )
    
    # Print summary
    print_summary(results)
    
    # Print detailed results if requested
    if args.detailed:
        print_detailed_results(results)
    
    return results


if __name__ == "__main__":
    results = main()

