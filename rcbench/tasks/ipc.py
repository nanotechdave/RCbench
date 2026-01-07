"""
Information Processing Capacity (IPC) Evaluator

Based on: Dambre, J., Verstraeten, D., Schrauwen, B. & Massar, S. 
"Information Processing Capacity of Dynamical Systems"
Scientific Reports 2, 514 (2012). DOI: 10.1038/srep00514

This evaluator measures the total computational capacity of a dynamical system
by decomposing it into orthonormal basis functions (Legendre polynomials for
uniform inputs). The total capacity is bounded by N (number of state variables)
and equals N if the system has fading memory.

The capacity is decomposed into:
- Linear memory capacity (degree 1)
- Nonlinear memory capacity (degree > 1)
- Cross-delay capacity (products of different delays)
"""

import numpy as np
from scipy.special import legendre
from sklearn.linear_model import Ridge, LinearRegression
from typing import Dict, List, Tuple, Union, Any, Optional
from itertools import combinations_with_replacement, product
from rcbench.tasks.baseevaluator import BaseEvaluator
from rcbench.tasks.featureselector import FeatureSelector
from rcbench.logger import get_logger

logger = get_logger(__name__)


class IPCEvaluator(BaseEvaluator):
    """
    Information Processing Capacity Evaluator.
    
    Computes the computational capacity of a reservoir system by measuring
    its ability to reconstruct orthonormal basis functions of the input.
    
    For uniform inputs in [-1, 1], uses Legendre polynomials as basis functions.
    The capacity C_i for each basis function z_i is the squared correlation
    coefficient between the target z_i and the reservoir's linear readout prediction.
    
    Total capacity = Σ C_i ≤ N (number of reservoir nodes)
    
    Reference:
        Dambre et al., "Information Processing Capacity of Dynamical Systems",
        Scientific Reports 2, 514 (2012).
    """
    
    def __init__(self,
                 input_signal: Union[np.ndarray, List[float]],
                 nodes_output: np.ndarray,
                 max_delay: int = 10,
                 max_degree: int = 3,
                 max_total_degree: int = None,
                 include_cross_terms: bool = True,
                 random_state: int = 42,
                 node_names: Optional[List[str]] = None,
                 plot_config: Optional[Any] = None) -> None:
        """
        Initialize the IPC Evaluator.
        
        Parameters:
        -----------
        input_signal : array-like
            Input stimulation signal. Should be uniform in [-1, 1] for
            Legendre polynomial basis to be orthonormal.
        nodes_output : np.ndarray
            Reservoir node output (features), shape [time_steps, n_nodes].
        max_delay : int
            Maximum delay to consider (default: 10).
        max_degree : int
            Maximum polynomial degree per variable (default: 3).
        max_total_degree : int, optional
            Maximum total degree for cross-terms. If None, equals max_degree.
        include_cross_terms : bool
            Whether to include products of polynomials at different delays.
        random_state : int
            Random seed for reproducibility.
        node_names : List[str], optional
            Names of the nodes.
        plot_config : optional
            Configuration for plotting.
        """
        super().__init__(input_signal, nodes_output, node_names)
        
        self.max_delay = max_delay
        self.max_degree = max_degree
        self.max_total_degree = max_total_degree if max_total_degree is not None else max_degree
        self.include_cross_terms = include_cross_terms
        self.random_state = random_state
        self.plot_config = plot_config
        
        # Generate all basis functions
        self.basis_functions = self._generate_basis_functions()
        
        # Storage for results
        self.evaluation_results = None
        self.capacity_by_degree = None
        self.capacity_by_delay = None
        
        # Node names
        if node_names is None:
            self.node_names = [f'Node {i}' for i in range(nodes_output.shape[1])]
        else:
            self.node_names = node_names
        
        logger.info(f"Initialized IPC Evaluator")
        logger.info(f"  Max delay: {self.max_delay}")
        logger.info(f"  Max degree: {self.max_degree}")
        logger.info(f"  Include cross-terms: {self.include_cross_terms}")
        logger.info(f"  Total basis functions: {len(self.basis_functions)}")
        logger.info(f"  Theoretical max capacity: {nodes_output.shape[1]}")
    
    def _legendre_polynomial(self, n: int, x: np.ndarray) -> np.ndarray:
        """
        Evaluate Legendre polynomial P_n(x).
        
        Normalized such that E[P_n(x) * P_m(x)] = δ_{nm} for uniform x in [-1,1].
        
        Parameters:
        -----------
        n : int
            Degree of the polynomial.
        x : np.ndarray
            Input values (should be in [-1, 1]).
            
        Returns:
        --------
        np.ndarray
            Legendre polynomial values, normalized for orthonormality.
        """
        if n == 0:
            return np.ones_like(x)
        
        # Get Legendre polynomial coefficients
        leg = legendre(n)
        result = leg(x)
        
        # Normalize: for Legendre polynomials on [-1, 1],
        # the normalization factor is sqrt((2n+1)/2)
        # This ensures E[P_n(x)^2] = 1 for uniform x in [-1, 1]
        normalization = np.sqrt((2 * n + 1) / 2)
        
        return result * normalization
    
    def _generate_basis_functions(self) -> Dict[Tuple, Dict[str, Any]]:
        """
        Generate all basis function specifications.
        
        Each basis function is a product of Legendre polynomials:
        z(t) = Π_i P_{d_i}(u(t - τ_i))
        
        Returns:
        --------
        dict
            Dictionary mapping function ID (tuple) to function specification.
            Key format: ((delay1, degree1), (delay2, degree2), ...)
        """
        basis_functions = {}
        
        # Single variable terms: P_d(u(t-τ)) for d >= 1, τ >= 1
        for delay in range(1, self.max_delay + 1):
            for degree in range(1, self.max_degree + 1):
                key = ((delay, degree),)
                basis_functions[key] = {
                    'terms': [(delay, degree)],
                    'total_degree': degree,
                    'num_delays': 1,
                    'description': f'P_{degree}(u(t-{delay}))'
                }
        
        # Cross-terms (products of different delays)
        if self.include_cross_terms:
            # Two-variable products
            for delay1 in range(1, self.max_delay + 1):
                for delay2 in range(delay1 + 1, self.max_delay + 1):
                    for degree1 in range(1, self.max_degree + 1):
                        for degree2 in range(1, self.max_degree + 1):
                            total_degree = degree1 + degree2
                            if total_degree <= self.max_total_degree:
                                key = ((delay1, degree1), (delay2, degree2))
                                basis_functions[key] = {
                                    'terms': [(delay1, degree1), (delay2, degree2)],
                                    'total_degree': total_degree,
                                    'num_delays': 2,
                                    'description': f'P_{degree1}(u(t-{delay1})) * P_{degree2}(u(t-{delay2}))'
                                }
        
        return basis_functions
    
    def _compute_target(self, basis_key: Tuple) -> np.ndarray:
        """
        Compute target values for a given basis function.
        
        Parameters:
        -----------
        basis_key : tuple
            Key identifying the basis function.
            
        Returns:
        --------
        np.ndarray
            Target values z(t) for the basis function.
        """
        spec = self.basis_functions[basis_key]
        terms = spec['terms']
        
        # Start with ones
        result = np.ones(len(self.input_signal))
        
        for delay, degree in terms:
            # Get delayed input
            delayed_input = np.roll(self.input_signal, delay)
            
            # Compute Legendre polynomial
            poly_values = self._legendre_polynomial(degree, delayed_input)
            
            # Multiply into result
            result = result * poly_values
        
        # Set invalid region to zero
        max_delay_in_terms = max(d for d, _ in terms)
        result[:max_delay_in_terms] = 0
        
        return result
    
    def evaluate_capacity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Evaluate capacity as squared correlation coefficient.
        
        C_i = cov(y_true, y_pred)² / (var(y_true) * var(y_pred))
        
        This equals 1 - NMSE when the prediction is optimal.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True target values.
        y_pred : np.ndarray
            Predicted values.
            
        Returns:
        --------
        float
            Capacity value in [0, 1].
        """
        n = len(y_true)
        
        # Calculate means
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        
        # Calculate covariance and variances
        diff_true = y_true - mean_true
        diff_pred = y_pred - mean_pred
        
        cov = np.sum(diff_true * diff_pred) / (n - 1)
        var_true = np.sum(diff_true ** 2) / (n - 1)
        var_pred = np.sum(diff_pred ** 2) / (n - 1)
        
        # Check for zero variance
        if var_true == 0 or var_pred == 0:
            return 0.0
        
        # Squared correlation coefficient
        capacity = (cov ** 2) / (var_true * var_pred)
        
        return max(0.0, min(1.0, capacity))  # Clamp to [0, 1]
    
    def run_evaluation(self,
                      basis_key: Tuple,
                      modeltype: str = "Ridge",
                      regression_alpha: float = 1.0,
                      train_ratio: float = 0.8) -> Dict[str, Any]:
        """
        Run evaluation for a specific basis function.
        
        Parameters:
        -----------
        basis_key : tuple
            Key identifying the basis function.
        modeltype : str
            Type of regression model ('Ridge' or 'Linear').
        regression_alpha : float
            Regularization parameter for Ridge regression.
        train_ratio : float
            Ratio of data to use for training.
            
        Returns:
        --------
        dict
            Dictionary containing evaluation results.
        """
        spec = self.basis_functions[basis_key]
        
        # Compute target for this basis function
        target = self._compute_target(basis_key)
        
        # Use data from max_delay onwards
        start_idx = self.max_delay
        data_length = len(self.input_signal) - self.max_delay
        
        # Extract valid data window
        X = self.nodes_output[start_idx:start_idx + data_length]
        y = target[start_idx:start_idx + data_length]
        
        # Split into train/test
        X_train, X_test, y_train, y_test = self.split_train_test(X, y, train_ratio)
        
        # Apply feature selection if set
        X_train_selected = self.apply_feature_selection(X_train)
        X_test_selected = self.apply_feature_selection(X_test)
        
        # Train regression model
        if modeltype.lower() == "ridge":
            model = Ridge(alpha=regression_alpha, random_state=self.random_state)
        elif modeltype.lower() == "linear":
            model = LinearRegression()
        else:
            raise ValueError("Model unrecognized, please select 'Ridge' or 'Linear'")
        
        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)
        
        # Evaluate capacity
        capacity = self.evaluate_capacity(y_test, y_pred)
        
        return {
            'basis_key': basis_key,
            'capacity': capacity,
            'total_degree': spec['total_degree'],
            'num_delays': spec['num_delays'],
            'description': spec['description'],
            'y_test': y_test,
            'y_pred': y_pred,
            'model': model
        }
    
    def calculate_total_capacity(self,
                                feature_selection_method: str = 'pca',
                                num_features: Union[int, str] = 'all',
                                modeltype: str = "Ridge",
                                regression_alpha: float = 1.0,
                                train_ratio: float = 0.8) -> Dict[str, Any]:
        """
        Calculate total information processing capacity.
        
        This evaluates all basis functions and sums up the capacities,
        organized by polynomial degree and delay.
        
        Parameters:
        -----------
        feature_selection_method : str
            Method for feature selection ('pca', 'kbest', 'none').
        num_features : int or str
            Number of features to use, or 'all'.
        modeltype : str
            Type of regression model ('Ridge' or 'Linear').
        regression_alpha : float
            Regularization parameter for Ridge regression.
        train_ratio : float
            Ratio of data to use for training.
            
        Returns:
        --------
        dict
            Dictionary containing:
            - total_capacity: Sum of all capacities
            - linear_memory_capacity: Sum of degree-1 capacities (standard MC)
            - nonlinear_capacity: Sum of degree > 1 capacities
            - capacity_by_degree: dict mapping degree to total capacity
            - capacity_by_delay: dict mapping delay to total capacity
            - all_results: dict mapping basis_key to result dict
        """
        logger.info("Calculating Information Processing Capacity...")
        
        # Set random seed
        np.random.seed(self.random_state)
        
        # Initialize feature selector
        self.feature_selector = FeatureSelector(random_state=self.random_state)
        
        # Perform feature selection once using first basis function
        first_key = list(self.basis_functions.keys())[0]
        first_target = self._compute_target(first_key)
        
        X = self.nodes_output[self.max_delay:]
        y = first_target[self.max_delay:]
        
        X_train, _, y_train, _ = self.split_train_test(X, y, train_ratio)
        
        self.feature_selection(
            X_train, y_train,
            method=feature_selection_method,
            num_features=num_features
        )
        
        logger.info(f"Selected {len(self.selected_features)} features")
        
        # Evaluate all basis functions
        all_results = {}
        total_capacity = 0.0
        linear_memory_capacity = 0.0
        nonlinear_capacity = 0.0
        
        capacity_by_degree = {}
        capacity_by_delay = {}
        
        total_functions = len(self.basis_functions)
        
        for i, basis_key in enumerate(self.basis_functions.keys()):
            if (i + 1) % 10 == 0:
                logger.info(f"Evaluating basis function {i+1}/{total_functions}")
            
            try:
                result = self.run_evaluation(
                    basis_key=basis_key,
                    modeltype=modeltype,
                    regression_alpha=regression_alpha,
                    train_ratio=train_ratio
                )
                
                capacity = result['capacity']
                degree = result['total_degree']
                
                all_results[basis_key] = result
                total_capacity += capacity
                
                # Categorize by degree
                if degree == 1:
                    linear_memory_capacity += capacity
                else:
                    nonlinear_capacity += capacity
                
                # Accumulate by degree
                if degree not in capacity_by_degree:
                    capacity_by_degree[degree] = 0.0
                capacity_by_degree[degree] += capacity
                
                # Accumulate by delay (for single-term functions)
                spec = self.basis_functions[basis_key]
                if spec['num_delays'] == 1:
                    delay = spec['terms'][0][0]
                    if delay not in capacity_by_delay:
                        capacity_by_delay[delay] = {'linear': 0.0, 'nonlinear': 0.0, 'total': 0.0}
                    if degree == 1:
                        capacity_by_delay[delay]['linear'] += capacity
                    else:
                        capacity_by_delay[delay]['nonlinear'] += capacity
                    capacity_by_delay[delay]['total'] += capacity
                    
            except Exception as e:
                logger.error(f"Error evaluating {basis_key}: {e}")
                all_results[basis_key] = {'capacity': 0.0, 'error': str(e)}
        
        # Store results
        self.evaluation_results = {
            'total_capacity': total_capacity,
            'linear_memory_capacity': linear_memory_capacity,
            'nonlinear_capacity': nonlinear_capacity,
            'capacity_by_degree': capacity_by_degree,
            'capacity_by_delay': capacity_by_delay,
            'all_results': all_results,
            'theoretical_max': self.nodes_output.shape[1],
            'num_basis_functions': len(self.basis_functions)
        }
        
        self.capacity_by_degree = capacity_by_degree
        self.capacity_by_delay = capacity_by_delay
        
        # Log summary
        logger.output(f"\n{'='*60}")
        logger.output("INFORMATION PROCESSING CAPACITY RESULTS")
        logger.output(f"{'='*60}")
        logger.output(f"Total Capacity: {total_capacity:.4f}")
        logger.output(f"Theoretical Maximum: {self.nodes_output.shape[1]}")
        logger.output(f"Efficiency: {total_capacity/self.nodes_output.shape[1]*100:.1f}%")
        logger.output(f"\nCapacity Decomposition:")
        logger.output(f"  Linear Memory Capacity (degree=1): {linear_memory_capacity:.4f}")
        logger.output(f"  Nonlinear Capacity (degree>1): {nonlinear_capacity:.4f}")
        logger.output(f"\nCapacity by Polynomial Degree:")
        for degree in sorted(capacity_by_degree.keys()):
            logger.output(f"  Degree {degree}: {capacity_by_degree[degree]:.4f}")
        
        return self.evaluation_results
    
    def get_memory_nonlinearity_tradeoff(self) -> Dict[str, np.ndarray]:
        """
        Analyze the memory vs nonlinearity trade-off.
        
        Returns capacity organized by delay for linear (degree=1) 
        and nonlinear (degree>1) components.
        
        Returns:
        --------
        dict
            Dictionary containing:
            - delays: array of delay values
            - linear_capacity: capacity at each delay for degree=1
            - nonlinear_capacity: capacity at each delay for degree>1
            - total_capacity: total capacity at each delay
        """
        if self.evaluation_results is None:
            raise ValueError("No results available. Run calculate_total_capacity first.")
        
        delays = sorted(self.capacity_by_delay.keys())
        
        linear_cap = np.array([self.capacity_by_delay[d]['linear'] for d in delays])
        nonlinear_cap = np.array([self.capacity_by_delay[d]['nonlinear'] for d in delays])
        total_cap = np.array([self.capacity_by_delay[d]['total'] for d in delays])
        
        return {
            'delays': np.array(delays),
            'linear_capacity': linear_cap,
            'nonlinear_capacity': nonlinear_cap,
            'total_capacity': total_cap
        }
    
    def plot_results(self) -> None:
        """
        Generate plots for the IPC results.
        
        Creates:
        1. Capacity by polynomial degree (bar chart)
        2. Memory-nonlinearity trade-off (stacked area plot)
        3. Cumulative capacity curve
        """
        if self.evaluation_results is None:
            logger.warning("No results available. Run calculate_total_capacity first.")
            return
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("Matplotlib not available for plotting")
            return
        
        # Get plot config settings
        show_plot = True
        figsize = (10, 6)
        dpi = 100
        save_dir = None
        
        if self.plot_config is not None:
            show_plot = getattr(self.plot_config, 'show_plot', True)
            figsize = getattr(self.plot_config, 'figsize', (10, 6))
            dpi = getattr(self.plot_config, 'dpi', 100)
            save_dir = getattr(self.plot_config, 'save_dir', None)
        
        # 1. Capacity by polynomial degree
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        degrees = sorted(self.capacity_by_degree.keys())
        capacities = [self.capacity_by_degree[d] for d in degrees]
        
        colors = ['#2ecc71' if d == 1 else '#3498db' for d in degrees]
        bars = ax.bar(degrees, capacities, color=colors, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Polynomial Degree', fontsize=12)
        ax.set_ylabel('Total Capacity', fontsize=12)
        ax.set_title('Information Processing Capacity by Polynomial Degree', fontsize=14)
        ax.set_xticks(degrees)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, cap in zip(bars, capacities):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{cap:.2f}', ha='center', va='bottom', fontsize=10)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#2ecc71', label='Linear (d=1)'),
                         Patch(facecolor='#3498db', label='Nonlinear (d>1)')]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/ipc_by_degree.png", dpi=dpi, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # 2. Memory-nonlinearity trade-off
        tradeoff = self.get_memory_nonlinearity_tradeoff()
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        delays = tradeoff['delays']
        
        ax.fill_between(delays, 0, tradeoff['linear_capacity'], 
                        alpha=0.7, label='Linear Memory', color='#2ecc71')
        ax.fill_between(delays, tradeoff['linear_capacity'], 
                        tradeoff['linear_capacity'] + tradeoff['nonlinear_capacity'],
                        alpha=0.7, label='Nonlinear', color='#e74c3c')
        
        ax.plot(delays, tradeoff['total_capacity'], 'k-', linewidth=2, label='Total')
        
        ax.set_xlabel('Delay τ', fontsize=12)
        ax.set_ylabel('Capacity', fontsize=12)
        ax.set_title('Memory-Nonlinearity Trade-off', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
        ax.set_xlim(delays[0], delays[-1])
        ax.set_ylim(0, None)
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/ipc_tradeoff.png", dpi=dpi, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # 3. Summary comparison
        fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
        
        results = self.evaluation_results
        categories = ['Total\nCapacity', 'Linear\nMemory', 'Nonlinear\nCapacity', 'Theoretical\nMax']
        values = [results['total_capacity'], 
                 results['linear_memory_capacity'],
                 results['nonlinear_capacity'],
                 results['theoretical_max']]
        colors = ['#9b59b6', '#2ecc71', '#e74c3c', '#95a5a6']
        
        bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1)
        
        ax.set_ylabel('Capacity', fontsize=12)
        ax.set_title('Information Processing Capacity Summary', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/ipc_summary.png", dpi=dpi, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        logger.info("Plotting completed")
    
    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the IPC evaluation results.
        
        Returns:
        --------
        dict
            Summary statistics and information.
        """
        if self.evaluation_results is None:
            return {
                'status': 'No evaluation results available',
                'max_delay': self.max_delay,
                'max_degree': self.max_degree,
                'num_basis_functions': len(self.basis_functions)
            }
        
        results = self.evaluation_results
        tradeoff = self.get_memory_nonlinearity_tradeoff()
        
        return {
            'total_capacity': results['total_capacity'],
            'linear_memory_capacity': results['linear_memory_capacity'],
            'nonlinear_capacity': results['nonlinear_capacity'],
            'theoretical_max': results['theoretical_max'],
            'efficiency': results['total_capacity'] / results['theoretical_max'],
            'capacity_by_degree': results['capacity_by_degree'],
            'max_delay': self.max_delay,
            'max_degree': self.max_degree,
            'num_basis_functions': results['num_basis_functions'],
            'linear_nonlinear_ratio': (results['linear_memory_capacity'] / 
                                       results['nonlinear_capacity'] 
                                       if results['nonlinear_capacity'] > 0 else float('inf'))
        }

