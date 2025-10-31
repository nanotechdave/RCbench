import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error
from typing import Dict, List, Tuple, Union, Any, Optional
from rcbench.tasks.baseevaluator import BaseEvaluator
from rcbench.tasks.featureselector import FeatureSelector
from rcbench.logger import get_logger
from rcbench.visualization.plot_config import BasePlotConfig

logger = get_logger(__name__)


class NonlinearMemoryEvaluator(BaseEvaluator):
    """
    Nonlinear Memory Benchmark Evaluator.
    
    This benchmark evaluates the memory-nonlinearity trade-off by computing:
        y(t) = sin(ν * s(t - τ))
    
    where:
        s(t) = input signal
        τ (tau) = delay in time steps (controls memory depth)
        ν (nu) = nonlinearity strength parameter
    
    The task difficulty is tuned by:
        - Larger τ → longer memory required
        - Larger ν → stronger nonlinearity required
    """
    
    def __init__(self,
                 input_signal: Union[np.ndarray, List[float]],
                 nodes_output: np.ndarray,
                 tau_values: Optional[List[int]] = None,
                 nu_values: Optional[List[float]] = None,
                 random_state: int = 42,
                 node_names: Optional[List[str]] = None,
                 plot_config: Optional[BasePlotConfig] = None) -> None:
        """
        Initialize the Nonlinear Memory evaluator.
        
        Parameters:
        -----------
        input_signal : array-like
            Input stimulation signal array.
        nodes_output : np.ndarray
            Reservoir node output (features), shape [time_steps, n_nodes].
        tau_values : List[int], optional
            List of delay values to evaluate. Default: [1, 2, 3, 4, 5, 6, 7, 8]
        nu_values : List[float], optional
            List of nonlinearity strength values to evaluate.
            Default: [0.1, 0.3, 1.0, 3.0, 10.0]
        random_state : int
            Random seed for reproducibility.
        node_names : List[str], optional
            Names of the nodes.
        plot_config : BasePlotConfig, optional
            Configuration for plotting.
        """
        super().__init__(input_signal, nodes_output, node_names)
        
        # Set default parameter ranges if not provided
        self.tau_values = tau_values if tau_values is not None else [1, 2, 3, 4, 5, 6, 7, 8]
        self.nu_values = nu_values if nu_values is not None else [0.1, 0.3, 1.0, 3.0, 10.0]
        
        self.random_state = random_state
        self.max_tau = max(self.tau_values)
        
        # Generate all targets for the parameter sweep
        self.targets = self.target_generator()
        
        # Storage for results
        self.evaluation_results = None
        self.capacity_matrix = None  # Will store C(τ, ν) values
        
        # Plotting configuration
        self.plot_config = plot_config
        
        logger.info(f"Initialized Nonlinear Memory Evaluator")
        logger.info(f"  τ (delay) values: {self.tau_values}")
        logger.info(f"  ν (nonlinearity) values: {self.nu_values}")
        logger.info(f"  Total parameter combinations: {len(self.tau_values) * len(self.nu_values)}")
    
    def target_generator(self) -> Dict[Tuple[int, float], np.ndarray]:
        """
        Generate target signals for all (τ, ν) parameter combinations.
        
        For each combination, compute:
            y(t) = sin(ν * s(t - τ))
        
        For t < τ, set y(t) = 0 (or they will be excluded during training/testing).
        
        Returns:
        --------
        dict : {(tau, nu): target_array}
            Dictionary mapping (τ, ν) tuples to target arrays.
        """
        targets = {}
        
        for tau in self.tau_values:
            for nu in self.nu_values:
                # Create delayed input signal
                delayed_signal = np.roll(self.input_signal, tau)
                
                # Set first tau values to zero (invalid region)
                delayed_signal[:tau] = 0
                
                # Apply nonlinear transformation
                target = np.sin(nu * delayed_signal)
                
                # Store with key (tau, nu)
                targets[(tau, nu)] = target
                
        logger.info(f"Generated {len(targets)} target signals for parameter sweep")
        return targets
    
    def evaluate_metric(self,
                       y_true: np.ndarray,
                       y_pred: np.ndarray,
                       metric: str = 'NMSE') -> float:
        """
        Evaluate the performance using the specified metric.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True target values
        y_pred : np.ndarray
            Predicted values
        metric : str
            Metric to use: 'NMSE', 'RNMSE', 'MSE', or 'Capacity'
        
        Returns:
        --------
        float
            Metric value
        """
        if metric == 'NMSE':
            var_y = np.var(y_true)
            if var_y == 0:
                return np.nan
            return np.mean((y_true - y_pred) ** 2) / var_y
        
        elif metric == 'RNMSE':
            var_y = np.var(y_true)
            if var_y == 0:
                return np.nan
            return np.sqrt(np.mean((y_true - y_pred) ** 2) / var_y)
        
        elif metric == 'MSE':
            return mean_squared_error(y_true, y_pred)
        
        elif metric == 'Capacity':
            # Capacity = 1 - NMSE
            var_y = np.var(y_true)
            if var_y == 0:
                return 0.0
            nmse = np.mean((y_true - y_pred) ** 2) / var_y
            return max(0.0, 1.0 - nmse)  # Capacity is bounded [0, 1]
        
        else:
            raise ValueError("Unsupported metric: choose 'NMSE', 'RNMSE', 'MSE', or 'Capacity'")
    
    def run_evaluation(self,
                      tau: int,
                      nu: float,
                      modeltype: str = "Ridge",
                      regression_alpha: float = 1.0,
                      train_ratio: float = 0.8,
                      metric: str = 'NMSE') -> Dict[str, Any]:
        """
        Run evaluation for a specific (τ, ν) parameter combination.
        
        Parameters:
        -----------
        tau : int
            Delay parameter (memory depth)
        nu : float
            Nonlinearity strength parameter
        modeltype : str
            Type of regression model ('Ridge' or 'Linear')
        regression_alpha : float
            Regularization parameter for Ridge regression
        train_ratio : float
            Ratio of data to use for training
        metric : str
            Metric to evaluate ('NMSE', 'RNMSE', 'MSE', or 'Capacity')
        
        Returns:
        --------
        dict
            Dictionary containing evaluation results
        """
        # Get target for this (tau, nu) combination
        if (tau, nu) not in self.targets:
            raise ValueError(f"Parameter combination (τ={tau}, ν={nu}) not found in targets")
        
        target = self.targets[(tau, nu)]
        
        # Use data from max_tau onwards to ensure valid data for all delays
        start_idx = self.max_tau
        data_length = len(self.input_signal) - self.max_tau
        
        # Extract data windows
        X = self.nodes_output[start_idx:start_idx + data_length]
        y = target[start_idx:start_idx + data_length]
        
        # Split into train/test
        split_idx = int(train_ratio * data_length)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Apply feature selection if configured
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
        
        # Evaluate performance
        error = self.evaluate_metric(y_test, y_pred, metric)
        
        # Also compute capacity for storage
        capacity = self.evaluate_metric(y_test, y_pred, 'Capacity')
        
        result = {
            'tau': tau,
            'nu': nu,
            'error': error,
            'capacity': capacity,
            'metric': metric,
            'model': model,
            'y_pred': y_pred,
            'y_test': y_test,
            'train_ratio': train_ratio
        }
        
        return result
    
    def run_parameter_sweep(self,
                           feature_selection_method: str = 'kbest',
                           num_features: Union[int, str] = 'all',
                           modeltype: str = "Ridge",
                           regression_alpha: float = 1.0,
                           train_ratio: float = 0.8,
                           metric: str = 'NMSE') -> Dict[str, Any]:
        """
        Run evaluation across all (τ, ν) parameter combinations.
        
        This method performs the complete benchmark sweep to map the
        memory-nonlinearity trade-off surface.
        
        Parameters:
        -----------
        feature_selection_method : str
            Method for feature selection ('kbest', 'pca', etc.)
        num_features : int or str
            Number of features to use, or 'all'
        modeltype : str
            Type of regression model ('Ridge' or 'Linear')
        regression_alpha : float
            Regularization parameter for Ridge regression
        train_ratio : float
            Ratio of data to use for training
        metric : str
            Metric to evaluate ('NMSE', 'RNMSE', 'MSE', or 'Capacity')
        
        Returns:
        --------
        dict
            Dictionary containing:
                - 'results': dict of {(tau, nu): result_dict}
                - 'capacity_matrix': 2D array of capacity values C(τ, ν)
                - 'error_matrix': 2D array of error values E(τ, ν)
                - 'tau_values': list of τ values
                - 'nu_values': list of ν values
        """
        logger.info("Starting nonlinear memory parameter sweep...")
        
        # Set random seed for reproducibility
        np.random.seed(self.random_state)
        
        # Initialize feature selector
        self.feature_selector = FeatureSelector(random_state=self.random_state)
        
        # Perform feature selection once using the first (tau, nu) combination
        first_tau, first_nu = self.tau_values[0], self.nu_values[0]
        first_target = self.targets[(first_tau, first_nu)]
        
        # Use data from max_tau onwards for feature selection
        X = self.nodes_output[self.max_tau:]
        y = first_target[self.max_tau:]
        
        # Split data for feature selection
        split_idx = int(train_ratio * len(y))
        X_train, _, y_train, _ = self.split_train_test(X, y, train_ratio)
        
        # Feature selection using BaseEvaluator method
        self.feature_selection(
            X_train, y_train,
            method=feature_selection_method,
            num_features=num_features
        )
        
        logger.info(f"Selected features: {self.selected_feature_names}")
        
        # Initialize result storage
        results = {}
        capacity_matrix = np.zeros((len(self.tau_values), len(self.nu_values)))
        error_matrix = np.zeros((len(self.tau_values), len(self.nu_values)))
        
        # Run evaluation for each (tau, nu) combination
        total_combinations = len(self.tau_values) * len(self.nu_values)
        current = 0
        
        for i, tau in enumerate(self.tau_values):
            for j, nu in enumerate(self.nu_values):
                current += 1
                logger.info(f"Evaluating ({current}/{total_combinations}): τ={tau}, ν={nu}")
                
                try:
                    result = self.run_evaluation(
                        tau=tau,
                        nu=nu,
                        modeltype=modeltype,
                        regression_alpha=regression_alpha,
                        train_ratio=train_ratio,
                        metric=metric
                    )
                    
                    results[(tau, nu)] = result
                    capacity_matrix[i, j] = result['capacity']
                    error_matrix[i, j] = result['error']
                    
                    logger.info(f"  → Error: {result['error']:.6f}, Capacity: {result['capacity']:.6f}")
                    
                except Exception as e:
                    logger.error(f"Error evaluating (τ={tau}, ν={nu}): {str(e)}")
                    results[(tau, nu)] = {'error': np.nan, 'capacity': 0.0, 'exception': str(e)}
                    capacity_matrix[i, j] = 0.0
                    error_matrix[i, j] = np.nan
        
        # Store results
        self.evaluation_results = {
            'results': results,
            'capacity_matrix': capacity_matrix,
            'error_matrix': error_matrix,
            'tau_values': self.tau_values,
            'nu_values': self.nu_values,
            'metric': metric,
            'feature_selection_method': feature_selection_method,
            'num_features': num_features if num_features != 'all' else len(self.selected_features),
            'selected_features': self.selected_feature_names
        }
        
        self.capacity_matrix = capacity_matrix
        
        logger.info("Parameter sweep completed!")
        logger.info(f"  Average capacity: {np.nanmean(capacity_matrix):.4f}")
        logger.info(f"  Max capacity: {np.nanmax(capacity_matrix):.4f}")
        logger.info(f"  Min capacity: {np.nanmin(capacity_matrix):.4f}")
        
        return self.evaluation_results
    
    def get_best_performance(self) -> Dict[str, Any]:
        """
        Get the (τ, ν) combination with the best performance (highest capacity).
        
        Returns:
        --------
        dict
            Dictionary with best parameters and their results
        """
        if self.evaluation_results is None:
            raise ValueError("No evaluation results available. Run run_parameter_sweep first.")
        
        capacity_matrix = self.evaluation_results['capacity_matrix']
        
        # Find maximum capacity
        best_idx = np.unravel_index(np.nanargmax(capacity_matrix), capacity_matrix.shape)
        best_tau = self.tau_values[best_idx[0]]
        best_nu = self.nu_values[best_idx[1]]
        best_capacity = capacity_matrix[best_idx]
        
        best_result = self.evaluation_results['results'][(best_tau, best_nu)]
        
        return {
            'tau': best_tau,
            'nu': best_nu,
            'capacity': best_capacity,
            'error': best_result['error'],
            'full_result': best_result
        }
    
    def get_memory_vs_nonlinearity_tradeoff(self) -> Dict[str, np.ndarray]:
        """
        Analyze the memory vs nonlinearity trade-off.
        
        Returns:
        --------
        dict
            Dictionary containing:
                - 'memory_performance': average capacity for each τ (averaged over ν)
                - 'nonlinearity_performance': average capacity for each ν (averaged over τ)
        """
        if self.evaluation_results is None:
            raise ValueError("No evaluation results available. Run run_parameter_sweep first.")
        
        capacity_matrix = self.evaluation_results['capacity_matrix']
        
        # Average capacity across ν for each τ (memory performance)
        memory_performance = np.nanmean(capacity_matrix, axis=1)
        
        # Average capacity across τ for each ν (nonlinearity performance)
        nonlinearity_performance = np.nanmean(capacity_matrix, axis=0)
        
        return {
            'memory_performance': memory_performance,
            'nonlinearity_performance': nonlinearity_performance,
            'tau_values': self.tau_values,
            'nu_values': self.nu_values
        }
    
    def plot_results(self, save_dir: Optional[str] = None) -> None:
        """
        Generate plots for the nonlinear memory benchmark results.
        
        Creates:
        1. Heatmap of capacity C(τ, ν)
        2. Memory vs delay plot (averaged over ν)
        3. Nonlinearity performance plot (averaged over τ)
        4. Selected parameter combinations predictions
        
        Parameters:
        -----------
        save_dir : str, optional
            Directory to save plots. If None, plots are shown but not saved.
        """
        if self.evaluation_results is None:
            logger.warning("No evaluation results available. Run run_parameter_sweep first.")
            return
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("Matplotlib not available for plotting")
            return
        
        capacity_matrix = self.evaluation_results['capacity_matrix']
        tau_values = self.evaluation_results['tau_values']
        nu_values = self.evaluation_results['nu_values']
        
        # 1. Heatmap of C(τ, ν)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(capacity_matrix, aspect='auto', origin='lower', cmap='viridis')
        ax.set_xticks(range(len(nu_values)))
        ax.set_yticks(range(len(tau_values)))
        ax.set_xticklabels([f'{nu:.1f}' for nu in nu_values])
        ax.set_yticklabels([f'{tau}' for tau in tau_values])
        ax.set_xlabel('ν (Nonlinearity strength)', fontsize=12)
        ax.set_ylabel('τ (Delay)', fontsize=12)
        ax.set_title('Capacity C(τ, ν): Memory-Nonlinearity Trade-off', fontsize=14)
        plt.colorbar(im, ax=ax, label='Capacity')
        
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'capacity_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
        # 2. Trade-off analysis
        tradeoff = self.get_memory_vs_nonlinearity_tradeoff()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Memory performance (averaged over ν)
        ax1.plot(tau_values, tradeoff['memory_performance'], 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('τ (Delay)', fontsize=12)
        ax1.set_ylabel('Average Capacity', fontsize=12)
        ax1.set_title('Memory Performance (averaged over ν)', fontsize=13)
        ax1.grid(True, alpha=0.3)
        
        # Nonlinearity performance (averaged over τ)
        ax2.plot(nu_values, tradeoff['nonlinearity_performance'], 's-', linewidth=2, markersize=8, color='orange')
        ax2.set_xlabel('ν (Nonlinearity strength)', fontsize=12)
        ax2.set_ylabel('Average Capacity', fontsize=12)
        ax2.set_title('Nonlinearity Performance (averaged over τ)', fontsize=13)
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'tradeoff_analysis.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
        # 3. Show best and worst cases
        best = self.get_best_performance()
        logger.output(f"\nBest performance:")
        logger.output(f"  τ={best['tau']}, ν={best['nu']}")
        logger.output(f"  Capacity: {best['capacity']:.4f}")
        logger.output(f"  Error: {best['error']:.6f}")
        
        logger.info("Plotting completed successfully")
    
    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the evaluation results.
        
        Returns:
        --------
        dict
            Summary statistics and information
        """
        if self.evaluation_results is None:
            return {
                'status': 'No evaluation results available',
                'tau_values': self.tau_values,
                'nu_values': self.nu_values,
                'total_combinations': len(self.tau_values) * len(self.nu_values)
            }
        
        capacity_matrix = self.evaluation_results['capacity_matrix']
        best = self.get_best_performance()
        tradeoff = self.get_memory_vs_nonlinearity_tradeoff()
        
        return {
            'tau_values': self.tau_values,
            'nu_values': self.nu_values,
            'total_combinations': len(self.tau_values) * len(self.nu_values),
            'average_capacity': float(np.nanmean(capacity_matrix)),
            'max_capacity': float(np.nanmax(capacity_matrix)),
            'min_capacity': float(np.nanmin(capacity_matrix)),
            'best_tau': best['tau'],
            'best_nu': best['nu'],
            'best_capacity': best['capacity'],
            'memory_performance': tradeoff['memory_performance'].tolist(),
            'nonlinearity_performance': tradeoff['nonlinearity_performance'].tolist(),
            'selected_features': self.evaluation_results.get('selected_features', [])
        }

