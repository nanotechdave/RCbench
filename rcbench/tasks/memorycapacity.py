import numpy as np
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Union, Any, Optional
from rcbench.tasks.baseevaluator import BaseEvaluator
from rcbench.tasks.featureselector import FeatureSelector
from rcbench.logger import get_logger
from rcbench.tasks.c_metrics import evaluate_mc
from rcbench.visualization.mc_plotter import MCPlotter
from dataclasses import dataclass
from typing import Optional

logger = get_logger(__name__)

@dataclass
class PlotConfig:
    """Configuration for plotting options."""
    save_dir: Optional[str] = None
    plot_mc_vs_delay: bool = True
    plot_feature_importance: bool = True
    plot_prediction_results: bool = True
    plot_cumulative_mc: bool = True
    plot_mc_heatmap: bool = True
    figsize: tuple = (10, 6)

class MemoryCapacityEvaluator(BaseEvaluator):
    def __init__(self, 
                 input_signal: np.ndarray, 
                 nodes_output: np.ndarray, 
                 max_delay: int = 30,
                 random_state: int = 42,
                 electrode_names: Optional[List[str]] = None) -> None:
        """
        Initializes the Memory Capacity evaluator.

        Parameters:
        - input_signal (np.ndarray): Input stimulation signal array.
        - nodes_output (np.ndarray): Reservoir node output (features).
        - max_delay (int): Maximum delay steps to evaluate.
        - random_state (int): Random seed for reproducibility.
        - electrode_names (Optional[List[str]]): Names of electrodes for plotting.
        """
        super().__init__(input_signal, nodes_output, electrode_names)
        self.max_delay: int = max_delay
        self.random_state = random_state
        self.targets = self.target_generator()
        self.plotter = MCPlotter()
        self.evaluation_results = None
        self.mc_matrix = None
        
        # Store electrode names if provided, otherwise create default ones
        if electrode_names is None:
            self.electrode_names = [f'Electrode {i}' for i in range(nodes_output.shape[1])]
        else:
            self.electrode_names = electrode_names

    def target_generator(self) -> Dict[int, np.ndarray]:
        """
        Generates delayed versions of the input signal.

        Returns:
        - dict: delay (int) -> delayed input (np.ndarray)
        """
        targets = {}
        for delay in range(1, self.max_delay + 1):
            targets[delay] = np.roll(self.input_signal, delay)
        return targets
    
    def run_evaluation(self,
                       delay: int,
                       regression_alpha: float = 1.0,
                       train_ratio: float = 0.8
                       ) -> Dict[str, Any]:
        """
        Run evaluation for a specific delay.
        
        Args:
            delay (int): Delay to evaluate
            regression_alpha (float): Ridge regression alpha parameter
            train_ratio (float): Ratio of data to use for training
        """
        if delay not in self.targets:
            raise ValueError(f"Delay '{delay}' not available.")

        target_waveform = self.targets[delay]
        
        # Adjust data for delay
        X = self.nodes_output[delay:]
        y = target_waveform[delay:]

        # Split data
        split_idx = int(train_ratio * len(y))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Apply feature selection to training and test data
        X_train_selected = self.apply_feature_selection(X_train)
        X_test_selected = self.apply_feature_selection(X_test)

        # Regression model
        model = Ridge(alpha=regression_alpha, random_state=self.random_state)
        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)

        # Evaluate MC
        mc = evaluate_mc(y_test, y_pred)

        result = {
            'delay': delay,
            'memory_capacity': mc,
            'model': model,
            'y_pred': y_pred,
            'y_test': y_test,
        }

        return result

    def calculate_total_memory_capacity(self,
                                      feature_selection_method: str = 'pca',
                                      num_features: int = 10,
                                      regression_alpha: float = 1.0,
                                      train_ratio: float = 0.8
                                      ) -> Dict[str, Union[float, Dict[int, float]]]:
        """
        Calculate total memory capacity across all delays.
        """
        # Set random seed for reproducibility
        np.random.seed(self.random_state)
        
        # Initialize a new feature selector with our random state
        self.feature_selector = FeatureSelector(random_state=self.random_state)
        
        # Perform feature selection once using the first delay's data
        first_delay = 1
        X = self.nodes_output[first_delay:]
        y = self.targets[first_delay][first_delay:]
        
        # Split data for feature selection
        split_idx = int(train_ratio * len(y))
        X_train, _, y_train, _ = self.split_train_test(X, y, train_ratio)
        
        # Feature selection using BaseEvaluator method
        self.feature_selection(
            X_train, y_train,
            method=feature_selection_method,
            num_features=num_features
        )
        
        # Log the selected electrodes
        logger.info(f"Selected electrodes: {self.selected_feature_names}")

        # Calculate memory capacity for all delays
        total_mc = 0
        delay_results = {}
        all_evaluation_results = {}
        
        # Create MC matrix for heatmap
        self.mc_matrix = np.zeros((self.max_delay, self.nodes_output.shape[1]))
        
        # Run evaluation for all delays
        for delay in range(1, self.max_delay + 1):
            result = self.run_evaluation(
                delay=delay,
                regression_alpha=regression_alpha,
                train_ratio=train_ratio
            )
            total_mc += result['memory_capacity']
            delay_results[delay] = result['memory_capacity']
            all_evaluation_results[delay] = result
            
            # Update MC matrix for heatmap
            self.mc_matrix[delay-1] = result['memory_capacity']

        self.evaluation_results = {
            'total_memory_capacity': total_mc,
            'delay_results': delay_results,
            'all_results': all_evaluation_results
        }

        return self.evaluation_results

    def plot_results(self, plot_config: PlotConfig) -> None:
        """
        Generate plots for the evaluation results.
        
        Args:
            plot_config (PlotConfig): Configuration for plotting options
        """
        if self.evaluation_results is None:
            logger.warning("No evaluation results available. Run calculate_total_memory_capacity first.")
            return

        delay_results = self.evaluation_results['delay_results']
        all_results = self.evaluation_results['all_results']  # Get stored results
        
        if plot_config.plot_mc_vs_delay:
            self.plotter.plot_mc_vs_delay(
                delay_results,
                save_path=f"{plot_config.save_dir}/mc_vs_delay.png" if plot_config.save_dir else None
            )
        
        if plot_config.plot_feature_importance and self.selected_features is not None:
            # We now have the actual electrode names
            feature_names = self.selected_feature_names
            
            # Get feature importance from the feature selector
            feature_importance = self.feature_selector.get_feature_importance()
            
            # Get importance scores for selected features
            importance_scores = np.array([feature_importance[name] for name in feature_names])
            
            self.plotter.plot_feature_importance(
                importance_scores,
                feature_names=feature_names,
                title=f'Feature Importance ({self.feature_selection_method})',
                save_path=f"{plot_config.save_dir}/feature_importance.png" if plot_config.save_dir else None
            )
        
        if plot_config.plot_cumulative_mc:
            self.plotter.plot_cumulative_mc(
                delay_results,
                save_path=f"{plot_config.save_dir}/cumulative_mc.png" if plot_config.save_dir else None
            )
        
        if plot_config.plot_mc_heatmap and self.mc_matrix is not None:
            self.plotter.plot_mc_heatmap(
                self.mc_matrix,
                range(1, self.max_delay + 1),
                range(self.nodes_output.shape[1]),
                save_path=f"{plot_config.save_dir}/mc_heatmap.png" if plot_config.save_dir else None
            )
        
        if plot_config.plot_prediction_results:
            # Use stored results instead of recomputing
            for delay in range(1, self.max_delay + 1):
                result = all_results[delay]  # Use stored result instead of recomputing
                self.plotter.plot_prediction_results(
                    y_true=result['y_test'],
                    y_pred=result['y_pred'],
                    title=f'Prediction Results for Delay {delay}',
                    save_path=f"{plot_config.save_dir}/prediction_delay_{delay}.png" if plot_config.save_dir else None
                )
