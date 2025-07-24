import numpy as np
from rcbench.tasks.baseevaluator import BaseEvaluator
from rcbench.tasks.featureselector import FeatureSelector
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from rcbench.logger import get_logger
from typing import Dict, List, Union, Optional, Any
from rcbench.visualization.sinx_plotter import SinxPlotter
from rcbench.visualization.plot_config import SinxPlotConfig

logger = get_logger(__name__)
class SinxEvaluator(BaseEvaluator):
    def __init__(self, 
                 input_signal: Union[np.ndarray, List[float]], 
                 nodes_output: np.ndarray,
                 node_names: Optional[List[str]] = None,
                 plot_config: Optional[SinxPlotConfig] = None
                 ) -> None:
        """
        Initializes the SinxEvaluator for approximating sin(normalized_input).

        Parameters:
        - input_signal (np.ndarray): Random white noise input signal.
        - nodes_output (np.ndarray): Reservoir node voltages (features).
        - node_names (Optional[List[str]]): Names of nodes.
        - plot_config (Optional[SinxPlotConfig]): Configuration for plotting.
        """
        super().__init__(input_signal, nodes_output, node_names)
        self.normalized_input = self._normalize_input(self.input_signal)
        self.target = np.sin(self.normalized_input)
        
        # Initialize plotter with provided config
        self.plotter = SinxPlotter(config=plot_config)

    def _normalize_input(self, x: np.ndarray) -> np.ndarray:
        """Normalize input to range [0, 2π]."""
        x_min = np.min(x)
        x_max = np.max(x)
        return 2 * np.pi * (x - x_min) / (x_max - x_min)

    def evaluate_metric(self, 
                        y_true: np.ndarray, 
                        y_pred: np.ndarray, 
                        metric: str = 'NMSE',
                        ) -> float:
        if metric == 'NMSE':
            return np.mean((y_true - y_pred) ** 2) / np.var(y_true)
        elif metric == 'RNMSE':
            return np.sqrt(np.mean((y_true - y_pred) ** 2) / np.var(y_true))
        elif metric == 'MSE':
            return mean_squared_error(y_true, y_pred)
        else:
            raise ValueError("Unsupported metric: choose 'NMSE', 'RNMSE', or 'MSE'")

    def run_evaluation(self,
                       metric: str = 'NMSE',
                       feature_selection_method: str = 'kbest',
                       num_features: Union[str, int] = 10,
                       modeltype="Ridge",
                       regression_alpha: float = 1.0,
                       train_ratio: float = 0.8,
                       ) -> Dict[str, Any]:
        """
        Run the sin(x) reconstruction task using reservoir outputs.

        Returns:
        - dict: result dictionary including accuracy, predictions, model, etc.
        """
        X = self.nodes_output
        y = self.target

        # Train/test split
        X_train, X_test, y_train, y_test = self.split_train_test(X, y, train_ratio)

        # Feature selection
        X_train_sel, selected_features, _ = self.feature_selection(X_train, y_train, feature_selection_method, num_features)
        if feature_selection_method == 'kbest':
            X_test_sel = X_test[:, selected_features]
        else:
            X_test_sel = self.apply_feature_selection(X_test)

        # Regression
        model = self.train_regression(X_train_sel, y_train, modeltype, alpha=regression_alpha)
        y_pred = model.predict(X_test_sel)

        accuracy = self.evaluate_metric(y_test, y_pred, metric)

        return {
            'accuracy': accuracy,
            'metric': metric,
            'selected_features': selected_features,
            'model': model,
            'y_pred': y_pred,
            'y_test': y_test,
            'train_ratio': train_ratio
        }
        
    def plot_results(self, existing_results: Optional[Dict[str, Any]] = None) -> None:
        """
        Generate plots for the sin(x) evaluation results.
        
        Args:
            existing_results (Optional[Dict[str, Any]]): Results from a previous run_evaluation call.
                                                        If None, run_evaluation will be called.
        """
        # Run evaluation if results not provided
        if existing_results is None:
            results = self.run_evaluation()
        else:
            results = existing_results
            
        # Create a time array for plotting
        time = np.arange(len(self.input_signal))
        
        # Create node outputs dictionary for visualization
        node_outputs = {}
        for i, name in enumerate(self.node_names):
            node_outputs[name] = self.nodes_output[:, i]
            
        # Get test data from results
        y_test = results['y_test']
        y_pred = results['y_pred']
        
        # Calculate starting index for test data based on train_ratio
        train_ratio = results.get('train_ratio', 0.8)
        test_start_idx = int(train_ratio * len(self.input_signal))
        
        # Create time array for test data
        test_time = time[test_start_idx:test_start_idx + len(y_test)]
        
        # First create general plots with the full dataset (not target-specific)
        self.plotter.plot_results(
            time=time,
            input_signal=self.input_signal,
            node_outputs=node_outputs,
            save_dir=None
        )
        
        # Then create prediction plots with the test data subset
        if y_test is not None and y_pred is not None:
            # Calculate the test portion of the input signal
            test_input_signal = self.input_signal[test_start_idx:test_start_idx + len(y_test)]
            
            # Create a dictionary for test node outputs
            test_node_outputs = {}
            for name, output in node_outputs.items():
                test_node_outputs[name] = output[test_start_idx:test_start_idx + len(y_test)]
            
            # Now generate the target-specific plots with properly aligned time arrays
            self.plotter.plot_results(
                time=test_time,
                input_signal=test_input_signal,
                node_outputs=test_node_outputs,
                y_true=y_test,
                y_pred=y_pred,
                target_name="sin(x)",
                save_dir=None
            )
