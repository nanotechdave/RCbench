import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Tuple, Any
import os.path as osp
from rcbench.logger import get_logger
from rcbench.visualization.plot_config import SinxPlotConfig
from rcbench.visualization.base_plotter import BasePlotter

logger = get_logger(__name__)

class SinxPlotter(BasePlotter):
    """
    Plotter for the Sin(x) task.
    """

    def __init__(self, config: Optional[SinxPlotConfig] = None):
        """
        Initialize the Sin(x) plotter with configuration.
        
        Args:
            config (Optional[SinxPlotConfig]): Configuration for plotting
        """
        super().__init__(config or SinxPlotConfig())
        self.config = self.config if isinstance(self.config, SinxPlotConfig) else SinxPlotConfig()

    def plot_results(self, 
                    time: np.ndarray,
                    input_signal: np.ndarray,
                    node_outputs: Dict[str, np.ndarray],
                    y_true: Optional[np.ndarray] = None,
                    y_pred: Optional[np.ndarray] = None,
                    target_name: str = "sin(x)",
                    save_dir: Optional[str] = None) -> None:
        """
        Generate all relevant plots for the Sin(x) task.
        
        Args:
            time (np.ndarray): Time array 
            input_signal (np.ndarray): Input signal values
            node_outputs (Dict[str, np.ndarray]): Dictionary mapping node names to output arrays
            y_true (Optional[np.ndarray]): True target values
            y_pred (Optional[np.ndarray]): Predicted target values
            target_name (str): Name of the target
            save_dir (Optional[str]): Directory to save plots
        """
        # Always generate general reservoir property plots
        self._plot_general_plots(time, input_signal, node_outputs, save_dir)
            
        # Create general frequency analysis plots if enabled
        if self.config.plot_frequency_analysis:
            self._plot_node_frequency_analysis(time, input_signal, node_outputs, save_dir)
        
        # If target data is provided, also create target-specific plots
        if y_true is not None and y_pred is not None:
            # Create prediction plots if enabled
            if self.config.plot_target_prediction:
                self._plot_prediction_plots(time, y_true, y_pred, target_name, save_dir)
            
            # Create target-specific frequency analysis (comparing target vs prediction)
            if self.config.plot_frequency_analysis:
                # Add target and prediction to signals for frequency analysis
                pred_signals = {
                    "input": input_signal,
                    f"{target_name}_true": y_true,
                    f"{target_name}_pred": y_pred
                }
                
                # Compute frequency analysis
                pred_freqs, pred_spectra = self.compute_frequency_analysis(time, pred_signals)
                
                # Plot frequency comparison
                save_path = None if save_dir is None else osp.join(save_dir, f"{target_name}_frequency.png")
                self.plot_frequency_analysis(
                    pred_freqs, 
                    pred_spectra, 
                    f"Frequency Analysis: {target_name}",
                    save_path,
                    self.config.frequency_range
                )
    
    def _plot_general_plots(self, 
                          time: np.ndarray,
                          input_signal: np.ndarray,
                          node_outputs: Dict[str, np.ndarray],
                          save_dir: Optional[str] = None) -> None:
        """
        Plot general plots including input signal, output responses, and nonlinearity.
        
        Args:
            time (np.ndarray): Time array 
            input_signal (np.ndarray): Input signal values
            node_outputs (Dict[str, np.ndarray]): Dictionary mapping node names to output arrays
            save_dir (Optional[str]): Directory to save plots
        """
        # Plot input signal if enabled
        if self.config.plot_input_signal:
            save_path = None if save_dir is None else osp.join(save_dir, "input_signal.png")
            self.plot_input_signal(time, input_signal, "Sin(x) Task Input Signal", save_path)
            
        # Plot output responses if enabled
        if self.config.plot_output_responses:
            save_path = None if save_dir is None else osp.join(save_dir, "output_responses.png")
            self.plot_output_responses(time, node_outputs, "Reservoir Node Responses", save_path)
            
        # Plot nonlinearity if enabled
        if self.config.plot_nonlinearity:
            save_path = None if save_dir is None else osp.join(save_dir, "nonlinearity.png")
            self.plot_nonlinearity(input_signal, node_outputs, "Node Nonlinearity", save_path, 
                                  style=self.config.nonlinearity_plot_style)
    
    def _plot_prediction_plots(self,
                             time: np.ndarray,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             target_name: str,
                             save_dir: Optional[str] = None) -> None:
        """
        Plot prediction-related plots for the Sin(x) task.
        
        Args:
            time (np.ndarray): Time array
            y_true (np.ndarray): True target values
            y_pred (np.ndarray): Predicted target values
            target_name (str): Name of the target
            save_dir (Optional[str]): Directory to save plots
        """
        # Plot target vs prediction
        save_path = None if save_dir is None else osp.join(save_dir, f"{target_name}_prediction.png")
        sample_count = self.config.prediction_sample_count
        title = f"Target vs Prediction: {target_name}"
        self.plot_target_prediction(y_true, y_pred, time, title, save_path, sample_count)
    
    def _plot_node_frequency_analysis(self,
                                   time: np.ndarray,
                                   input_signal: np.ndarray,
                                   node_outputs: Dict[str, np.ndarray],
                                   save_dir: Optional[str] = None) -> None:
        """
        Plot frequency analysis for reservoir nodes (general, not target-specific).
        
        Args:
            time (np.ndarray): Time array
            input_signal (np.ndarray): Input signal values
            node_outputs (Dict[str, np.ndarray]): Dictionary mapping node names to output arrays
            save_dir (Optional[str]): Directory to save plots
        """
        # Compute frequency analysis for input and nodes
        signals = {"input": input_signal}
        # Add a subset of node outputs for analysis
        node_names = list(node_outputs.keys())
        for i, node_name in enumerate(node_names):
            if i >= 5:  # Limit to 5 nodes for clarity
                break
            signals[f"node_{node_name}"] = node_outputs[node_name]
        
        # Run frequency analysis
        frequencies, power_spectra = self.compute_frequency_analysis(time, signals)
        
        # Plot node spectra (input vs nodes)
        save_path = None if save_dir is None else osp.join(save_dir, "node_spectra.png")
        node_spectra = {key: value for key, value in power_spectra.items() if key != "input"}
        self.plot_node_spectra(
            frequencies, 
            power_spectra["input"],
            node_spectra,
            "Node Frequency Response", 
            save_path,
            self.config.frequency_range
        ) 