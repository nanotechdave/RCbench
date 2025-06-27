import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union
from rcbench.logger import get_logger
from rcbench.visualization.plot_config import MCPlotConfig
from rcbench.visualization.base_plotter import BasePlotter

logger = get_logger(__name__)

class MCPlotter(BasePlotter):
    """Class for visualizing Memory Capacity evaluation results."""
    
    def __init__(self, config: Optional[MCPlotConfig] = None):
        """
        Initialize the MCPlotter.
        
        Args:
            config (Optional[MCPlotConfig]): Configuration for the plotter
        """
        super().__init__(config or MCPlotConfig())
    
    def plot_mc_vs_delay(self, 
                        delay_results: Dict[int, float],
                        title: Optional[str] = None,
                        save_path: Optional[str] = None) -> None:
        """
        Plot Memory Capacity as a function of delay.
        
        Args:
            delay_results (Dict[int, float]): Dictionary mapping delays to MC values
            title (Optional[str]): Plot title
            save_path (Optional[str]): Path to save the plot
        """
        delays = list(delay_results.keys())
        mc_values = list(delay_results.values())
        
        self._create_figure()
        plt.plot(delays, mc_values, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Delay (k)')
        plt.ylabel('Memory Capacity')
        plt.title(title or 'Memory Capacity vs Delay')
        plt.grid(True)
        
        self._save_or_show(save_path)
    
    def plot_feature_importance(self,
                              feature_importance: np.ndarray,
                              feature_names: Optional[List[str]] = None,
                              title: Optional[str] = None,
                              save_path: Optional[str] = None) -> None:
        """
        Plot feature importance scores.
        
        Args:
            feature_importance (np.ndarray): Array of feature importance scores
            feature_names (Optional[List[str]]): List of feature names
            title (Optional[str]): Plot title
            save_path (Optional[str]): Path to save the plot
        """
        n_features = len(feature_importance)
        if feature_names is None:
            feature_names = [f'Feature {i+1}' for i in range(n_features)]
        
        # Sort features by importance for consistent display
        sorted_indices = np.argsort(feature_importance)[::-1]  # Sort in descending order
        sorted_importance = feature_importance[sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]
        
        plt.figure(figsize=self.figsize)
        plt.bar(range(n_features), sorted_importance)
        plt.xticks(range(n_features), sorted_names, rotation=45)
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.title(title or 'Feature Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi)
            plt.close()
        elif self.config.show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_prediction_results(self,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             time: Optional[np.ndarray] = None,
                             title: Optional[str] = None,
                             save_path: Optional[str] = None,
                             sample_count: Optional[int] = None) -> None:
        """
        Plot true vs predicted values for a specific delay.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            time (Optional[np.ndarray]): Time array for x-axis
            title (Optional[str]): Plot title
            save_path (Optional[str]): Path to save the plot
            sample_count (Optional[int]): Number of samples to show (for large datasets)
        """
        # For memory capacity plots, adjust the time axis to show the shift relationship
        # Extract delay number from title
        if title and 'Delay' in title:
            try:
                delay_value = int(title.split()[-1])
                if time is not None:
                    # Shift the time axis by the delay amount to show the temporal relationship
                    # This makes it clear that each delay is the same waveform shifted in time
                    time = time - delay_value
            except (ValueError, IndexError):
                pass  # If we can't parse delay, use original time
        
        sample_count = sample_count or self.config.prediction_sample_count
        
        # For memory capacity plots, disable sampling to preserve shift relationships
        # The sampling destroys the temporal alignment between different delays
        if title and 'Delay' in title:
            sample_count = None  # Disable sampling for MC plots
        
        self.plot_target_prediction(
            y_true, 
            y_pred, 
            time=time, 
            title=title, 
            save_path=save_path,
            sample_count=sample_count,
            x_label='Time',
            y_label='Value'
        )
    
    def plot_cumulative_mc(self,
                          delay_results: Dict[int, float],
                          title: Optional[str] = None,
                          save_path: Optional[str] = None) -> None:
        """
        Plot cumulative Memory Capacity as a function of delay.
        
        Args:
            delay_results (Dict[int, float]): Dictionary mapping delays to MC values
            title (Optional[str]): Plot title
            save_path (Optional[str]): Path to save the plot
        """
        delays = list(delay_results.keys())
        mc_values = list(delay_results.values())
        cumulative_mc = np.cumsum(mc_values)
        
        self._create_figure()
        plt.plot(delays, cumulative_mc, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Delay (k)')
        plt.ylabel('Cumulative Memory Capacity')
        plt.title(title or 'Cumulative Memory Capacity vs Delay')
        plt.grid(True)
        
        self._save_or_show(save_path)
    
    def plot_mc_heatmap(self,
                       mc_matrix: np.ndarray,
                       delay_range: range,
                       feature_range: range,
                       title: Optional[str] = None,
                       save_path: Optional[str] = None,
                       colormap: Optional[str] = None) -> None:
        """
        Plot Memory Capacity as a heatmap for different delays and features.
        
        Args:
            mc_matrix (np.ndarray): 2D array of MC values
            delay_range (range): Range of delays
            feature_range (range): Range of features
            title (Optional[str]): Plot title
            save_path (Optional[str]): Path to save the plot
            colormap (Optional[str]): Matplotlib colormap to use, defaults to config value
        """
        colormap = colormap or self.config.heatmap_colormap
        
        self.plot_heatmap(
            data=mc_matrix,
            x_labels=feature_range,
            y_labels=delay_range,
            title=title or 'Memory Capacity Heatmap',
            save_path=save_path,
            colormap=colormap,
            x_label='Features',
            y_label='Delay (k)',
            colorbar_label='Memory Capacity'
        )
    
    def plot_all(self, 
                delay_results: Dict[int, float],
                feature_importance: np.ndarray,
                y_true: np.ndarray,
                y_pred: np.ndarray,
                mc_matrix: Optional[np.ndarray] = None,
                delay_range: Optional[range] = None,
                feature_range: Optional[range] = None,
                feature_names: Optional[List[str]] = None) -> None:
        """
        Generate all enabled plots based on configuration.
        
        Args:
            delay_results: Dictionary mapping delays to MC values
            feature_importance: Array of feature importance scores
            y_true: True values for prediction plot
            y_pred: Predicted values for prediction plot
            mc_matrix: 2D array of MC values for heatmap
            delay_range: Range of delays for heatmap
            feature_range: Range of features for heatmap
            feature_names: List of feature names for importance plot
        """
        # Create plots based on config settings
        if self.config.plot_mc_vs_delay:
            self.plot_mc_vs_delay(
                delay_results,
                save_path=self.config.get_save_path("mc_vs_delay.png")
            )
        
        if self.config.plot_feature_importance:
            self.plot_feature_importance(
                feature_importance,
                feature_names=feature_names,
                save_path=self.config.get_save_path("feature_importance.png")
            )
        
        if self.config.plot_cumulative_mc:
            self.plot_cumulative_mc(
                delay_results,
                save_path=self.config.get_save_path("cumulative_mc.png")
            )
        
        if self.config.plot_mc_heatmap and mc_matrix is not None:
            self.plot_mc_heatmap(
                mc_matrix,
                delay_range or range(mc_matrix.shape[0]),
                feature_range or range(mc_matrix.shape[1]),
                save_path=self.config.get_save_path("mc_heatmap.png")
            )
        
        if self.config.plot_prediction_results:
            self.plot_prediction_results(
                y_true,
                y_pred,
                save_path=self.config.get_save_path("prediction_results.png")
            )
