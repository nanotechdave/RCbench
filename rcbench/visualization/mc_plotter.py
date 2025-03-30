import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union
from rcbench.logger import get_logger

logger = get_logger(__name__)

class MCPlotter:
    """Class for visualizing Memory Capacity evaluation results."""
    
    def __init__(self, figsize: tuple = (10, 6)):
        """
        Initialize the MCPlotter.
        
        Args:
            figsize (tuple): Default figure size for plots
        """
        self.figsize = figsize
        try:
            plt.style.use('seaborn')
        except OSError:
            logger.warning("Seaborn style not available. Using default matplotlib style.")
            plt.style.use('default')
    
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
        
        plt.figure(figsize=self.figsize)
        plt.plot(delays, mc_values, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Delay (k)')
        plt.ylabel('Memory Capacity')
        plt.title(title or 'Memory Capacity vs Delay')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
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
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_prediction_results(self,
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              time: Optional[np.ndarray] = None,
                              title: Optional[str] = None,
                              save_path: Optional[str] = None) -> None:
        """
        Plot true vs predicted values.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            time (Optional[np.ndarray]): Time array for x-axis
            title (Optional[str]): Plot title
            save_path (Optional[str]): Path to save the plot
        """
        if time is None:
            time = np.arange(len(y_true))
        
        plt.figure(figsize=self.figsize)
        plt.plot(time, y_true, 'b-', label='True', alpha=0.7)
        plt.plot(time, y_pred, 'r--', label='Predicted', alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(title or 'True vs Predicted Values')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
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
        
        plt.figure(figsize=self.figsize)
        plt.plot(delays, cumulative_mc, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Delay (k)')
        plt.ylabel('Cumulative Memory Capacity')
        plt.title(title or 'Cumulative Memory Capacity vs Delay')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_mc_heatmap(self,
                       mc_matrix: np.ndarray,
                       delay_range: range,
                       feature_range: range,
                       title: Optional[str] = None,
                       save_path: Optional[str] = None) -> None:
        """
        Plot Memory Capacity as a heatmap for different delays and features.
        
        Args:
            mc_matrix (np.ndarray): 2D array of MC values
            delay_range (range): Range of delays
            feature_range (range): Range of features
            title (Optional[str]): Plot title
            save_path (Optional[str]): Path to save the plot
        """
        plt.figure(figsize=self.figsize)
        plt.imshow(mc_matrix, aspect='auto', cmap='viridis')
        plt.colorbar(label='Memory Capacity')
        plt.xlabel('Features')
        plt.ylabel('Delay (k)')
        plt.title(title or 'Memory Capacity Heatmap')
        
        # Add axis labels
        plt.xticks(range(len(feature_range)), feature_range)
        plt.yticks(range(len(delay_range)), delay_range)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_all_results(self,
                        delay_results: Dict[int, float],
                        feature_importance: np.ndarray,
                        y_true: np.ndarray,
                        y_pred: np.ndarray,
                        save_dir: Optional[str] = None) -> None:
        """
        Generate all visualization plots for MC evaluation results.
        
        Args:
            delay_results (Dict[int, float]): Dictionary mapping delays to MC values
            feature_importance (np.ndarray): Array of feature importance scores
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            save_dir (Optional[str]): Directory to save the plots
        """
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
        
        # Plot MC vs delay
        self.plot_mc_vs_delay(
            delay_results,
            save_path=f"{save_dir}/mc_vs_delay.png" if save_dir else None
        )
        
        # Plot feature importance
        self.plot_feature_importance(
            feature_importance,
            save_path=f"{save_dir}/feature_importance.png" if save_dir else None
        )
        
        # Plot prediction results
        self.plot_prediction_results(
            y_true,
            y_pred,
            save_path=f"{save_dir}/prediction_results.png" if save_dir else None
        )
        
        # Plot cumulative MC
        self.plot_cumulative_mc(
            delay_results,
            save_path=f"{save_dir}/cumulative_mc.png" if save_dir else None
        )
