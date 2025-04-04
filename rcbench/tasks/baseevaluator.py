from sklearn.linear_model import Ridge, LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from rcbench.tasks.featureselector import FeatureSelector
from rcbench.logger import get_logger

logger = get_logger(__name__)

class BaseEvaluator:
    def __init__(self, 
                input_signal: np.ndarray, 
                nodes_output: np.ndarray,
                electrode_names: Optional[List[str]] = None):
        """
        Initialize the BaseEvaluator.
        
        Args:
            input_signal (np.ndarray): Input signal.
            nodes_output (np.ndarray): Output of the nodes.
            electrode_names (Optional[List[str]]): Names of the electrodes.
        """
        self.input_signal: np.ndarray = input_signal
        self.nodes_output: np.ndarray = nodes_output
        
        # Create electrode names if not provided
        if electrode_names is None:
            self.electrode_names = [f'Electrode {i}' for i in range(nodes_output.shape[1])]
        else:
            self.electrode_names = electrode_names
            
        # Initialize feature selector and variables
        self.feature_selector = FeatureSelector()
        self.selected_features = None
        self.selected_feature_names = None
        self.feature_selection_method = None
        self.num_features = None

    def feature_selection(self, 
                          X: np.ndarray, 
                          y: np.ndarray, 
                          method: str = 'kbest', 
                          num_features: Union[int, str] = 'all',
                          ) -> Tuple[np.ndarray, List[int], List[str]]:
        """
        Perform feature selection using the FeatureSelector module.
        
        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target values.
            method (str): Feature selection method ('pca' or 'kbest').
            num_features (Union[int, str]): Number of features to select or 'all'.
            
        Returns:
            Tuple[np.ndarray, List[int], List[str]]: Selected features, their indices, and their names.
        """
        # Store parameters
        self.feature_selection_method = method
        self.num_features = num_features
        
        # Use the FeatureSelector with electrode names
        X_selected, selected_indices, selected_names = self.feature_selector.select_features(
            X=X, 
            y=y, 
            electrode_names=self.electrode_names,
            method=method, 
            num_features=num_features
        )
        
        # Store results
        self.selected_features = selected_indices
        self.selected_feature_names = selected_names
        
        logger.info(f"Selected features using {method}: {self.selected_feature_names}")
        
        # Return selected features and indices
        return X_selected, selected_indices, selected_names

    def apply_feature_selection(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the stored feature selection to new data.
        """
        return self.feature_selector.transform(X)

    def train_regression(self, 
                         X_train: np.ndarray, 
                         y_train: np.ndarray,
                         modeltype: str = "Ridge", 
                         alpha: float = 1.0,
                         ) -> Union[Ridge, LinearRegression]:
        if modeltype.lower() == "ridge":
            model = Ridge(alpha=alpha)

        elif modeltype.lower() == "linear":
            model = LinearRegression()
        
        else:
            raise ValueError("Model unrecognized, please select Ridge or Linear")
        model.fit(X_train, y_train)
        return model

    def split_train_test(self, 
                         X: np.ndarray, 
                         y: np.ndarray, 
                         train_ratio: float = 0.8,
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        split_idx = int(len(y) * train_ratio)
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
