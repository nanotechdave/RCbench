import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from typing import Dict, List, Tuple, Union, Optional
from rcbench.logger import get_logger

logger = get_logger(__name__)

class FeatureSelector:
    """
    A class to handle feature selection methods for reservoir computing tasks.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the feature selector.
        
        Args:
            random_state (int): Random seed for reproducibility.
        """
        self.random_state = random_state
        self.pca = None
        self.scaler = None
        self.imputer = None
        self.feature_importance = None
        self.selected_features = None
        self.selected_feature_names = None
        self.feature_selection_method = None
        self.num_features = None
    
    def select_features(self, 
                       X: np.ndarray, 
                       y: np.ndarray, 
                       node_names: List[str],
                       method: str = 'pca', 
                       num_features: Union[int, str] = 'all') -> Tuple[np.ndarray, List[int], List[str]]:
        """
        Select features from input data using the specified method.
        
        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target values.
            node_names (List[str]): List of node names.
            method (str): Feature selection method ('pca' or 'kbest').
            num_features (Union[int, str]): Number of features to select or 'all'.
            
        Returns:
            Tuple[np.ndarray, List[int], List[str]]: Selected features, their indices, and their names.
        """
        self.feature_selection_method = method
        self.num_features = num_features
        
        # Set random seed for reproducibility
        np.random.seed(self.random_state)
        
        if method == 'kbest':
            return self._select_kbest(X, y, node_names, num_features)
        elif method == 'pca':
            return self._select_pca(X, node_names, num_features)
        else:
            raise ValueError(f"Unsupported method: {method}. Choose 'kbest' or 'pca'")
    
    def _select_kbest(self, 
                     X: np.ndarray, 
                     y: np.ndarray,
                     node_names: List[str],
                     num_features: Union[int, str]) -> Tuple[np.ndarray, List[int], List[str]]:
        """
        Select features using SelectKBest.
        
        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target values.
            node_names (List[str]): List of node names.
            num_features (Union[int, str]): Number of features to select or 'all'.
            
        Returns:
            Tuple[np.ndarray, List[int], List[str]]: Selected features, their indices, and their names.
        """
        selector = SelectKBest(score_func=f_regression, k=num_features)
        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True).tolist()
        
        # Store the results
        self.selected_features = selected_indices
        self.selected_feature_names = [node_names[i] for i in selected_indices]
        
        # Create feature importance series
        self.feature_importance = pd.Series(
            selector.scores_, 
            index=node_names
        )
        
        return X_selected, selected_indices, self.selected_feature_names
    
    def _select_pca(self, 
                   X: np.ndarray, 
                   node_names: List[str],
                   num_features: Union[int, str]) -> Tuple[np.ndarray, List[int], List[str]]:
        """
        Select features using PCA based on first component loadings.
        
        Args:
            X (np.ndarray): Input features.
            node_names (List[str]): List of node names.
            num_features (Union[int, str]): Number of features to select or 'all'.
            
        Returns:
            Tuple[np.ndarray, List[int], List[str]]: Selected features, their indices, and their names.
        """
        # Handle 'all' case
        if num_features == 'all':
            num_features = X.shape[1]
        
        # Log node names
        logger.debug(f"PCA feature selection with nodes: {node_names}")
        
        # Create DataFrame with node names as columns
        X_df = pd.DataFrame(X, columns=node_names)
        
        # Initialize preprocessing pipeline with fixed random state
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()
        
        # Apply preprocessing
        processed_data = pd.DataFrame(
            self.imputer.fit_transform(X_df), 
            columns=X_df.columns
        )
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(processed_data), 
            columns=processed_data.columns
        )
        
        # Apply PCA with fixed random state
        self.pca = PCA(random_state=self.random_state)
        self.pca.fit(X_scaled)
        
        # Calculate feature importance using absolute values of the first component
        loadings = np.abs(self.pca.components_[0])
        
        # Create a DataFrame to help with stable sorting
        importance_df = pd.DataFrame({
            'node': node_names,
            'importance': loadings,
            'index': range(len(node_names))
        })
        
        # Sort by importance (descending), then by index for stability
        importance_df = importance_df.sort_values(['importance', 'index'], ascending=[False, True])
        
        # Get top features
        top_features = importance_df.head(num_features)
        
        # Extract selected indices and names
        self.selected_features = top_features['index'].astype(int).tolist()
        self.selected_feature_names = top_features['node'].tolist()
        
        # Create feature importance Series
        self.feature_importance = pd.Series(loadings, index=node_names)
        
        # Log selected nodes
        logger.info(f"Selected nodes using PCA: {self.selected_feature_names}")
        
        # Return selected features
        X_selected = X[:, self.selected_features]
        return X_selected, self.selected_features, self.selected_feature_names
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply feature selection to new data.
        
        Args:
            X (np.ndarray): Input features.
            
        Returns:
            np.ndarray: Selected features.
        """
        if self.selected_features is None:
            raise ValueError("Feature selection has not been performed yet. Call select_features first.")
        
        return X[:, self.selected_features]
    
    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance scores.
        
        Returns:
            pd.Series: Feature importance scores indexed by node names.
        """
        if self.feature_importance is None:
            raise ValueError("Feature selection has not been performed yet. Call select_features first.")
        
        return self.feature_importance
    
    def get_selected_feature_names(self) -> List[str]:
        """
        Get the names of the selected features.
        
        Returns:
            List[str]: Names of the selected features.
        """
        if self.selected_feature_names is None:
            raise ValueError("Feature selection has not been performed yet. Call select_features first.")
        
        return self.selected_feature_names 