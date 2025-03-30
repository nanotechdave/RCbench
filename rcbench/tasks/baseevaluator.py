from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
class BaseEvaluator:
    def __init__(self, input_signal: np.ndarray, nodes_output: np.ndarray):
        self.input_signal: np.ndarray = input_signal
        self.nodes_output: np.ndarray = nodes_output

    def feature_selection(self, 
                          X: np.ndarray, 
                          y: np.ndarray, 
                          method: str = 'kbest', 
                          num_features: int = 10,
                          ) -> Tuple[np.ndarray, Union[List[int], List[str]]]:
        if method == 'kbest':
            selector = SelectKBest(score_func=f_regression, k=num_features)
            X_selected = selector.fit_transform(X, y)
            selected_indices = selector.get_support(indices=True).tolist()
            return X_selected, selected_indices
        elif method == 'pca':
            pca = PCA(n_components=num_features)
            X_selected = pca.fit_transform(X)
            return X_selected, [f"PCA_{i+1}" for i in range(num_features)]
        else:
            raise ValueError("Unsupported method: choose 'kbest' or 'pca'")

    def train_regression(self, 
                         X_train: np.ndarray, 
                         y_train: np.ndarray, 
                         alpha: float = 1.0,
                         ) -> Ridge:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        return model

    def split_train_test(self, 
                         X: np.ndarray, 
                         y: np.ndarray, 
                         train_ratio: float = 0.8,
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        split_idx = int(len(y) * train_ratio)
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
