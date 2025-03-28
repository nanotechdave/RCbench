from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
class BaseEvaluator:
    def __init__(self, input_signal, nodes_output):
        self.input_signal = input_signal
        self.nodes_output = nodes_output

    def feature_selection(self, X, y, method='kbest', num_features=10):
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

    def train_regression(self, X_train, y_train, alpha=1.0):
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        return model

    def split_train_test(self, X, y, train_ratio=0.8):
        split_idx = int(len(y) * train_ratio)
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
