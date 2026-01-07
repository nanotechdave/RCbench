import numpy as np
from scipy.spatial.distance import pdist, squareform
from rcbench.tasks.baseevaluator import BaseEvaluator
from rcbench.logger import get_logger
from typing import Tuple, Dict, Any

logger = get_logger(__name__)

class KernelRankEvaluator(BaseEvaluator):
    """
    Evaluates the kernel rank (KR) of reservoir states.
    
    This evaluator computes the kernel (Gram) matrix from the reservoir node outputs
    and determines the effective rank using Singular Value Decomposition (SVD).
    
    The kernel rank measures the effective dimensionality of the reservoir's 
    computational space - i.e., how many of the N nodes are providing linearly 
    independent information.
    
    Note on Dale et al. (2019) Algorithm 2:
        The original kernel rank definition from Dale et al. requires applying
        multiple distinct input streams to the reservoir and collecting the final
        state from each. This implementation adapts the concept for single time-series
        data by computing the kernel matrix in NODE SPACE (N×N) rather than 
        sample space (T×T). This gives the effective dimensionality of node outputs
        over the recorded time series.
        
        For the full Dale et al. methodology, users should:
        1. Apply m distinct input streams to the reservoir
        2. Collect the final state vector for each input
        3. Build matrix M of shape (N, m) and compute its rank

    Parameters:
        nodes_output : np.ndarray
            Reservoir states with shape (T, N), where T is the number of timesteps
            and N is the number of nodes.
        kernel : str, optional
            Type of kernel to use. Options:
              - 'linear': Uses the dot-product kernel, K = X^T X (N×N matrix).
              - 'rbf': Uses the Gaussian (RBF) kernel between nodes.
            Default is 'linear'.
        sigma : float, optional
            Parameter for the RBF kernel (ignored if kernel is 'linear'). Default is 1.0.
        threshold : float, optional
            Relative threshold for counting singular values (values > threshold*max_singular_value are counted).
            Default is 1e-6.
    """
    def __init__(self, 
                 nodes_output: np.ndarray, 
                 kernel: str = 'linear', 
                 sigma: float = 1.0, 
                 threshold: float = 1e-6,
                 ) -> None:
        self.nodes_output: np.ndarray = nodes_output
        self.kernel: str = kernel
        self.sigma: float = sigma
        self.threshold: float = threshold
        
        # Store dimensions for reference
        self.n_samples, self.n_nodes = nodes_output.shape

    def compute_kernel_matrix(self) -> np.ndarray:
        """
        Computes the kernel (Gram) matrix from the reservoir states in NODE SPACE.
        
        The kernel matrix K has shape (N, N) where N is the number of nodes.
        K[i,j] represents the similarity/correlation between node i and node j
        across all time samples.
        
        Returns:
            np.ndarray: The computed kernel matrix with shape (N, N).
        """
        states = self.nodes_output  # Shape: (T, N)
        
        if self.kernel == 'linear':
            # Linear kernel in node space: K = X^T @ X → (N, N) matrix
            # K[i,j] = sum over time of (node_i * node_j)
            K = np.dot(states.T, states)
            
        elif self.kernel == 'rbf':
            # RBF kernel between nodes (columns)
            # Transpose to get (N, T) so each row is a node's time series
            nodes_as_rows = states.T  # Shape: (N, T)
            
            # Compute pairwise squared Euclidean distances between nodes
            # Each node is a T-dimensional vector
            dists = squareform(pdist(nodes_as_rows, 'sqeuclidean'))
            
            # Apply RBF kernel: K[i,j] = exp(-||node_i - node_j||^2 / (2*sigma^2))
            K = np.exp(-dists / (2 * self.sigma**2))
        else:
            raise ValueError("Unsupported kernel type. Please use 'linear' or 'rbf'.")
        
        return K

    def compute_kernel_rank(self) -> Tuple[int, np.ndarray]:
        """
        Computes the effective kernel rank based on the singular values of the kernel matrix.
        
        The effective rank represents the number of linearly independent computational
        dimensions provided by the reservoir nodes.
        
        Returns:
            effective_rank (int): The effective rank (number of singular values above threshold * max_singular_value).
            singular_values (np.ndarray): The singular values of the kernel matrix (sorted in descending order).
        """
        K = self.compute_kernel_matrix()
        
        # Compute the SVD of the kernel matrix
        U, s, Vh = np.linalg.svd(K, full_matrices=False)
        
        # Calculate effective rank based on singular values
        s_max = np.max(s)
        effective_rank = np.sum(s > (self.threshold * s_max))
        
        return effective_rank, s

    def run_evaluation(self) -> Dict[str, Any]:
        """
        Runs the kernel rank evaluation.
        
        Returns:
            dict: A dictionary containing:
                - 'kernel_rank': The computed effective rank
                - 'singular_values': The singular values of the kernel matrix
                - 'kernel': The kernel type used
                - 'sigma': The sigma parameter (for RBF kernel)
                - 'threshold': The threshold used for rank computation
                - 'n_nodes': Number of reservoir nodes
                - 'n_samples': Number of time samples
                - 'kernel_matrix_shape': Shape of the kernel matrix (N, N)
        """
        rank, singular_values = self.compute_kernel_rank()
        logger.info(f"Computed Kernel Rank: {rank} (out of {self.n_nodes} nodes)")
        
        return {
            'kernel_rank': rank,
            'singular_values': singular_values,
            'kernel': self.kernel,
            'sigma': self.sigma,
            'threshold': self.threshold,
            'n_nodes': self.n_nodes,
            'n_samples': self.n_samples,
            'kernel_matrix_shape': (self.n_nodes, self.n_nodes)
        }
