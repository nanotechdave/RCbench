import numpy as np
from scipy.spatial.distance import pdist, squareform
from rcbench.tasks.baseevaluator import BaseEvaluator
from rcbench.logger import get_logger
from typing import Tuple, Dict, Any, Optional, Union, List

logger = get_logger(__name__)

class KernelRankEvaluator(BaseEvaluator):
    """
    Evaluates the kernel rank (KR) of reservoir states.
    
    Based on the methodology described in Wringe et al. (2025) "Reservoir Computing 
    Benchmarks: a tutorial review and critique" (arXiv:2405.06561), the kernel rank
    measures the computational capacity of a reservoir by analyzing the rank of the
    state matrix.
    
    The kernel rank can be computed in two modes:
    1. **Combined mode** (recommended): Concatenates the input signal with reservoir 
       states to capture the combined dynamics of input and reservoir responses.
    2. **Nodes-only mode**: Uses only the reservoir node outputs.
    
    This evaluator computes the kernel (Gram) matrix in feature space (N×N) for 
    efficiency, since rank(X.T @ X) = rank(X @ X.T) = rank(X). The effective rank 
    is determined using Singular Value Decomposition (SVD).

    Parameters:
        nodes_output : np.ndarray
            Reservoir states with shape (T, N), where T is the number of timesteps
            and N is the number of nodes.
        input_signal : np.ndarray, optional
            Input signal with shape (T,) or (T, 1). If provided, it will be 
            concatenated with nodes_output to form the combined state matrix.
            This captures the relationship between input and reservoir dynamics.
        kernel : str, optional
            Type of kernel to use. Options:
              - 'linear': Uses the dot-product kernel in feature space, K = X.T @ X.
              - 'rbf': Uses the Gaussian (RBF) kernel in sample space,
                       K[i,j] = exp(-||x_i - x_j||^2 / (2*sigma^2)).
            Default is 'linear'.
        sigma : float, optional
            Parameter for the RBF kernel (ignored if kernel is 'linear'). Default is 1.0.
        threshold : float, optional
            Relative threshold for counting singular values (values > threshold*max_singular_value are counted).
            Default is 1e-6.
            
    Note:
        For linear kernels, the computation uses the feature-space Gram matrix (N×N) 
        instead of the sample-space matrix (T×T). This is mathematically equivalent 
        since both matrices share the same non-zero eigenvalues, but is much faster 
        when N << T (typical case with 50 nodes and 3000+ samples).
            
    References:
        Wringe, C., Trefzer, M., & Stepney, S. (2025). "Reservoir Computing 
        Benchmarks: a tutorial review and critique". arXiv:2405.06561
    """
    def __init__(self, 
                 nodes_output: np.ndarray,
                 input_signal: Optional[Union[np.ndarray, List[float]]] = None,
                 kernel: str = 'linear', 
                 sigma: float = 1.0, 
                 threshold: float = 1e-6,
                 ) -> None:
        self.nodes_output: np.ndarray = nodes_output
        self.kernel: str = kernel
        self.sigma: float = sigma
        self.threshold: float = threshold
        
        # Process input signal if provided
        if input_signal is not None:
            input_signal = np.asarray(input_signal)
            if input_signal.ndim == 1:
                input_signal = input_signal.reshape(-1, 1)
            
            # Validate dimensions match
            if len(input_signal) != len(nodes_output):
                raise ValueError(
                    f"Input signal length ({len(input_signal)}) must match "
                    f"nodes_output length ({len(nodes_output)})"
                )
            
            # Concatenate input signal with nodes output
            self.state_matrix = np.hstack([input_signal, nodes_output])
            self.include_input = True
            logger.info(f"Kernel rank will be computed on combined matrix: "
                       f"input (1) + nodes ({nodes_output.shape[1]}) = {self.state_matrix.shape[1]} features")
        else:
            self.state_matrix = nodes_output
            self.include_input = False
            logger.info(f"Kernel rank will be computed on nodes only: {nodes_output.shape[1]} features")
        
        self.n_samples, self.n_features = self.state_matrix.shape

    def compute_kernel_matrix(self) -> np.ndarray:
        """
        Computes the kernel (Gram) matrix from the state matrix.
        
        For linear kernels, uses feature-space formulation (N×N) for efficiency.
        For RBF kernels, uses sample-space formulation (T×T) as required by the
        kernel definition.
        
        Mathematical note: For linear kernels, rank(X.T @ X) = rank(X @ X.T) = rank(X),
        and both matrices share identical non-zero eigenvalues. Using the smaller
        matrix is faster: O(N³) vs O(T³) when N << T.
        
        Returns:
            np.ndarray: The computed kernel matrix.
                - Linear kernel: shape (N, N) - feature-space Gram matrix
                - RBF kernel: shape (T, T) - sample-space kernel matrix
        """
        states = self.state_matrix  # Shape: (T, N)
        
        if self.kernel == 'linear':
            # Feature-space Gram matrix: K = X.T @ X
            # Shape: (N, N) where N = number of features (nodes + optional input)
            # This is much faster when N << T (typical case)
            # rank(X.T @ X) = rank(X @ X.T) = rank(X)
            K = np.dot(states.T, states)
        elif self.kernel == 'rbf':
            # RBF kernel must be computed in sample space
            # K[i,j] = exp(-||x_i - x_j||^2 / (2*sigma^2))
            # Shape: (T, T)
            dists = squareform(pdist(states, 'sqeuclidean'))
            K = np.exp(-dists / (2 * self.sigma**2))
        else:
            raise ValueError("Unsupported kernel type. Please use 'linear' or 'rbf'.")
        
        return K

    def compute_kernel_rank(self) -> Tuple[int, np.ndarray]:
        """
        Computes the effective kernel rank based on the singular values of the kernel matrix.
        
        The effective rank represents the number of linearly independent computational
        dimensions in the combined input-reservoir system.
        
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
                - 'n_features': Number of features in state matrix (input + nodes or just nodes)
                - 'n_samples': Number of time samples
                - 'include_input': Whether input signal was included
        """
        rank, singular_values = self.compute_kernel_rank()
        
        mode = "input + nodes" if self.include_input else "nodes only"
        logger.info(f"Computed Kernel Rank: {rank} (mode: {mode}, features: {self.n_features})")
        
        return {
            'kernel_rank': rank,
            'singular_values': singular_values,
            'kernel': self.kernel,
            'sigma': self.sigma,
            'threshold': self.threshold,
            'n_features': self.n_features,
            'n_samples': self.n_samples,
            'include_input': self.include_input
        }
