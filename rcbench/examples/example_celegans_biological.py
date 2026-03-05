"""
C. elegans Biological Reservoir Use-Case (MC, IPC, KR, GR)

This self-running example shows how to:
1. Load a C. elegans connectome adjacency from local `.npy` files.
2. Preprocess structure (gap-junction duplication, isolated-node removal).
3. Simulate reservoir dynamics on that graph.
4. Evaluate states with RCbench's built-in evaluators:
   Memory Capacity (MC), Information Processing Capacity (IPC),
   Kernel Rank (KR), and Generalization Rank (GR).

Expected files in `rcbench/examples/data`:
- `ce_adj.npy` 
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

# Allow running the example directly from source without installing the package.
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from rcbench import (
    GeneralizationRankEvaluator,
    IPCEvaluator,
    KernelRankEvaluator,
    MemoryCapacityEvaluator,
)
from rcbench.logger import get_logger


logger = get_logger(__name__)
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class ExampleConfig:
    random_state: int = 7
    total_steps: int = 3000
    washout: int = 400
    spectral_radius: float = 0.90
    leak_rate: float = 0.35
    input_scale: float = 0.80
    perturb_std: float = 0.01
    mc_max_delay: int = 40
    ipc_max_delay: int = 10
    ipc_max_degree: int = 3
    ridge_alpha: float = 0.1
    train_ratio: float = 0.8


def load_connectome_adjacency(data_dir: Path) -> Tuple[np.ndarray, str]:
    """Load adjacency matrix from local `.npy` files."""
    suffix = ""
    adj_path = data_dir / f"ce_adj{suffix}.npy"

    if not adj_path.exists():
        raise FileNotFoundError(f"Missing required data file: {adj_path}")

    adj = np.load(adj_path).astype(np.float64, copy=False)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Adjacency must be square, got shape {adj.shape}")

    if not np.isfinite(adj).all():
        adj = np.where(np.isfinite(adj), adj, 0.0)

    np.fill_diagonal(adj, 0.0)
    return adj, f"ce_adj{suffix}.npy"


def apply_gap_duplication(adj: np.ndarray, enabled: bool) -> Tuple[np.ndarray, int]:
    """
    Enforce symmetric gap-junction proxy from reciprocal positive edges.

    Proxy rule: if adjacency has positive weights in both directions (i->j and j->i),
    treat the pair as electrical coupling and set both directions to the mean magnitude.
    """
    W = adj.copy()

    reciprocal_positive = (adj > 0.0) & (adj.T > 0.0)
    i_idx, j_idx = np.where(np.triu(reciprocal_positive, k=1))

    if enabled:
        for i, j in zip(i_idx, j_idx):
            g = 0.5 * (abs(adj[i, j]) + abs(adj[j, i]))
            W[i, j] = g
            W[j, i] = g

    return W, len(i_idx)


def drop_isolated_nodes(W: np.ndarray, drop: bool) -> Tuple[np.ndarray, int]:
    """Optionally remove nodes with zero in-degree and out-degree."""
    if not drop:
        return W, 0

    degree = np.count_nonzero(W, axis=0) + np.count_nonzero(W, axis=1)
    keep_mask = degree > 0
    removed_count = int(np.count_nonzero(~keep_mask))
    return W[np.ix_(keep_mask, keep_mask)], removed_count


def scale_to_spectral_radius(W: np.ndarray, target_sr: float) -> Tuple[np.ndarray, float]:
    """Scale matrix to target spectral radius for stable ESN-like dynamics."""
    eigvals = np.linalg.eigvals(W)
    native_sr = float(np.max(np.abs(eigvals)))
    if native_sr < 1e-12:
        return W.copy(), native_sr
    return (target_sr / native_sr) * W, native_sr


def sample_input_weights(n_nodes: int, input_scale: float, rng: np.random.Generator) -> np.ndarray:
    """Random input projection vector normalized to requested scale."""
    win = rng.normal(0.0, 1.0, size=n_nodes)
    win = input_scale * win / (np.linalg.norm(win) + 1e-12)
    return win.astype(np.float64)


def run_reservoir(
    W: np.ndarray,
    win: np.ndarray,
    input_signal: np.ndarray,
    leak_rate: float,
) -> np.ndarray:
    """Single-reservoir simulation: x(t) = (1-l)x(t-1) + l*tanh(Wx + Win*u)."""
    n_steps = input_signal.shape[0]
    n_nodes = W.shape[0]

    x = np.zeros(n_nodes, dtype=np.float64)
    states = np.zeros((n_steps, n_nodes), dtype=np.float64)

    one_minus_leak = 1.0 - leak_rate
    for t in range(n_steps):
        pre = W @ x + win * input_signal[t]
        x = one_minus_leak * x + leak_rate * np.tanh(pre)
        states[t] = x

    return states


def run_example(
    config: ExampleConfig,
    duplicate_gap_junctions: bool,
    drop_isolated: bool,
) -> None:
    data_dir = Path(__file__).resolve().parent / "data"

    raw_adj, source_label = load_connectome_adjacency(data_dir=data_dir)
    W_gapped, gap_pair_count = apply_gap_duplication(raw_adj, duplicate_gap_junctions)
    W_pre, removed_count = drop_isolated_nodes(W_gapped, drop_isolated)

    W, native_sr = scale_to_spectral_radius(W_pre, config.spectral_radius)

    rng = np.random.default_rng(config.random_state)
    input_signal = rng.uniform(-1.0, 1.0, size=config.total_steps).astype(np.float64)
    input_signal -= np.mean(input_signal)

    win = sample_input_weights(W.shape[0], config.input_scale, rng)

    states_clean = run_reservoir(W, win, input_signal, config.leak_rate)
    noisy_input = np.clip(
        input_signal + config.perturb_std * rng.normal(size=input_signal.shape),
        -1.0,
        1.0,
    )
    states_noisy = run_reservoir(W, win, noisy_input, config.leak_rate)

    # Wash out initial transient before RCbench evaluation.
    eval_input = input_signal[config.washout:]
    eval_states = states_clean[config.washout:]

    # For GR, use the state difference between clean and noisy drives.
    train_len = int((config.total_steps - config.washout) * config.train_ratio)
    delta_states = (
        states_noisy[config.washout : config.washout + train_len]
        - states_clean[config.washout : config.washout + train_len]
    )

    mc_eval = MemoryCapacityEvaluator(
        input_signal=eval_input,
        nodes_output=eval_states,
        max_delay=config.mc_max_delay,
        random_state=config.random_state,
    )
    mc_results = mc_eval.calculate_total_memory_capacity(
        feature_selection_method="pca",
        num_features="all",
        modeltype="Ridge",
        regression_alpha=config.ridge_alpha,
        train_ratio=config.train_ratio,
    )

    ipc_eval = IPCEvaluator(
        input_signal=eval_input,
        nodes_output=eval_states,
        max_delay=config.ipc_max_delay,
        max_degree=config.ipc_max_degree,
        max_total_degree=config.ipc_max_degree,
        include_cross_terms=True,
        random_state=config.random_state,
    )
    ipc_results = ipc_eval.calculate_total_capacity(
        feature_selection_method="pca",
        num_features="all",
        modeltype="Ridge",
        regression_alpha=config.ridge_alpha,
        train_ratio=config.train_ratio,
    )

    kr_eval = KernelRankEvaluator(
        nodes_output=eval_states,
        input_signal=eval_input,
        kernel="linear",
        threshold=1e-6,
    )
    kr_results = kr_eval.run_evaluation()

    gr_eval = GeneralizationRankEvaluator(states=delta_states, threshold=1e-3)
    gr_results = gr_eval.run_evaluation()

    print("=" * 72)
    print("C. ELEGANS BIOLOGICAL RESERVOIR EXAMPLE (RCbench)")
    print("=" * 72)
    print(f"Connectome source: {source_label}")
    print(f"Nodes loaded: {raw_adj.shape[0]}")
    print(f"Nodes after preprocessing: {W_pre.shape[0]}")
    print(f"Removed isolated nodes: {removed_count}")
    print(f"Directed edges after preprocessing: {int(np.count_nonzero(W_pre))}")
    print(f"Reciprocal-positive pairs (gap proxy): {gap_pair_count}")
    print(f"Gap duplication: {duplicate_gap_junctions}")
    print(f"Native spectral radius: {native_sr:.4f}")
    print(f"Scaled spectral radius target: {config.spectral_radius:.4f}")

    print("\nMetric Results")
    print(f"  MC  (total): {mc_results['total_memory_capacity']:.4f}")
    print(f"  IPC (total): {ipc_results['total_capacity']:.4f}")
    print(f"  IPC linear memory component: {ipc_results['linear_memory_capacity']:.4f}")
    print(f"  KR  (combined input+nodes): {kr_results['kernel_rank']}")
    print(f"  GR  (noise-difference states): {gr_results['generalization_rank']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run C. elegans biological reservoir example with RCbench metrics."
    )
    parser.add_argument(
        "--no-gap-duplication",
        action="store_true",
        help="Disable symmetric duplication for reciprocal-positive edge pairs.",
    )
    parser.add_argument(
        "--keep-isolated",
        action="store_true",
        help="Keep isolated nodes instead of dropping them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExampleConfig()
    run_example(
        config=config,
        duplicate_gap_junctions=not args.no_gap_duplication,
        drop_isolated=not args.keep_isolated,
    )


if __name__ == "__main__":
    main()
##please check out my papers!! https://doi.org/10.1016/j.isci.2025.114436
#and soon to be preprint Determinants of Hyperparameter Invariance in Connectome Reservoir Computing by Miles Churchland