"""
graph_builder.py — Road Network Graph Constructor
===================================================
Converts the 4-road intersection into a graph adjacency matrix
suitable for Chebyshev spectral graph convolution in STGCN.

Graph definition:
    Nodes   = {0: North, 1: South, 2: East, 3: West} approach roads
    Edges   = Physical connectivity at the intersection
              (all approach roads share the same center node,
               so all pairs are connected — fully connected graph
               with self-loops)

Outputs:
    A_hat   : Normalized adjacency matrix   shape (N, N)
    L_cheb  : Chebyshev basis matrices      shape (Ks, N, N)
"""

import logging
from typing import Tuple, List

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adjacency Matrix Construction
# ---------------------------------------------------------------------------

def build_adjacency_matrix(num_nodes: int = 4) -> np.ndarray:
    """
    Build raw adjacency matrix for the intersection graph.

    All approach roads connect to every other approach road through
    the shared center intersection node → fully connected graph.

    Parameters
    ----------
    num_nodes : int
        Number of approach roads (default 4: N, S, E, W).

    Returns
    -------
    A : np.ndarray, shape (num_nodes, num_nodes)
        Symmetric adjacency matrix with self-loops.
    """
    # Fully connected with self-loops
    A = np.ones((num_nodes, num_nodes), dtype=np.float32)
    logger.debug(f"Built {num_nodes}x{num_nodes} adjacency matrix (fully connected).")
    return A


def build_weighted_adjacency(
    positions: List[Tuple[float, float]] = None,
    sigma: float = 1.0,
    epsilon: float = 0.5,
) -> np.ndarray:
    """
    Build distance-weighted adjacency matrix using Gaussian kernel.

    W_ij = exp(-d_ij^2 / sigma^2)  if d_ij < epsilon, else 0

    Parameters
    ----------
    positions : list of (x, y) tuples for each node.
        Default: cross-shaped layout at unit distance.
    sigma : float
        Gaussian kernel width.
    epsilon : float
        Distance threshold; edges beyond this are pruned to 0.

    Returns
    -------
    A : np.ndarray, shape (num_nodes, num_nodes)
    """
    if positions is None:
        # Default: N(0,1), S(0,-1), E(1,0), W(-1,0)
        positions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    n = len(positions)
    pos = np.array(positions, dtype=np.float32)

    # Pairwise Euclidean distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]   # (n, n, 2)
    dist = np.sqrt((diff ** 2).sum(axis=-1))                # (n, n)

    # Gaussian kernel
    W = np.exp(-(dist ** 2) / (sigma ** 2))
    # Threshold
    W[W < epsilon] = 0.0
    # Self-loops
    np.fill_diagonal(W, 1.0)

    logger.debug(f"Built weighted adjacency matrix with sigma={sigma}, epsilon={epsilon}.")
    return W.astype(np.float32)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    """
    Symmetric normalization: A_hat = D^{-1/2} (A + I) D^{-1/2}

    Parameters
    ----------
    A : np.ndarray, shape (N, N)

    Returns
    -------
    A_hat : np.ndarray, shape (N, N)
    """
    A_tilde = A + np.eye(A.shape[0], dtype=A.dtype)  # add self-loops
    D = np.diag(A_tilde.sum(axis=1))                  # degree matrix
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D.diagonal() + 1e-8))
    A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    return A_hat.astype(np.float32)


def compute_scaled_laplacian(A_hat: np.ndarray) -> np.ndarray:
    """
    Compute the scaled/normalized graph Laplacian for Chebyshev expansion.

    L_tilde = (2 / lambda_max) * L - I
    where L = I - A_hat (normalized Laplacian),
          lambda_max ≈ 2 for normalized L.

    Parameters
    ----------
    A_hat : np.ndarray, shape (N, N) — normalized adjacency.

    Returns
    -------
    L_tilde : np.ndarray, shape (N, N)
    """
    N = A_hat.shape[0]
    L = np.eye(N, dtype=np.float32) - A_hat   # L = I - A_hat
    # Eigenvalue decomposition to find lambda_max
    eigenvalues = np.linalg.eigvalsh(L)
    lambda_max = float(eigenvalues.max())
    if lambda_max < 1e-8:
        lambda_max = 2.0  # fallback
    L_tilde = (2.0 / lambda_max) * L - np.eye(N, dtype=np.float32)
    return L_tilde


# ---------------------------------------------------------------------------
# Chebyshev Basis
# ---------------------------------------------------------------------------

def chebyshev_polynomials(L_tilde: np.ndarray, Ks: int) -> List[np.ndarray]:
    """
    Compute Chebyshev polynomial basis matrices T_0, T_1, ..., T_{Ks-1}.

    Recurrence:
        T_0 = I
        T_1 = L_tilde
        T_k = 2 * L_tilde * T_{k-1} - T_{k-2}

    Parameters
    ----------
    L_tilde : np.ndarray, shape (N, N)
    Ks : int
        Polynomial order (number of basis matrices).

    Returns
    -------
    cheb_polys : list of np.ndarray, each shape (N, N)
    """
    N = L_tilde.shape[0]
    cheb_polys = [np.eye(N, dtype=np.float32)]       # T_0 = I

    if Ks == 1:
        return cheb_polys

    cheb_polys.append(L_tilde.copy())                # T_1 = L_tilde

    for k in range(2, Ks):
        T_k = 2.0 * L_tilde @ cheb_polys[-1] - cheb_polys[-2]
        cheb_polys.append(T_k)

    logger.debug(f"Computed {Ks} Chebyshev polynomial matrices.")
    return cheb_polys


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def build_graph(config: dict) -> dict:
    """
    Full graph construction pipeline.

    Parameters
    ----------
    config : dict
        Project config from config.yaml.

    Returns
    -------
    graph : dict with keys:
        'A'          : raw adjacency       (N, N)
        'A_hat'      : normalized adjacency (N, N)
        'L_tilde'    : scaled Laplacian    (N, N)
        'cheb_polys' : list of (N, N) Chebyshev matrices
        'num_nodes'  : int
    """
    num_nodes = config.get("intersection", {}).get("num_roads", 4)
    Ks = config.get("stgcn", {}).get("Ks", 3)

    A = build_adjacency_matrix(num_nodes)
    A_hat = normalize_adjacency(A)
    L_tilde = compute_scaled_laplacian(A_hat)
    cheb_polys = chebyshev_polynomials(L_tilde, Ks)

    logger.info(
        f"Graph built: {num_nodes} nodes, Ks={Ks}, "
        f"A_hat range=[{A_hat.min():.3f}, {A_hat.max():.3f}]"
    )

    return {
        "A": A,
        "A_hat": A_hat,
        "L_tilde": L_tilde,
        "cheb_polys": cheb_polys,
        "num_nodes": num_nodes,
    }


if __name__ == "__main__":
    import yaml
    logging.basicConfig(level=logging.DEBUG)

    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    graph = build_graph(cfg)
    print("Adjacency matrix A:")
    print(graph["A"])
    print("\nNormalized A_hat:")
    print(graph["A_hat"])
    print(f"\nChebyshev polynomials: {len(graph['cheb_polys'])} matrices")
