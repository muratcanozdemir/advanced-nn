"""Write rules for a fixed-size linear store S (d_v x d_k), read as S @ k.

All take keys K (M, d_k) and values V (M, d_v) in write order and return S.
"""

import torch


def hebbian(K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """S = sum_i v_i k_i^T. Linear attention's state."""
    return V.T @ K


def delta(K: torch.Tensor, V: torch.Tensor, beta: float = 1.0, passes: int = 1) -> torch.Tensor:
    """S <- S + beta (v - S k) k^T, one pair at a time. DeltaNet's state update (keys assumed unit norm).

    With beta = 1 each write makes the newest pair exact, overwriting what older pairs left along k.
    Repeated passes are Kaczmarz's method: for M <= d_k they converge to the exact least-squares store;
    for M > d_k they settle into a cycle around it whose distance shrinks with beta.
    """
    S = K.new_zeros(V.shape[1], K.shape[1])
    for _ in range(passes):
        for k, v in zip(K, V):
            S += beta * torch.outer(v - S @ k, k)
    return S


def least_squares(K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """S = argmin ||K S^T - V||: exact recall while M <= d_k, the optimal linear store beyond."""
    return (torch.linalg.pinv(K) @ V).T
