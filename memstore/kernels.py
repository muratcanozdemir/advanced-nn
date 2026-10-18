"""Kernels sigma: turn a similarity score s = k . x into a memory weight w = sigma(s).

A kernel with `log_weight` defined (the exponential) lets normalisation run in log space,
so softmax stays numerically stable at large beta.
"""

import math

import torch


class Kernel:
    name = "kernel"
    log_weight = None

    def __call__(self, s: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def second_moment(self, d: int) -> float:
        """E[sigma(s)^2] for s ~ N(0, 1/d), the score of a random unit key against a fixed unit query."""
        raise NotImplementedError

    def __repr__(self):
        return self.name


class Linear(Kernel):
    name = "linear"

    def __call__(self, s):
        return s

    def second_moment(self, d):
        return 1.0 / d


class ReLUPow(Kernel):
    """sigma(s) = max(s, 0)^p. p = 1 is the ReLU FFN; p = 2 approximates a GLU's effective kernel."""

    def __init__(self, p: int = 1):
        self.p = p
        self.name = "relu" if p == 1 else f"relu^{p}"

    def __call__(self, s):
        return torch.relu(s) ** self.p

    def second_moment(self, d):
        # E[relu(s)^(2p)] = 1/2 * E[s^(2p)] = 1/2 * (2p-1)!! / d^p
        double_fact = math.prod(range(2 * self.p - 1, 0, -2))
        return 0.5 * double_fact / d**self.p


class Exp(Kernel):
    """sigma(s) = exp(beta * s). With sum normalisation this is softmax attention."""

    def __init__(self, beta: float = 1.0):
        self.beta = beta
        self.name = f"exp(beta={beta:g})"

    def log_weight(self, s):
        return self.beta * s

    def __call__(self, s):
        return torch.exp(self.beta * s)

    def second_moment(self, d):
        return math.exp(2 * self.beta**2 / d)


class Identity(Kernel):
    """For layers whose score function already produces the weight (e.g. SwiGLU's gated score)."""

    name = "identity"

    def __call__(self, s):
        return s


linear = Linear()
relu = ReLUPow(1)
relu2 = ReLUPow(2)
relu3 = ReLUPow(3)
identity = Identity()
