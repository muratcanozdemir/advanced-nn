"""Each grid row built from `read` must equal its standard implementation."""

import math

import pytest
import torch
import torch.nn.functional as F

from memstore import kernels, normalize, read
from memstore.layers import FFN, AdditiveScore, AllAttention, Attention, SwiGLUFFN

torch.manual_seed(0)
B, T, D, H = 2, 7, 16, 4


@pytest.fixture
def x():
    return torch.randn(B, T, D)


def test_relu_ffn_matches_two_matmuls(x):
    ffn = FFN(D, 32)
    ref = torch.relu(x @ ffn.keys.T) @ ffn.values
    torch.testing.assert_close(ffn(x), ref)


def test_swiglu_matches_reference(x):
    ffn = SwiGLUFFN(D, 32)
    ref = (F.silu(x @ ffn.keys.T) * (x @ ffn.gates.T)) @ ffn.values
    torch.testing.assert_close(ffn(x), ref)


def test_softmax_ffn_is_attention_over_weights(x):
    """Sukhbaatar 4.1 / Vaswani v2 appendix: ReLU -> softmax turns the FFN into attention over parameters."""
    ffn = FFN(D, 32, kernel=kernels.Exp(1.0), norm="sum")
    ref = torch.softmax(x @ ffn.keys.T, dim=-1) @ ffn.values
    torch.testing.assert_close(ffn(x), ref)


@pytest.mark.parametrize("causal", [True, False])
def test_attention_matches_sdpa(x, causal):
    attn = Attention(D, H, causal=causal)
    q, k, v = attn.split(x)
    ref = attn.merge(F.scaled_dot_product_attention(q, k, v, is_causal=causal))
    torch.testing.assert_close(attn(x), ref)


def test_all_attention_with_no_persistent_memories_is_attention(x):
    aa, attn = AllAttention(D, H, n_persistent=0), Attention(D, H)
    attn.load_state_dict({k: v for k, v in aa.state_dict().items() if not k.startswith("p")})
    torch.testing.assert_close(aa(x), attn(x))


def test_all_attention_joint_softmax_matches_explicit(x):
    aa = AllAttention(D, H, n_persistent=5, norm="sum")
    q, k, v = aa.split(x)
    K = torch.cat([k, aa.pkeys.expand(B, -1, -1, -1)], 2)
    V = torch.cat([v, aa.pvalues.expand(B, -1, -1, -1)], 2)
    s = q @ K.transpose(-1, -2) / math.sqrt(aa.dh)
    s[..., :T] = s[..., :T].masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))
    ref = aa.merge(torch.softmax(s, -1) @ V)
    torch.testing.assert_close(aa(x), ref)


def test_joint_vs_split_mass():
    s = torch.randn(3, 10)
    joint = normalize(s, kernels.Exp(2.0), "sum", pools=(6, 4))
    split = normalize(s, kernels.Exp(2.0), "split", pools=(6, 4))
    torch.testing.assert_close(joint.sum(-1), torch.ones(3))
    torch.testing.assert_close(split[:, :6].sum(-1), torch.ones(3))
    torch.testing.assert_close(split[:, 6:].sum(-1), torch.ones(3))


def test_topk_softmax_keeps_k_per_pool():
    s = torch.randn(5, 12)
    n = normalize(s, kernels.Exp(1.0), "split", pools=(8, 4), topk=3)
    assert ((n[:, :8] > 0).sum(-1) == 3).all()
    assert ((n[:, 8:] > 0).sum(-1) == 3).all()


def test_linear_kernel_unnormalised_is_outer_product_store():
    """Linear kernel, no normalisation: sum_i (k_i . x) v_i = (V^T K) x, a fixed d_v x d_k state."""
    K, V, q = torch.randn(20, D), torch.randn(20, 8), torch.randn(3, D)
    state = V.T @ K
    torch.testing.assert_close(read(q, K, V, kernels.linear), q @ state.T)


@pytest.mark.parametrize("kernel", [kernels.linear, kernels.relu, kernels.relu2, kernels.relu3, kernels.Exp(4.0)])
def test_second_moment_monte_carlo(kernel):
    d = 64
    s = torch.randn(2_000_000, dtype=torch.float64) / math.sqrt(d)
    mc = (kernel(s) ** 2).mean().item()
    assert mc == pytest.approx(kernel.second_moment(d), rel=0.03)


def test_additive_score_matches_loop():
    score = AdditiveScore(D, 12, hidden=20)
    q, K = torch.randn(B, 3, D), torch.randn(B, 5, 12)
    ref = torch.stack(
        [
            torch.stack(
                [
                    torch.stack([score.v(torch.tanh(score.W(q[b, i]) + score.U(K[b, j]))).squeeze() for j in range(5)])
                    for i in range(3)
                ]
            )
            for b in range(B)
        ]
    )
    torch.testing.assert_close(score(q, K), ref)


@pytest.mark.parametrize("norm", ["sum", "split"])
def test_fused_softmax_path_matches_general_path(norm):
    """read() routes exp-kernel softmax to fused attention; it must equal normalize() @ V."""
    q, K, V = torch.randn(2, 3, 9, 8), torch.randn(2, 3, 14, 8), torch.randn(2, 3, 14, 5)
    mask = torch.cat([torch.ones(9, 10, dtype=torch.bool).tril(), torch.ones(9, 4, dtype=torch.bool)], 1)
    mask[0, :10] = False  # first query sees no context memory: that pool must contribute zero under split
    kern = kernels.Exp(0.7)
    ref = normalize(q @ K.transpose(-1, -2), kern, norm, pools=(10, 4), mask=mask) @ V
    torch.testing.assert_close(read(q, K, V, kern, norm, pools=(10, 4), mask=mask), ref)


def test_fused_path_broadcasts_shared_keys():
    q, K, V = torch.randn(2, 3, 9, 8), torch.randn(3, 6, 8), torch.randn(3, 6, 5)
    ref = torch.softmax(q @ K.transpose(-1, -2), -1) @ V
    torch.testing.assert_close(read(q, K, V, kernels.Exp(1.0), "sum"), ref)
