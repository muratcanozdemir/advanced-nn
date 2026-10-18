import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "arms", Path(__file__).parents[1] / "experiments/06_persistent_memory/model.py"
)
arms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arms)

CFG = dict(layers=2, d=32, n_heads=4, hidden=48, context=16)


def params(arm):
    return sum(p.numel() for p in arms.ByteLM(arm, **CFG).parameters())


@pytest.mark.parametrize("arm", arms.ARMS)
def test_forward_shape(arm):
    out = arms.ByteLM(arm, **CFG)(torch.randint(256, (3, 16)))
    assert out.shape == (3, 16, 256)


def test_parameter_counts_match_except_n0():
    """FFN arms carry two extra LayerNorms per block (2 * 2 * d * layers), so allow that difference."""
    base = params("all-attn")
    for arm in ("ffn", "ff-attn", "attn-split", "head-split"):
        assert abs(params(arm) - base) <= 4 * CFG["d"] * CFG["layers"], arm
    assert params("n0") < base


@pytest.mark.parametrize("arm", arms.ARMS)
def test_causal(arm):
    """Changing a future byte must not change earlier logits."""
    torch.manual_seed(0)
    m = arms.ByteLM(arm, **CFG).eval()
    x = torch.randint(256, (1, 16))
    y = x.clone()
    y[0, 10] = (y[0, 10] + 1) % 256
    torch.testing.assert_close(m(x)[:, :10], m(y)[:, :10])
