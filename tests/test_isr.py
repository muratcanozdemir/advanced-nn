"""Fast check of experiment 01.

E[ISR^2] = (M - 1) E[sigma(s)^2] / sigma(c)^2 holds exactly, because values are independent unit vectors and
the target weight sigma(c) is fixed. The median is lower than sqrt of that mean for heavy-tailed kernels
(relu^3, exp at small M), so the test compares means of squares, not medians.
"""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("isr", Path(__file__).parents[1] / "experiments/01_isr_capacity/run.py")
isr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(isr)

MS = [64, 128, 2048]


@pytest.fixture(scope="module")
def curves():
    return isr.isr_curves(trials=256, ms=MS, seed=1)


@pytest.mark.parametrize("kern", isr.KERNELS, ids=lambda k: k.name)
@pytest.mark.parametrize("c", isr.COSINES)
@pytest.mark.parametrize("m", MS)
def test_mean_square_isr_matches_analytic(curves, kern, c, m):
    xs = curves[(kern.name, c)][m]
    mean_sq = sum(x * x for x in xs) / len(xs)
    # Worst case (relu^3, M = 64) has a per-trial relative variance near 1.4: SE ~ 7.5% at 256 trials.
    assert mean_sq == pytest.approx(isr.analytic(kern, c, m) ** 2, rel=0.3)


@pytest.mark.parametrize("c", isr.COSINES)
def test_sharper_kernels_interfere_less(curves, c):
    order = [isr.median(curves[(k.name, c)][2048]) for k in isr.KERNELS]
    assert order == sorted(order, reverse=True)
