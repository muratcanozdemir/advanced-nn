import torch

from memstore import writes


def unit(x):
    return x / x.norm(dim=-1, keepdim=True)


def pairs(M, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return unit(torch.randn(M, d, generator=g, dtype=torch.float64)), torch.randn(
        M, 8, generator=g, dtype=torch.float64
    )


def test_hebbian_is_sum_of_outer_products():
    K, V = pairs(5)
    torch.testing.assert_close(writes.hebbian(K, V), sum(torch.outer(v, k) for k, v in zip(K, V)))


def test_least_squares_exact_below_dk():
    K, V = pairs(12)
    torch.testing.assert_close(K @ writes.least_squares(K, V).T, V)


def test_delta_beta1_makes_newest_write_exact():
    K, V = pairs(40)
    torch.testing.assert_close(writes.delta(K, V) @ K[-1], V[-1])


def test_delta_many_passes_converges_to_least_squares_below_dk():
    K, V = pairs(12)
    torch.testing.assert_close(
        writes.delta(K, V, beta=0.5, passes=3000), writes.least_squares(K, V), atol=1e-4, rtol=1e-4
    )


def test_delta_above_dk_approaches_least_squares_as_beta_shrinks():
    K, V = pairs(30)
    ls = writes.least_squares(K, V)
    gaps = [(writes.delta(K, V, beta=b, passes=int(40 / b)) - ls).norm().item() for b in (0.4, 0.1, 0.025)]
    assert gaps[0] > gaps[1] > gaps[2]
