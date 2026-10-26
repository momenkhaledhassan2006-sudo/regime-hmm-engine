"""Core correctness tests for the log-space Gaussian HMM."""

import itertools

import numpy as np
import pytest
import torch

from hmm_engine import GaussianHMM


def _sticky_two_state_truth() -> GaussianHMM:
    """Ground-truth 2-state, 2-D HMM with sticky transitions (A_ii = 0.95)."""
    return GaussianHMM(n_states=2).set_params(
        startprob=[0.5, 0.5],
        transmat=[[0.95, 0.05], [0.05, 0.95]],
        means=[[-2.0, -1.0], [2.0, 1.5]],
        covars=[[[1.0, 0.3], [0.3, 1.0]], [[1.5, -0.4], [-0.4, 0.8]]],
    )


def _three_state_truth() -> GaussianHMM:
    """Overlapping 3-state HMM: a harder, slower-converging EM problem."""
    return GaussianHMM(n_states=3).set_params(
        startprob=[1 / 3, 1 / 3, 1 / 3],
        transmat=[[0.90, 0.07, 0.03], [0.05, 0.90, 0.05], [0.03, 0.07, 0.90]],
        means=[[-1.0, 0.0], [0.0, 1.0], [1.0, -0.5]],
        covars=[
            [[1.0, 0.2], [0.2, 1.0]],
            [[0.8, -0.3], [-0.3, 1.2]],
            [[1.2, 0.0], [0.0, 0.7]],
        ],
    )


def test_monotonic_log_likelihood():
    """Baum-Welch must never decrease the log-likelihood: LL_i >= LL_{i-1} - 1e-5."""
    for truth, k in ((_sticky_two_state_truth(), 2), (_three_state_truth(), 3)):
        X, _ = truth.sample(1200, seed=3)
        model = GaussianHMM(n_states=k, init="kmeans", seed=1)
        model.fit(X, max_iters=40, tol=0.0)  # tol=0 -> run all 40 iterations
        ll = np.asarray(model.log_likelihood_history_)
        assert len(ll) == 41
        diffs = np.diff(ll)
        assert np.all(diffs >= -1e-5), f"LL decreased: min step {diffs.min():.3e}"
        assert ll[-1] > ll[0]


def test_synthetic_recovery():
    """EM recovers the transition matrix and the emission means of a sticky 2-state HMM."""
    truth = _sticky_two_state_truth()
    X, _ = truth.sample(3000, seed=7)

    model = GaussianHMM(n_states=2, init="kmeans", n_init=2, seed=0)
    model.fit(X, max_iters=200, tol=1e-8)

    # Resolve label switching by sorting states on the first mean coordinate.
    order = torch.argsort(model.means_[:, 0])
    means = model.means_[order].numpy()
    A = model.transmat_[order][:, order].numpy()

    np.testing.assert_allclose(means, truth.means_.numpy(), atol=0.2)
    np.testing.assert_allclose(A, truth.transmat_.numpy(), atol=0.03)
    # The fitted model should be at least as good as the generating model.
    assert model.score(X) >= truth.score(X) - 1e-6


def test_causality():
    """Filtered posteriors at t ignore x_{t+1}; smoothed posteriors do not."""
    truth = _sticky_two_state_truth()
    X, _ = truth.sample(300, seed=11)
    model = GaussianHMM(n_states=2, init="kmeans", seed=0).fit(X, max_iters=100, tol=1e-8)

    t = 150
    mu0, mu1 = model.means_[0], model.means_[1]
    # Make x_t genuinely ambiguous (midpoint of the two regimes) so that the posterior at t
    # is not saturated, then compare two futures: x_{t+1} looks like regime 0 vs regime 1.
    X_a = X.clone()
    X_a[t] = 0.5 * (mu0 + mu1)
    X_a[t + 1] = mu0
    X_b = X_a.clone()
    X_b[t + 1] = mu1

    causal_a = model.predict_proba(X_a, causal=True)
    causal_b = model.predict_proba(X_b, causal=True)
    smooth_a = model.predict_proba(X_a, causal=False)
    smooth_b = model.predict_proba(X_b, causal=False)

    # Strictly no look-ahead: every filtered row up to and including t is unchanged.
    assert (causal_a[: t + 1] - causal_b[: t + 1]).abs().max().item() < 1e-12
    # Row t+1 itself must react to its own observation.
    assert (causal_a[t + 1] - causal_b[t + 1]).abs().max().item() > 1e-3
    # Smoothing uses the future: the posterior at t moves when x_{t+1} changes.
    assert (smooth_a[t] - smooth_b[t]).abs().max().item() > 1e-3


def test_posteriors_are_distributions():
    truth = _sticky_two_state_truth()
    X, _ = truth.sample(400, seed=5)
    for causal in (True, False):
        P = truth.predict_proba(X, causal=causal)
        assert P.dtype == torch.float64
        assert torch.all(P >= 0)
        assert (P.sum(dim=1) - 1.0).abs().max().item() < 1e-10
    # At the final time step filtering and smoothing coincide.
    Pc = truth.predict_proba(X, causal=True)
    Ps = truth.predict_proba(X, causal=False)
    assert (Pc[-1] - Ps[-1]).abs().max().item() < 1e-9


def test_forward_matches_brute_force_and_viterbi_is_optimal():
    """Check the log-space recursions against exhaustive enumeration on a short sequence."""
    truth = _sticky_two_state_truth()
    X, _ = truth.sample(8, seed=2)
    log_b = truth._log_emission(X)
    T, K = log_b.shape

    scores = []
    paths = list(itertools.product(range(K), repeat=T))
    for path in paths:
        lp = truth.log_startprob_[path[0]] + log_b[0, path[0]]
        for t in range(1, T):
            lp = lp + truth.log_transmat_[path[t - 1], path[t]] + log_b[t, path[t]]
        scores.append(lp)
    scores = torch.stack(scores)

    assert abs(float(torch.logsumexp(scores, dim=0)) - truth.score(X)) < 1e-10
    assert tuple(truth.decode_viterbi(X).tolist()) == paths[int(scores.argmax())]


def test_covariances_stay_positive_definite_on_degenerate_data():
    """Collinear features must not break the Cholesky thanks to reg_covar."""
    g = torch.Generator().manual_seed(0)
    a = torch.randn(400, 1, generator=g, dtype=torch.float64)
    X = torch.cat([a, 2.0 * a, a + 1e-9], dim=1)  # rank-1 data in 3-D
    model = GaussianHMM(n_states=2, reg_covar=1e-4).fit(X, max_iters=20, tol=1e-8)
    eig = torch.linalg.eigvalsh(model.covars_)
    assert torch.all(eig > 0)
    assert np.isfinite(model.log_likelihood_)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        GaussianHMM(n_states=0)
    with pytest.raises(ValueError):
        GaussianHMM(n_states=2).fit(torch.tensor([[float("nan"), 0.0], [0.0, 1.0], [1.0, 2.0]]))
    with pytest.raises(RuntimeError):
        GaussianHMM(n_states=2).predict_proba(torch.zeros(5, 2))
