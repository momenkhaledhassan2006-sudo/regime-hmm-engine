"""Tests for BIC, Hungarian alignment and classification metrics."""

import math

import numpy as np
import pytest

from hmm_engine import (
    GaussianHMM,
    align_states_hungarian,
    calculate_classification_metrics,
    compute_bic,
)


def test_hungarian_recovers_a_label_permutation():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, size=2000)
    perm = np.array([3, 0, 4, 1, 2])  # unsupervised labels = permuted true labels
    y_pred = perm[y_true]
    aligned, mapping = align_states_hungarian(y_true, y_pred)
    assert np.array_equal(aligned, y_true)
    assert mapping == {int(perm[c]): c for c in range(5)}


def test_hungarian_handles_more_states_than_classes():
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([2, 2, 3, 0, 0, 1, 1, 0])  # 4 predicted states, 2 true classes
    aligned, mapping = align_states_hungarian(y_true, y_pred)
    assert set(mapping.keys()) == {0, 1, 2, 3}
    assert set(np.unique(aligned)) <= {0, 1}
    assert aligned.shape == y_true.shape


def test_classification_metrics_against_hand_computation():
    y_true = np.array([0, 0, 1, 1, 2, 2, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 2, 0, 2])
    m = calculate_classification_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(6 / 8)
    # per-class F1: class0 -> 2*1/(2*1+1+1)=0.5 ; class1 -> 2*2/(4+1+0)=0.8 ; class2 -> 2*3/(6+0+1)=6/7
    assert m["macro_f1"] == pytest.approx((0.5 + 0.8 + 6 / 7) / 3)
    # row (true) counts = [2, 2, 4]; column (predicted) counts = [2, 3, 3]
    p_e = (2 * 2 + 2 * 3 + 4 * 3) / 64.0
    assert m["cohen_kappa"] == pytest.approx((0.75 - p_e) / (1 - p_e))


def test_perfect_and_chance_level_kappa():
    y = np.arange(100) % 4
    assert calculate_classification_metrics(y, y)["cohen_kappa"] == pytest.approx(1.0)
    const = np.zeros(100, dtype=int)
    assert calculate_classification_metrics(y, const)["cohen_kappa"] == pytest.approx(0.0)


def test_bic_formula():
    truth = GaussianHMM(n_states=2).set_params(
        [0.5, 0.5],
        [[0.9, 0.1], [0.1, 0.9]],
        [[-1.0], [1.0]],
        [[[1.0]], [[1.0]]],
    )
    X, _ = truth.sample(500, seed=1)
    ll = truth.score(X)
    # K=2, D=1 -> (K-1) + K(K-1) + K*D + K*D(D+1)/2 = 1 + 2 + 2 + 2 = 7
    assert truth.n_parameters == 7
    assert compute_bic(truth, X, ll) == pytest.approx(7 * math.log(500) - 2 * ll)
    assert compute_bic(truth, X) == pytest.approx(compute_bic(truth, X, ll))


def test_bic_prefers_true_number_of_states():
    truth = GaussianHMM(n_states=2).set_params(
        [0.5, 0.5],
        [[0.95, 0.05], [0.05, 0.95]],
        [[-3.0, 0.0], [3.0, 0.0]],
        [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
    )
    X, _ = truth.sample(1500, seed=4)
    bic = {}
    for k in (1, 2, 4):
        m = GaussianHMM(n_states=k, init="kmeans", seed=0).fit(X, max_iters=100, tol=1e-7)
        bic[k] = compute_bic(m, X, m.log_likelihood_)
    assert bic[2] < bic[1]
    assert bic[2] < bic[4]
