"""Model selection and evaluation metrics for unsupervised HMM state sequences."""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .core import GaussianHMM

IntArrayLike = Union[np.ndarray, torch.Tensor, Sequence[int]]


def _as_int_vector(a: IntArrayLike) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    arr = np.asarray(a).astype(np.int64).ravel()
    return arr


def compute_bic(
    model: GaussianHMM,
    X: Union[torch.Tensor, np.ndarray],
    log_likelihood: Optional[float] = None,
) -> float:
    """Bayesian Information Criterion ``BIC = p * ln(T) - 2 * ln L`` (lower is better).

    ``p`` is the number of free parameters of the fitted model: ``(K-1)`` start
    probabilities, ``K(K-1)`` transition probabilities, ``K*D`` means and
    ``K*D(D+1)/2`` covariance entries. If ``log_likelihood`` is ``None`` it is
    computed with ``model.score(X)``.
    """
    n_obs = int(X.shape[0])
    ll = model.score(X) if log_likelihood is None else float(log_likelihood)
    return float(model.n_parameters * math.log(n_obs) - 2.0 * ll)


def confusion_matrix(
    y_true: IntArrayLike,
    y_pred: IntArrayLike,
    labels: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Confusion matrix with rows = true label, columns = predicted label.

    Returns ``(matrix, labels)``. By default ``labels`` is the sorted union of
    the labels that occur in ``y_true`` and ``y_pred``.
    """
    t = _as_int_vector(y_true)
    p = _as_int_vector(y_pred)
    if t.shape != p.shape:
        raise ValueError("y_true and y_pred must have the same length.")
    lab = np.union1d(t, p) if labels is None else np.asarray(labels, dtype=np.int64)
    index = {int(l): i for i, l in enumerate(lab)}
    cm = np.zeros((len(lab), len(lab)), dtype=np.int64)
    for a, b in zip(t, p):
        if int(a) in index and int(b) in index:
            cm[index[int(a)], index[int(b)]] += 1
    return cm, lab


def align_states_hungarian(
    y_true: IntArrayLike,
    y_pred: IntArrayLike,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Resolve label switching between unsupervised states and ground-truth classes.

    Builds the contingency matrix ``C[i, j] = #{t : pred_t = i and true_t = j}``
    and solves the assignment problem ``max sum_i C[i, sigma(i)]`` with
    ``scipy.optimize.linear_sum_assignment`` (Hungarian algorithm). If there are
    more predicted states than true classes, the unmatched predicted states are
    mapped to the true class they overlap with most.

    Returns
    -------
    y_pred_aligned : np.ndarray
        Predictions relabelled into the ground-truth label space.
    mapping : dict
        ``{predicted_state: true_label}``.
    """
    t = _as_int_vector(y_true)
    p = _as_int_vector(y_pred)
    if t.shape != p.shape:
        raise ValueError("y_true and y_pred must have the same length.")
    if t.size == 0:
        raise ValueError("Empty label sequences.")

    true_labels = np.unique(t)
    pred_labels = np.unique(p)
    p_idx = np.searchsorted(pred_labels, p)
    t_idx = np.searchsorted(true_labels, t)

    contingency = np.zeros((len(pred_labels), len(true_labels)), dtype=np.int64)
    np.add.at(contingency, (p_idx, t_idx), 1)

    rows, cols = linear_sum_assignment(contingency, maximize=True)
    mapping: Dict[int, int] = {int(pred_labels[r]): int(true_labels[c]) for r, c in zip(rows, cols)}
    for r, state in enumerate(pred_labels):
        if int(state) not in mapping:
            mapping[int(state)] = int(true_labels[int(contingency[r].argmax())])

    lut = np.array([mapping[int(s)] for s in pred_labels], dtype=np.int64)
    return lut[p_idx], mapping


def calculate_classification_metrics(
    y_true: IntArrayLike,
    y_pred_aligned: IntArrayLike,
) -> Dict[str, float]:
    """Accuracy, macro-averaged F1 and Cohen's kappa.

    Macro-F1 averages the per-class F1 over the union of labels present in
    ``y_true`` and ``y_pred_aligned``; classes with no true and no predicted
    samples are not included. Cohen's kappa is ``(p_o - p_e) / (1 - p_e)``.
    """
    cm, _ = confusion_matrix(y_true, y_pred_aligned)
    n = float(cm.sum())
    if n == 0:
        raise ValueError("Empty label sequences.")

    tp = np.diag(cm).astype(np.float64)
    row = cm.sum(axis=1).astype(np.float64)  # true counts
    col = cm.sum(axis=0).astype(np.float64)  # predicted counts

    accuracy = float(tp.sum() / n)

    denom = row + col  # = 2 TP + FP + FN
    f1 = np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    macro_f1 = float(f1.mean())

    p_e = float((row * col).sum() / (n * n))
    if p_e >= 1.0:
        kappa = 1.0 if accuracy >= 1.0 else 0.0
    else:
        kappa = float((accuracy - p_e) / (1.0 - p_e))

    return {"accuracy": accuracy, "macro_f1": macro_f1, "cohen_kappa": kappa}
