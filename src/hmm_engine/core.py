"""Multivariate Gaussian Hidden Markov Model in log-probability space.

Everything (forward/backward recursions, E-step, M-step, Viterbi) is computed
in ``torch.float64`` using log-sum-exp arithmetic, so no probability is ever
multiplied in linear space and underflow cannot occur for long sequences.

Notation
--------
``T`` observations, ``K`` hidden states, ``D`` features.

* ``log_b[t, k]    = log p(x_t | z_t = k)``                  (T, K)
* ``log_alpha[t,k] = log p(x_{1:t}, z_t = k)``               (T, K)
* ``log_beta[t, k] = log p(x_{t+1:T} | z_t = k)``            (T, K)
* ``log_gamma[t,k] = log p(z_t = k | x_{1:T})``              (T, K)
* ``log_xi_sum[i,j] = log sum_t p(z_t = i, z_{t+1} = j | x_{1:T})``  (K, K)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

ArrayLike = Union[torch.Tensor, np.ndarray, Sequence[Sequence[float]]]

_DTYPE = torch.float64
_LOG_2PI = math.log(2.0 * math.pi)
# exp(-745) is the smallest positive float64 (denormal); used to keep log-values finite.
_MIN_LOG = -745.0


# --------------------------------------------------------------------------- #
# Initialisation helpers
# --------------------------------------------------------------------------- #
def _kmeans(
    X: torch.Tensor,
    k: int,
    generator: torch.Generator,
    max_iter: int = 100,
) -> torch.Tensor:
    """K-means (k-means++ seeding, Lloyd iterations). Returns labels of shape (T,)."""
    T, D = X.shape
    centers = torch.empty(k, D, dtype=X.dtype, device=X.device)

    first = int(torch.randint(0, T, (1,), generator=generator).item())
    centers[0] = X[first]
    d2 = ((X - centers[0]) ** 2).sum(dim=1)
    for c in range(1, k):
        total = d2.sum()
        if float(total) <= 0.0:  # all points identical to chosen centres
            probs = torch.full((T,), 1.0 / T, dtype=X.dtype)
        else:
            probs = (d2 / total).cpu()
        idx = int(torch.multinomial(probs, 1, generator=generator).item())
        centers[c] = X[idx]
        d2 = torch.minimum(d2, ((X - centers[c]) ** 2).sum(dim=1))

    labels = torch.full((T,), -1, dtype=torch.long, device=X.device)
    for _ in range(max_iter):
        dist = torch.cdist(X, centers)  # (T, k)
        new_labels = dist.argmin(dim=1)
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
        onehot = torch.nn.functional.one_hot(labels, k).to(X.dtype)  # (T, k)
        counts = onehot.sum(dim=0)
        sums = onehot.T @ X
        for c in range(k):
            if counts[c] > 0:
                centers[c] = sums[c] / counts[c]
            else:  # empty cluster: re-seed on the point farthest from its centre
                far = int(dist.min(dim=1).values.argmax().item())
                centers[c] = X[far]
    return labels


def _quantile_labels(X: torch.Tensor, k: int, feature: Optional[int]) -> torch.Tensor:
    """Equal-mass quantile bins of a 1-D score (a chosen feature, or the first PC)."""
    T = X.shape[0]
    if feature is None:
        Xc = X - X.mean(dim=0, keepdim=True)
        cov = (Xc.T @ Xc) / max(T - 1, 1)
        _, evecs = torch.linalg.eigh(cov)
        score = Xc @ evecs[:, -1]
    else:
        score = X[:, feature]
    ranks = score.argsort().argsort()  # rank-based => exactly equal-sized bins, tie safe
    return torch.clamp((ranks * k) // T, max=k - 1)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class GaussianHMM:
    """Hidden Markov Model with full-covariance multivariate Gaussian emissions.

    Parameters
    ----------
    n_states:
        Number of hidden states ``K``.
    reg_covar:
        Diagonal regularisation ``reg_covar * I`` added to every covariance in the
        M-step (after symmetry projection) to guarantee positive definiteness.
    init:
        ``"kmeans"`` (k-means++ / Lloyd) or ``"quantile"`` (equal-mass bins of
        ``init_feature``, or of the first principal component when it is ``None``).
    init_feature:
        Feature index used by the quantile initialiser.
    n_init:
        Number of EM restarts (seeds ``seed, seed+1, ...``); the best final
        log-likelihood is kept.
    seed:
        Base random seed for the initialiser.
    transition_floor:
        Smallest probability allowed in the start vector and transition matrix
        (keeps log-values finite).
    device:
        Torch device. All computation uses ``torch.float64``.
    verbose:
        Print EM progress.

    Attributes (after ``fit``)
    --------------------------
    log_startprob_, log_transmat_, means_, covars_, log_likelihood_history_,
    log_likelihood_, n_iter_, converged_
    """

    def __init__(
        self,
        n_states: int,
        reg_covar: float = 1e-6,
        init: str = "kmeans",
        init_feature: Optional[int] = None,
        n_init: int = 1,
        seed: int = 0,
        transition_floor: float = 1e-12,
        device: Union[str, torch.device] = "cpu",
        verbose: bool = False,
    ) -> None:
        if n_states < 1:
            raise ValueError("n_states must be >= 1")
        if init not in ("kmeans", "quantile"):
            raise ValueError("init must be 'kmeans' or 'quantile'")
        if reg_covar <= 0.0:
            raise ValueError("reg_covar must be strictly positive")
        if n_init < 1:
            raise ValueError("n_init must be >= 1")
        if not 0.0 < transition_floor < 1.0 / n_states:
            raise ValueError("transition_floor must be in (0, 1/n_states)")

        self.n_states = int(n_states)
        self.reg_covar = float(reg_covar)
        self.init = init
        self.init_feature = init_feature
        self.n_init = int(n_init)
        self.seed = int(seed)
        self.transition_floor = float(transition_floor)
        self.device = torch.device(device)
        self.verbose = bool(verbose)
        self._log_floor = math.log(self.transition_floor)

        self.n_features_: Optional[int] = None
        self.log_startprob_: Optional[torch.Tensor] = None
        self.log_transmat_: Optional[torch.Tensor] = None
        self.means_: Optional[torch.Tensor] = None
        self.covars_: Optional[torch.Tensor] = None
        self._chol: Optional[torch.Tensor] = None

        self.log_likelihood_history_: List[float] = []
        self.all_run_log_likelihoods_: List[float] = []
        self.log_likelihood_: float = float("nan")
        self.n_iter_: int = 0
        self.converged_: bool = False

    # ------------------------------------------------------------------ #
    # Convenience views
    # ------------------------------------------------------------------ #
    @property
    def startprob_(self) -> torch.Tensor:
        self._check_fitted()
        return self.log_startprob_.exp()

    @property
    def transmat_(self) -> torch.Tensor:
        self._check_fitted()
        return self.log_transmat_.exp()

    @property
    def n_parameters(self) -> int:
        """Number of free parameters (used by the BIC)."""
        self._check_fitted()
        K, D = self.n_states, int(self.n_features_)
        return (K - 1) + K * (K - 1) + K * D + K * D * (D + 1) // 2

    # ------------------------------------------------------------------ #
    # Input handling / parameter plumbing
    # ------------------------------------------------------------------ #
    def _check_fitted(self) -> None:
        if self.means_ is None or self.log_transmat_ is None:
            raise RuntimeError("Model has no parameters: call fit() or set_params() first.")

    def _prepare(self, X: ArrayLike) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            t = X.detach().to(device=self.device, dtype=_DTYPE)
        else:
            t = torch.as_tensor(np.asarray(X, dtype=np.float64), dtype=_DTYPE, device=self.device)
        if t.ndim == 1:
            t = t.unsqueeze(1)
        if t.ndim != 2:
            raise ValueError(f"X must be 2-D (T, D); got shape {tuple(t.shape)}")
        if not bool(torch.isfinite(t).all()):
            raise ValueError("X contains NaN or inf values")
        if self.n_features_ is not None and t.shape[1] != self.n_features_:
            raise ValueError(f"X has {t.shape[1]} features, model expects {self.n_features_}")
        return t.contiguous()

    def _symmetrize_regularize(self, covars: torch.Tensor) -> torch.Tensor:
        """Symmetry projection followed by ``+ reg_covar * I``."""
        D = covars.shape[-1]
        sym = 0.5 * (covars + covars.transpose(-1, -2))
        return sym + self.reg_covar * torch.eye(D, dtype=_DTYPE, device=covars.device)

    def _cholesky(self, covars: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cholesky factor of every covariance, adding escalating jitter if needed."""
        D = covars.shape[-1]
        eye = torch.eye(D, dtype=_DTYPE, device=covars.device)
        jitter = 0.0
        for _ in range(10):
            cand = covars + jitter * eye
            chol, info = torch.linalg.cholesky_ex(cand)
            if bool((info == 0).all()):
                return cand, chol
            jitter = self.reg_covar if jitter == 0.0 else jitter * 10.0
        raise np.linalg.LinAlgError("Covariance matrices are not positive definite.")

    def _store(
        self,
        log_pi: torch.Tensor,
        log_A: torch.Tensor,
        means: torch.Tensor,
        covars: torch.Tensor,
    ) -> None:
        """Floor, renormalise (in log space) and store parameters."""
        log_pi = torch.clamp(log_pi, min=self._log_floor)
        log_pi = log_pi - torch.logsumexp(log_pi, dim=0)
        log_A = torch.clamp(log_A, min=self._log_floor)
        log_A = log_A - torch.logsumexp(log_A, dim=1, keepdim=True)
        covars, chol = self._cholesky(covars)
        self.log_startprob_ = log_pi
        self.log_transmat_ = log_A
        self.means_ = means
        self.covars_ = covars
        self._chol = chol

    def set_params(
        self,
        startprob: ArrayLike,
        transmat: ArrayLike,
        means: ArrayLike,
        covars: ArrayLike,
    ) -> "GaussianHMM":
        """Set parameters explicitly (useful for sampling / testing)."""
        K = self.n_states
        pi = torch.as_tensor(np.asarray(startprob, dtype=np.float64), dtype=_DTYPE, device=self.device)
        A = torch.as_tensor(np.asarray(transmat, dtype=np.float64), dtype=_DTYPE, device=self.device)
        mu = torch.as_tensor(np.asarray(means, dtype=np.float64), dtype=_DTYPE, device=self.device)
        cov = torch.as_tensor(np.asarray(covars, dtype=np.float64), dtype=_DTYPE, device=self.device)
        if pi.shape != (K,) or A.shape != (K, K) or mu.ndim != 2 or mu.shape[0] != K:
            raise ValueError("Inconsistent parameter shapes.")
        D = mu.shape[1]
        if cov.shape != (K, D, D):
            raise ValueError("covars must have shape (K, D, D).")
        if not torch.allclose(pi.sum(), torch.ones((), dtype=_DTYPE, device=pi.device), atol=1e-8):
            raise ValueError("startprob must sum to 1.")
        if not torch.allclose(A.sum(dim=1), torch.ones(K, dtype=_DTYPE, device=A.device), atol=1e-8):
            raise ValueError("Every row of transmat must sum to 1.")
        self.n_features_ = D
        self._store(
            torch.log(pi.clamp_min(self.transition_floor)),
            torch.log(A.clamp_min(self.transition_floor)),
            mu,
            self._symmetrize_regularize(cov),
        )
        return self

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #
    def _initialize(self, X: torch.Tensor, seed: int) -> None:
        T, D = X.shape
        K = self.n_states
        if T < max(K, 2):
            raise ValueError(f"Need at least max(K, 2) = {max(K, 2)} observations; got {T}.")
        self.n_features_ = D

        if self.init == "kmeans":
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed))
            labels = _kmeans(X, K, gen).to(X.device)
        else:
            labels = _quantile_labels(X, K, self.init_feature)

        onehot = torch.nn.functional.one_hot(labels, K).to(_DTYPE)  # (T, K)
        counts = onehot.sum(dim=0)  # (K,)
        safe = counts.clamp_min(1.0)
        means = (onehot.T @ X) / safe.unsqueeze(1)

        diff = X.unsqueeze(1) - means.unsqueeze(0)  # (T, K, D)
        wdiff = onehot.unsqueeze(2) * diff
        cov = torch.einsum("tkd,tke->kde", wdiff, diff) / safe.view(K, 1, 1)

        # Clusters with < 2 points fall back to the pooled covariance.
        Xc = X - X.mean(dim=0, keepdim=True)
        pooled = (Xc.T @ Xc) / max(T - 1, 1)
        small = counts < 2
        if bool(small.any()):
            cov[small] = pooled
        covars = self._symmetrize_regularize(cov)

        # Transition counts from the hard label sequence (+1 Laplace smoothing).
        trans = torch.zeros(K, K, dtype=_DTYPE, device=X.device)
        trans.index_put_(
            (labels[:-1], labels[1:]),
            torch.ones(T - 1, dtype=_DTYPE, device=X.device),
            accumulate=True,
        )
        trans = trans + 1.0
        log_A = torch.log(trans) - torch.log(trans.sum(dim=1, keepdim=True))
        log_pi = torch.full((K,), -math.log(K), dtype=_DTYPE, device=X.device)

        self._store(log_pi, log_A, means, covars)

    # ------------------------------------------------------------------ #
    # Emission log-density
    # ------------------------------------------------------------------ #
    def _log_emission(self, X: torch.Tensor) -> torch.Tensor:
        """``log N(x_t | mu_k, Sigma_k)`` for all t, k via batched Cholesky solves."""
        T, D = X.shape
        diff = X.unsqueeze(0) - self.means_.unsqueeze(1)  # (K, T, D)
        z = torch.linalg.solve_triangular(self._chol, diff.transpose(1, 2), upper=False)  # (K, D, T)
        maha = z.pow(2).sum(dim=1)  # (K, T)
        log_det = 2.0 * torch.log(torch.diagonal(self._chol, dim1=-2, dim2=-1)).sum(dim=-1)  # (K,)
        log_b = -0.5 * (D * _LOG_2PI + log_det.unsqueeze(1) + maha)
        return log_b.T.contiguous()  # (T, K)

    # ------------------------------------------------------------------ #
    # Dynamic programming (log space)
    # ------------------------------------------------------------------ #
    def _forward(self, log_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. Returns ``log_alpha`` (T, K) and the sequence log-likelihood."""
        T, K = log_b.shape
        log_A = self.log_transmat_
        log_alpha = torch.empty_like(log_b)
        log_alpha[0] = self.log_startprob_ + log_b[0]
        for t in range(1, T):
            log_alpha[t] = torch.logsumexp(log_alpha[t - 1].unsqueeze(1) + log_A, dim=0) + log_b[t]
        log_likelihood = torch.logsumexp(log_alpha[-1], dim=0)
        return log_alpha, log_likelihood

    def _backward(self, log_b: torch.Tensor) -> torch.Tensor:
        """Backward pass. Returns ``log_beta`` (T, K) with ``log_beta[T-1] = 0``."""
        T, K = log_b.shape
        log_A = self.log_transmat_
        log_beta = torch.zeros_like(log_b)
        for t in range(T - 2, -1, -1):
            log_beta[t] = torch.logsumexp(log_A + (log_b[t + 1] + log_beta[t + 1]).unsqueeze(0), dim=1)
        return log_beta

    def _e_step(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(log_gamma, log_xi_sum, log_likelihood)``."""
        log_b = self._log_emission(X)
        log_alpha, ll = self._forward(log_b)
        log_beta = self._backward(log_b)

        log_gamma = log_alpha + log_beta - ll
        log_gamma = log_gamma - torch.logsumexp(log_gamma, dim=1, keepdim=True)

        # log xi[t, i, j] = log_alpha[t, i] + log A[i, j] + log b[t+1, j] + log_beta[t+1, j] - ll
        log_xi = (
            log_alpha[:-1].unsqueeze(2)
            + self.log_transmat_.unsqueeze(0)
            + (log_b[1:] + log_beta[1:]).unsqueeze(1)
            - ll
        )  # (T-1, K, K)
        log_xi_sum = torch.logsumexp(log_xi, dim=0)  # (K, K)
        return log_gamma, log_xi_sum, ll

    def _m_step(self, X: torch.Tensor, log_gamma: torch.Tensor, log_xi_sum: torch.Tensor) -> None:
        """Closed-form M-step. Normalisations are done with log-sum-exp."""
        # Initial distribution and transition matrix (log space).
        log_pi = log_gamma[0]
        log_A = torch.clamp(log_xi_sum, min=self._log_floor)
        log_A = log_A - torch.logsumexp(log_A, dim=1, keepdim=True)

        # Per-state responsibilities, normalised over time in log space:
        # w[t, k] = gamma[t, k] / sum_t gamma[t, k]  (columns sum to one).
        log_g = torch.clamp(log_gamma, min=_MIN_LOG)
        log_Nk = torch.logsumexp(log_g, dim=0)  # (K,)
        w = torch.exp(log_g - log_Nk.unsqueeze(0))  # (T, K)

        means = w.T @ X  # (K, D)
        diff = X.unsqueeze(1) - means.unsqueeze(0)  # (T, K, D)
        wdiff = w.unsqueeze(2) * diff
        cov = torch.einsum("tkd,tke->kde", wdiff, diff)  # (K, D, D)
        covars = self._symmetrize_regularize(cov)

        self._store(log_pi, log_A, means, covars)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def _run_em(self, X: torch.Tensor, max_iters: int, tol: float) -> Tuple[List[float], bool]:
        T = X.shape[0]
        log_gamma, log_xi_sum, ll = self._e_step(X)
        history = [float(ll)]
        converged = False
        for it in range(1, max_iters + 1):
            self._m_step(X, log_gamma, log_xi_sum)
            log_gamma, log_xi_sum, ll_new = self._e_step(X)
            history.append(float(ll_new))
            delta = history[-1] - history[-2]
            if self.verbose:
                print(f"[EM] iter {it:4d}  loglik = {history[-1]:.6f}  delta = {delta:+.3e}")
            if abs(delta) / T < tol:
                converged = True
                break
        return history, converged

    @torch.no_grad()
    def fit(self, X: ArrayLike, max_iters: int = 200, tol: float = 1e-6) -> "GaussianHMM":
        """Baum-Welch EM.

        Convergence is declared when the log-likelihood improvement *per
        observation* falls below ``tol``. ``log_likelihood_history_[i]`` is the
        log-likelihood after ``i`` M-steps (index 0 is the initialisation), so
        for exact EM it is non-decreasing.
        """
        Xt = self._prepare(X)
        best: Optional[Dict[str, object]] = None
        self.all_run_log_likelihoods_ = []
        for run in range(self.n_init):
            self._initialize(Xt, self.seed + run)
            history, converged = self._run_em(Xt, int(max_iters), float(tol))
            self.all_run_log_likelihoods_.append(history[-1])
            if best is None or history[-1] > best["history"][-1]:
                best = {
                    "history": history,
                    "converged": converged,
                    "params": (
                        self.log_startprob_.clone(),
                        self.log_transmat_.clone(),
                        self.means_.clone(),
                        self.covars_.clone(),
                        self._chol.clone(),
                    ),
                }
        (
            self.log_startprob_,
            self.log_transmat_,
            self.means_,
            self.covars_,
            self._chol,
        ) = best["params"]
        self.log_likelihood_history_ = best["history"]
        self.log_likelihood_ = best["history"][-1]
        self.n_iter_ = len(best["history"]) - 1
        self.converged_ = bool(best["converged"])
        return self

    @torch.no_grad()
    def score(self, X: ArrayLike) -> float:
        """Total log-likelihood ``log p(x_{1:T})`` of a sequence."""
        self._check_fitted()
        Xt = self._prepare(X)
        _, ll = self._forward(self._log_emission(Xt))
        return float(ll)

    @torch.no_grad()
    def predict_proba(self, X: ArrayLike, causal: bool = True) -> torch.Tensor:
        """State posteriors, shape (T, K).

        ``causal=True``  : filtered ``P(z_t = k | x_{1:t})`` (forward pass only,
                           strictly no future information).
        ``causal=False`` : smoothed ``P(z_t = k | x_{1:T})`` (forward-backward).
        """
        self._check_fitted()
        Xt = self._prepare(X)
        log_b = self._log_emission(Xt)
        log_alpha, ll = self._forward(log_b)
        if causal:
            log_post = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
        else:
            log_beta = self._backward(log_b)
            log_post = log_alpha + log_beta - ll
            log_post = log_post - torch.logsumexp(log_post, dim=1, keepdim=True)
        return torch.exp(log_post)

    @torch.no_grad()
    def decode_viterbi(self, X: ArrayLike) -> torch.Tensor:
        """Most likely state path ``argmax_z p(z_{1:T}, x_{1:T})`` (LongTensor, shape (T,))."""
        self._check_fitted()
        Xt = self._prepare(X)
        log_b = self._log_emission(Xt)
        T, K = log_b.shape
        log_A = self.log_transmat_

        delta = torch.empty_like(log_b)
        psi = torch.zeros(T, K, dtype=torch.long, device=log_b.device)
        delta[0] = self.log_startprob_ + log_b[0]
        for t in range(1, T):
            scores = delta[t - 1].unsqueeze(1) + log_A  # (K_prev, K_next)
            best, arg = scores.max(dim=0)
            delta[t] = best + log_b[t]
            psi[t] = arg

        psi_np = psi.cpu().numpy()
        path = np.empty(T, dtype=np.int64)
        path[-1] = int(delta[-1].argmax().item())
        for t in range(T - 2, -1, -1):
            path[t] = psi_np[t + 1, path[t + 1]]
        return torch.from_numpy(path)

    def predict(self, X: ArrayLike) -> torch.Tensor:
        """Alias of :meth:`decode_viterbi`."""
        return self.decode_viterbi(X)

    @torch.no_grad()
    def sample(self, n_samples: int, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw ``(X, z)`` from the model. ``X``: (n, D) float64, ``z``: (n,) long."""
        self._check_fitted()
        gen = torch.Generator(device="cpu")
        gen.manual_seed(0 if seed is None else int(seed))
        pi = self.log_startprob_.exp().cpu()
        A = self.log_transmat_.exp().cpu()
        z = np.empty(n_samples, dtype=np.int64)
        z[0] = int(torch.multinomial(pi, 1, generator=gen).item())
        for t in range(1, n_samples):
            z[t] = int(torch.multinomial(A[z[t - 1]], 1, generator=gen).item())
        z_t = torch.from_numpy(z)
        D = int(self.n_features_)
        eps = torch.randn(n_samples, D, generator=gen, dtype=_DTYPE)
        L = self._chol.cpu()[z_t]  # (n, D, D)
        X = self.means_.cpu()[z_t] + torch.einsum("tde,te->td", L, eps)
        return X.to(self.device), z_t.to(self.device)
