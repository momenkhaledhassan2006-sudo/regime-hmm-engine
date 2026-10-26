# regime-hmm-engine — Project Guide

*A companion to the README. The README tells you **what the code does**; this guide tells you **why it exists, how the mathematics works, how every file fits together, and what you need to learn to understand it**. It is written for a student who has not seen hidden Markov models before.*

> **Honest status note.** The code in this repository was written with AI assistance (Claude, Anthropic) from a written specification. It has been verified by 19 automated tests and by running both scripts on *synthetic* data. The two real-data benchmarks (SPY, Sleep-EDF) need an internet connection and have to be run by you; this guide contains **no real-data results** and tells you where to put yours.

---

## How to use this guide

| If you have... | Do this |
|---|---|
| 15 minutes | Read [Part 0](#part-0--the-project-in-one-page) and [Part 1](#part-1--goal-questions-and-claims) |
| 1 afternoon | Add [Part 3](#part-3--the-ideas-in-plain-language) (intuition and the toy example) |
| 1–2 weeks | Follow the [study roadmap in Part 2](#part-2--what-to-learn-first), then read [Part 4](#part-4--the-mathematics-step-by-step) and [Part 5](#part-5--the-code-file-by-file) next to the code |
| Before an interview / application | Finish with [Part 7](#part-7--exercises-learn-by-experiment) and [Part 8](#part-8--talking-about-this-project-honestly) |

---

## Part 0 — The project in one page

**One sentence.** `regime-hmm-engine` is a hidden Markov model (HMM) written in PyTorch that learns, *without being told the answers*, which hidden "regime" a time series is in — and is tested on two very different series: stock-market data (calm vs. turbulent regimes) and brain-wave data (sleep stages).

**The core idea.** Many real systems have a hidden *state* you cannot observe directly, and that state changes slowly. You only see noisy measurements that *depend* on the state.

| | Hidden state | What you can measure |
|---|---|---|
| Market | Calm / transitional / crash regime | Daily return, daily high–low range, 10-day volatility |
| Sleeping brain | Wake / N1 / N2 / N3 / REM | Power of EEG rhythms (delta, theta, alpha, sigma, beta) in each 30-second window |

**The pipeline in six steps**

1. Turn raw data into a small table of numbers per time step (*features*).
2. Standardise them (z-scores).
3. Fit an HMM by the Baum–Welch (EM) algorithm. No labels are used.
4. Ask the model for state probabilities, in one of two ways:
   - **filtering** — uses only the past (what a trader could really know), or
   - **smoothing / Viterbi** — uses the whole recording (what a sleep scorer can do afterwards).
5. Evaluate: trading performance for markets; agreement with clinicians for sleep.
6. Plot.

**Why it is interesting.** The same engine (`core.py`, about 550 lines including comments) handles both problems. What changes is only the features and the *question you are allowed to ask* (past-only vs. whole-sequence). The project makes that distinction concrete and testable.

---

## Part 1 — Goal, questions and claims

### 1.1 The goal

Build a **correct, numerically stable, from-scratch HMM engine** (no HMM library such as `hmmlearn`), and use it as a *research-style benchmark* to show that one probabilistic model can describe latent regimes in two unrelated domains while respecting the information constraints of each.

### 1.2 The questions it asks

**Finance question.** *If I train an HMM on 2005–2018 SPY data and then use only information available up to day t, can it recognise a crash-like regime in 2019–2026 — and would staying in cash during those days have changed risk-adjusted performance compared with buy-and-hold?*

**Neuroscience question.** *How closely does an unsupervised 5-state HMM, fed only five EEG band-power numbers per 30-second epoch, reproduce the sleep stages scored by human experts?* (Measured with accuracy, macro-F1 and Cohen's kappa after matching state labels with the Hungarian algorithm.)

**Method question.** *Can one implementation serve both domains, and can we prove — with tests, not just claim — that the "causal" mode never peeks into the future?*

**Engineering question.** *Can an HMM be implemented so that long sequences do not underflow, covariance matrices never become invalid, and the EM algorithm provably behaves (log-likelihood never decreases)?*

> The repository supplies the **method and the experiment**. The *answers* to the first two questions are numbers that appear when **you** run the scripts. Do not quote any answer until you have.

### 1.3 What the project tries to show

1. **Correctness.** The forward pass, backward pass and Viterbi path match brute-force enumeration on small problems; EM increases the likelihood every iteration; EM recovers known parameters from simulated data.
2. **Causal vs. offline inference.** Filtered probabilities at time *t* are unchanged if you alter any later data point; smoothed probabilities are not (`test_causality`). This is the difference between *tradable* and *retrospective*.
3. **Domain generality.** The identical engine works on a market feature matrix `(T, 3)` and an EEG feature matrix `(T, 5)`.
4. **Honest evaluation.** Normalisation uses training-window statistics only, the trading signal is lagged one day, and unsupervised states are matched to labels *only for scoring*, never for training.

### 1.4 Problems it solves (and where in the code)

| Problem | Why it is a problem | How it is solved | Where |
|---|---|---|---|
| **Numerical underflow** | Multiplying 1,000 probabilities of 0.01 gives 1e-2000, which float64 rounds to **0.0** | Work with logarithms and use *log-sum-exp* | `_forward`, `_backward`, `_e_step`, `_m_step` in `core.py` |
| **Invalid covariance matrices** | A state with few points, or collinear features, gives a singular matrix; its inverse and log-determinant blow up | Symmetrise, add `reg_covar · I`, factor with Cholesky, add escalating jitter if needed | `_symmetrize_regularize`, `_cholesky` |
| **Bad EM starting points** | EM only finds a *local* optimum; random starts can give poor solutions | k-means++ initialisation and several restarts; keep the best likelihood | `_kmeans`, `_initialize`, `fit(n_init=...)` |
| **Look-ahead bias** | Using future data (even through the z-score mean) makes a backtest look better than reality | Filtering only; z-scores from the training window; signal lagged one day | `predict_proba(causal=True)`, `zscore_features`, `build_strategy_frame` |
| **Label switching** | EM state "2" is not the same thing as clinician stage "N2"; numbering is arbitrary | Find the best one-to-one matching with the Hungarian algorithm | `align_states_hungarian` |
| **Choosing the number of states** | More states always fit better | Penalise complexity with BIC | `compute_bic` |
| **Evaluating unsupervised output** | Plain accuracy is misleading when classes are imbalanced (N2 dominates) | Macro-F1 and Cohen's kappa | `calculate_classification_metrics` |
| **Heterogeneous raw data** | EDF files, annotations, 30-s epochs, merged stages 3+4, long wake periods | MNE epoching, stage mapping, wake-margin cropping | `epochs_from_raw` |
| **Trusting your own code** | HMM code fails silently (plausible but wrong numbers) | Brute-force and synthetic-recovery tests | `tests/` |

### 1.5 Why an HMM rather than something simpler?

A fair question to expect from a professor: *"Why not just say 'crash' when volatility is above a threshold?"*

- A threshold flickers day to day. An HMM has a **transition matrix with large diagonal entries**, so it expects regimes to persist and ignores one-day blips.
- An HMM combines **several features at once** (via full covariance) rather than one number.
- It outputs a **probability**, not a yes/no, so you can choose a threshold that matches your risk appetite.
- The regimes are **learned**, not hand-defined.

The price: it assumes the Markov property, Gaussian emissions and a fixed number of states (see 1.6).

### 1.6 What the project does *not* show

- **It does not prove a trading edge.** One asset, one train/test split, no realistic execution model (costs are optional and default to zero). A single out-of-sample path can look good or bad by luck.
- **It does not show that unsupervised HMMs beat modern sleep-staging systems.** Published automatic stagers use deep networks, several channels (EOG, EMG) and many subjects. This uses one EEG channel and one night.
- **Single subject.** Nothing here says the sleep result would generalise to other people.
- **Gaussian emissions are an approximation.** Volatility features are heavy-tailed.
- **Fixed K.** K = 3 (finance) and K = 5 (sleep) are chosen by design; `--bic-sweep` lets you check.

---

## Part 2 — What to learn first

Priorities: **A** = needed to follow the code at all · **B** = needed to follow the math · **C** = nice context.
Times are rough, for a first pass at the "enough for this project" depth.

> **About the links.** I could not browse YouTube from my environment, so I cannot promise exact video titles. Items marked ✔ are pages I **did** confirm exist. For everything else I give a **channel** (recalled from memory — double-check) and a **search phrase** you can paste into YouTube. Pick a video that is recent, has good ratings, and works through a numerical example.

| # | Concept | Pri. | Why you need it here | Depth needed | Time | Where to learn |
|---|---|---|---|---|---|---|
| 1 | Python, NumPy arrays, shapes and broadcasting | A | Every line of `core.py` is tensor arithmetic with shapes like `(T, K, D)` | Be able to predict the output shape of `a.unsqueeze(1) + B` | 3 h | PyTorch official "Learn the Basics" tutorial; search *"numpy broadcasting explained"* |
| 2 | Conditional probability, joint, marginal, Bayes' rule | A | Forward/backward and posteriors are all conditional probabilities | Do 3–4 small Bayes problems by hand | 3 h | 3Blue1Brown, *"Bayes theorem, the geometry of changing beliefs"*; Khan Academy probability |
| 3 | Expectation, variance, covariance, correlation | A | Covariance matrices describe how features move together | Know what each entry of a 3×3 covariance matrix means | 2 h | StatQuest (search *"StatQuest covariance"*) |
| 4 | Gaussian and **multivariate** Gaussian | A | Each state emits a Gaussian vector | Know what mean and covariance do to the "cloud" shape | 2 h | Search *"multivariate normal distribution covariance matrix intuition"* |
| 5 | Matrices: product, transpose, inverse, determinant, eigenvalues | B | Transition matrix, covariances, Cholesky | Matrix product shapes; "positive definite = all eigenvalues > 0" | 6 h | 3Blue1Brown *Essence of linear algebra* (chapters on matrices, determinant, inverse, eigen) |
| 6 | Maximum likelihood estimation (MLE) | A | EM is "MLE with hidden variables" | Derive the MLE of a Gaussian mean by hand | 2 h | StatQuest *"Maximum Likelihood, clearly explained"* (search the title) |
| 7 | **K-means** clustering | A | Used to initialise the HMM | Run it by hand on 6 points | 1 h | StatQuest, *K-means clustering* ✔ (title confirmed in StatQuest's catalogue) |
| 8 | PCA (first principal component) | C | The optional `"quantile"` initialiser sorts by PC1 | Know PC1 = direction of largest variance | 1 h | StatQuest, *Principal Component Analysis (PCA), Step-by-Step* ✔ |
| 9 | **Markov chains** and transition matrices | A | The hidden state sequence is a Markov chain | Multiply a distribution by a transition matrix | 2 h | Search *"Markov chain explained transition matrix"*; channel Normalized Nerd has a Markov-chain/HMM series (verify) |
| 10 | **Hidden Markov models** (forward algorithm, Viterbi) | A | The whole project | Run forward and Viterbi on a 2-state, 3-step example | 6 h | ✔ Andrew Moore's "Hidden Markov Models" tutorial (Auton Lab): `https://autonlab.org/tutorials/hidden-markov-models.html` (covers dynamic programming, state estimation, most-likely path and learning with EM) · ✔ MIT OCW 7.91J Lecture 10 (video; HMMs and Viterbi, with a biology flavour): `https://ocw.mit.edu/courses/7-91j-foundations-of-computational-and-systems-biology-spring-2014/resources/lecture-10-markov-and-hidden-markov-models-of-genomic-and-protein-features/` · ✔ Wikipedia "Hidden Markov model" · YouTube search *"hidden markov model forward algorithm worked example"*, channels Serrano.Academy and mathematicalmonk (verify) |
| 11 | **Expectation–Maximisation (EM)** and Gaussian mixtures | A | Baum–Welch *is* EM | Explain E-step vs M-step in your own words | 5 h | Channels Victor Lavrenko and mathematicalmonk (verify); search *"EM algorithm Gaussian mixture model explained"* |
| 12 | **Log-sum-exp trick** | A | Why everything is in log space | Reproduce the −999.687 example in Part 4.9 | 1 h | Search *"log-sum-exp trick explained numerical stability"* |
| 13 | Cholesky decomposition, positive-definite matrices | B | Stable evaluation of Gaussian log-density | Know Σ = L Lᵀ and why it is cheaper than inverting | 1.5 h | Search *"Cholesky decomposition explained"* |
| 14 | Returns, log returns, volatility | A (finance) | The finance features | Compute a log return and a 10-day volatility by hand | 2 h | Search *"log returns vs simple returns"*, *"realized volatility explained"* |
| 15 | Backtesting, **look-ahead bias**, overfitting, Sharpe ratio, drawdown | A (finance) | The strategy and its evaluation | Explain why using future data invalidates a backtest | 3 h | Search *"look-ahead bias backtesting"*, *"Sharpe ratio explained"*, *"maximum drawdown explained"* |
| 16 | Fourier transform and power spectral density | B (EEG) | EEG features are band powers | Know "power at each frequency" | 3 h | 3Blue1Brown *"But what is the Fourier Transform?"* (recalled); Steve Brunton's Fourier series (verify) |
| 17 | Welch's method | B (EEG) | How the PSD is estimated per epoch | Know why averaging overlapping windows reduces noise | 1 h | Search *"Welch method power spectral density explained"* |
| 18 | EEG rhythms and sleep stages (AASM / R&K, hypnogram) | A (EEG) | What the labels and features mean | Name the stages and their signature rhythms | 3 h | Search *"sleep stages EEG delta theta alpha spindles K-complex"* and *"how to read a hypnogram"*; ✔ dataset page: `https://physionet.org/content/sleep-edfx/1.0.0/` |
| 19 | Confusion matrix, accuracy, precision/recall, F1 | A | Evaluation | Compute macro-F1 from a 3×3 matrix | 2 h | StatQuest *Confusion Matrix* (search) |
| 20 | Cohen's kappa | B | Chance-corrected agreement | Reproduce the example in Part 4.14 | 1 h | Search *"Cohen's kappa explained"* |
| 21 | AIC / BIC | B | Choosing K | Know "fit minus complexity penalty" | 1 h | Search *"AIC BIC model selection explained"* |
| 22 | Hungarian (assignment) algorithm | B | Label alignment | Solve a 3×3 assignment by hand | 1 h | Search *"Hungarian algorithm assignment problem"* |
| 23 | Polars, MNE-Python, yfinance basics | C | Data plumbing | Read a Polars `with_columns` chain | 3 h | Polars user guide; MNE "sleep stage classification from polysomnography" tutorial (search) |

**Suggested order (about four weeks at 6–8 h/week).**

- **Week 1 — foundations:** rows 1–8 (shapes, probability, Gaussians, MLE, k-means).
- **Week 2 — the model:** rows 9–13 (Markov chains → HMM → EM → log-space → Cholesky). Do the toy example in Part 3 by hand.
- **Week 3 — the domains:** rows 14–18 and 19–22 (finance concepts, EEG/sleep, metrics).
- **Week 4 — consolidate:** read Parts 4–5 against the code, do Part 7 exercises, rehearse Part 8.

---

## Part 3 — The ideas in plain language

### 3.1 Hidden vs. observed

Imagine a friend in another city who tells you only whether they took an umbrella. You never see the weather (hidden), only the umbrella (observed). From a string of umbrella reports you try to guess the weather sequence. An HMM is exactly that, with two ingredients:

- **Transition probabilities `A`**: how likely tomorrow's weather is given today's (weather is "sticky").
- **Emission distributions**: how likely each observation is given the weather.

In our project the "weather" is the market regime or the sleep stage, and the "umbrella" is a vector of numbers (volatility, EEG band powers) modelled as a **Gaussian**.

### 3.2 Three questions you can ask an HMM

| Question | Name | Uses | Code |
|---|---|---|---|
| "How likely is this data under my model?" | Likelihood (forward algorithm) | all data | `score` |
| "What is the probability of each state **now**, given the data **so far**?" | **Filtering** | past only | `predict_proba(causal=True)` |
| "What is the probability of each state at time t, given **all** the data?" | **Smoothing** | past and future | `predict_proba(causal=False)` |
| "What single sequence of states is most likely?" | **Viterbi decoding** | all data | `decode_viterbi` |
| "What are the best parameters?" | **Learning** (Baum–Welch/EM) | all data | `fit` |

### 3.3 A toy example you can do by hand

Two states (1 and 2), two time steps. Suppose

- start probabilities **π = (0.6, 0.4)**
- transition matrix **A = [[0.7, 0.3], [0.4, 0.6]]** (row = from, column = to)
- how well each state explains each observation (the emission density values):
  - observation 1: **b₁ = (0.5, 0.1)**
  - observation 2: **b₂ = (0.2, 0.6)**

**Forward pass.**

- α₁ = π ⊙ b₁ = (0.6·0.5, 0.4·0.1) = **(0.30, 0.04)**
- α₂(1) = (0.30·0.7 + 0.04·0.4) · 0.2 = 0.226 · 0.2 = **0.0452**
- α₂(2) = (0.30·0.3 + 0.04·0.6) · 0.6 = 0.114 · 0.6 = **0.0684**
- Likelihood of the data = 0.0452 + 0.0684 = **0.1136**

**Filtering** at t = 1 uses only α₁: (0.30, 0.04) / 0.34 = **(0.882, 0.118)** — "probably state 1".

**Smoothing** at t = 1 also asks "what do later observations say?". Backward quantities: β₂ = (1, 1); β₁(1) = 0.7·0.2·1 + 0.3·0.6·1 = 0.32; β₁(2) = 0.4·0.2 + 0.6·0.6 = 0.44. Then γ₁ ∝ α₁ ⊙ β₁ = (0.096, 0.0176), divided by 0.1136 gives **(0.845, 0.155)**.

The smoothed belief in state 1 dropped from 0.882 to 0.845, because the *second* observation looks much more like state 2, and state 1 → state 2 is possible. **That is look-ahead.** A trader at t = 1 does not have observation 2; a sleep scorer reviewing the whole night does.

**Viterbi.** δ₁ = (0.30, 0.04). δ₂(1) = max(0.30·0.7, 0.04·0.4)·0.2 = 0.042 (came from state 1); δ₂(2) = max(0.30·0.3, 0.04·0.6)·0.6 = 0.054 (came from state 1). The best end state is 2 (0.054), reached from state 1 → the most likely path is **1 → 2**.

You can check all of these numbers with the engine itself. The underscore methods are private and used here for learning only; the Gaussian parameters passed to `set_params` are irrelevant because we feed the emission densities `b` in directly:

```python
import torch
from hmm_engine import GaussianHMM

m = GaussianHMM(2).set_params([0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]], [[0.0], [1.0]], [[[1.0]], [[1.0]]])
b = torch.tensor([[0.5, 0.1], [0.2, 0.6]], dtype=torch.float64)   # b[t, k]
log_alpha, ll = m._forward(b.log())
log_beta = m._backward(b.log())
print(log_alpha.exp())                                   # [[0.30, 0.04], [0.0452, 0.0684]]
print(ll.exp())                                          # 0.1136
print((log_alpha[0] - torch.logsumexp(log_alpha[0], 0)).exp())   # filtered at t=1: [0.882, 0.118]
print((log_alpha + log_beta - ll).exp()[0])              # smoothed at t=1: [0.845, 0.155]
```

Doing the calculation once **by hand first**, and only then running this, is the single most useful exercise in this guide.

### 3.4 Why "unsupervised"?

Nobody tells the model "this day was a crash" or "this epoch is N2". EM starts with a guess (from k-means), computes how responsible each state is for each time step, re-estimates the parameters from those responsibilities, and repeats. The states that emerge are *clusters in feature space that persist in time*. Whether they mean "crash" or "N3" is something **you** infer afterwards — by inspecting their means (finance) or by comparing them with clinician labels (sleep).

---

## Part 4 — The mathematics, step by step

### 4.1 Notation

| Symbol | Meaning | Shape / range in the code |
|---|---|---|
| T | number of time steps | `X.shape[0]` |
| K | number of hidden states | `n_states` |
| D | number of features | `X.shape[1]` (3 finance, 5 sleep) |
| x_t | observation at time t | row `X[t]`, shape `(D,)` |
| z_t | hidden state at time t | integer 0…K−1 |
| π_k | P(z₁ = k) | `startprob_` `(K,)` |
| A_ij | P(z_t = j given z_{t−1} = i) | `transmat_` `(K, K)`, rows sum to 1 |
| μ_k, Σ_k | mean and covariance of state k | `means_` `(K, D)`, `covars_` `(K, D, D)` |
| b_k(x_t) | emission density of x_t under state k | `exp(log_b[t, k])` |
| α_t(k) | p(x₁…x_t, z_t = k) | `log_alpha[t, k]` (log) |
| β_t(k) | p(x_{t+1}…x_T given z_t = k) | `log_beta[t, k]` (log) |
| γ_t(k) | P(z_t = k given all data) | `exp(log_gamma[t, k])` |
| ξ_t(i,j) | P(z_t = i, z_{t+1} = j given all data) | summed over t in `log_xi_sum` |

### 4.2 The model

The hidden chain obeys the **Markov property** (tomorrow depends on today only), and each observation depends only on the current state:

$$
p(z_{1:T}, x_{1:T}) = \pi_{z_1}\, b_{z_1}(x_1)\prod_{t=2}^{T} A_{z_{t-1} z_t}\, b_{z_t}(x_t),
\qquad x_t \mid z_t = k \sim \mathcal{N}(\mu_k, \Sigma_k).
$$

The number of free parameters is p = (K−1) + K(K−1) + K·D + K·D(D+1)/2. For finance (K=3, D=3) that is **35**; for sleep (K=5, D=5) it is **124**.

### 4.3 The Gaussian emission and why Cholesky

For one state, the multivariate Gaussian log-density is

$$
\log b_k(x) = -\tfrac12\Big[D\log 2\pi + \log\det\Sigma_k + (x-\mu_k)^\top\Sigma_k^{-1}(x-\mu_k)\Big].
$$

Inverting Σ directly is slow and can be inaccurate. Instead factor **Σ = L Lᵀ** (L lower-triangular):

- log det Σ = 2 · Σ_d log L_dd (just the diagonal),
- (x−μ)ᵀΣ⁻¹(x−μ) = ‖L⁻¹(x−μ)‖², obtained by *solving* a triangular system.

In code (`_log_emission`):

```python
z = torch.linalg.solve_triangular(self._chol, diff.transpose(1, 2), upper=False)
maha = z.pow(2).sum(dim=1)                      # Mahalanobis distance squared
log_det = 2.0 * torch.log(torch.diagonal(self._chol, dim1=-2, dim2=-1)).sum(dim=-1)
log_b = -0.5 * (D * _LOG_2PI + log_det.unsqueeze(1) + maha)
```

All K states are processed at once ("batched"). During development I compared this function with `scipy.stats.multivariate_normal`; the two agreed to about 1e-15. (It is not a separate unit test, but the brute-force test in Part 5.7 would fail if the densities were wrong.)

### 4.4 Forward algorithm (likelihood and filtering)

Define α_t(k) = p(x₁,…,x_t, z_t = k). Then

$$
\alpha_1(k) = \pi_k\, b_k(x_1),\qquad
\alpha_t(j) = b_j(x_t)\sum_{i=1}^{K}\alpha_{t-1}(i)\,A_{ij}.
$$

*Reading it:* "to be in state j now, I could have come from any state i (sum over i), then I must explain today's observation (multiply by b_j)."

The data likelihood is p(x₁:T) = Σ_k α_T(k). **Filtering** just normalises one row: P(z_t = k given x₁:t) = α_t(k) / Σ_j α_t(j). Because α_t only uses x₁…x_t, filtering *cannot* use the future — this is the mathematical basis of `test_causality`.

Cost: O(T·K²).

### 4.5 Backward algorithm

β_t(i) = p(x_{t+1},…,x_T given z_t = i):

$$
\beta_T(i) = 1,\qquad \beta_t(i) = \sum_{j} A_{ij}\, b_j(x_{t+1})\,\beta_{t+1}(j).
$$

### 4.6 Smoothing posteriors

Multiplying "past + now" by "future" gives the joint of the whole data and the state at t: α_t(k) β_t(k) = p(x₁:T, z_t = k). Dividing by the likelihood gives

$$
\gamma_t(k) = P(z_t = k \mid x_{1:T}) = \frac{\alpha_t(k)\,\beta_t(k)}{p(x_{1:T})}.
$$

Likewise, the probability of a *transition* at time t:

$$
\xi_t(i,j) = P(z_t=i, z_{t+1}=j \mid x_{1:T}) = \frac{\alpha_t(i)\,A_{ij}\,b_j(x_{t+1})\,\beta_{t+1}(j)}{p(x_{1:T})}.
$$

### 4.7 Viterbi decoding

Replace "sum over previous states" by "take the best previous state":

$$
\delta_1(k)=\pi_k b_k(x_1),\quad
\delta_t(j)=b_j(x_t)\max_i\big[\delta_{t-1}(i)A_{ij}\big],\quad
\psi_t(j)=\arg\max_i\big[\delta_{t-1}(i)A_{ij}\big].
$$

Start from the state with the largest δ_T and follow ψ backwards. The result is the single most probable *whole path*. Note: this is **not** the same as picking argmax of γ_t at every t (that could give an impossible path, e.g. a transition with probability ≈ 0).

In code:

```python
scores = delta[t - 1].unsqueeze(1) + log_A        # (K_prev, K_next)
best, arg = scores.max(dim=0)
delta[t] = best + log_b[t]
psi[t] = arg
```

### 4.8 Learning: Baum–Welch (EM)

We want the parameters θ = (π, A, μ, Σ) that maximise log p(x₁:T; θ), but the states z are hidden. EM repeats two steps.

**Why it works (sketch).** For any distribution q over state sequences, Jensen's inequality gives

log p(x; θ) ≥ E_q[log p(x, z; θ)] − E_q[log q(z)].

Choosing q = p(z given x; θ_old) makes this bound *touch* the true log-likelihood at θ_old (that is the **E-step**). Maximising the bound over θ (the **M-step**) can only raise the true log-likelihood. Hence the log-likelihood **never decreases** — which `test_monotonic_log_likelihood` checks numerically.

**E-step.** Run forward and backward; compute γ_t(k) and ξ_t(i,j).

**M-step.** Because log p(x, z; θ) splits into separate sums for π, A and the emissions, each part has a closed-form maximiser:

$$
\pi_k=\gamma_1(k),\qquad
A_{ij}=\frac{\sum_{t=1}^{T-1}\xi_t(i,j)}{\sum_{t=1}^{T-1}\sum_{j'}\xi_t(i,j')},
$$

$$
\mu_k=\frac{\sum_t\gamma_t(k)\,x_t}{\sum_t\gamma_t(k)},\qquad
S_k=\frac{\sum_t\gamma_t(k)\,(x_t-\mu_k)(x_t-\mu_k)^\top}{\sum_t\gamma_t(k)},\qquad
\Sigma_k=\tfrac12(S_k+S_k^\top)+\varepsilon I.
$$

(The first two come from maximising a sum of counts × log-probabilities under "must sum to one" with a Lagrange multiplier; the Gaussian ones are the usual weighted MLE.) "Expected counts" is a good mental model: A_ij is the *expected number of i→j transitions divided by the expected number of departures from i*.

In code the weights are normalised in log space:

```python
log_Nk = torch.logsumexp(log_g, dim=0)           # log of sum_t gamma_t(k)
w = torch.exp(log_g - log_Nk.unsqueeze(0))       # each column sums to 1
means = w.T @ X
cov = torch.einsum("tkd,tke->kde", wdiff, diff)  # weighted covariance, all states at once
```

**Stopping rule.** Stop when the log-likelihood gain *per observation* is below `tol`, or after `max_iters`.

### 4.9 Why log space (with numbers)

- A product of 1,000 probabilities of 0.01 is 10⁻²⁰⁰⁰. In float64 (smallest ≈ 10⁻³⁰⁸) `0.01**1000` evaluates to exactly **0.0**. The *logarithm* is just 1000·ln 0.01 = **−4605.17**, no problem.
- Sums of probabilities become **log-sum-exp**: log(Σ e^{a_i}) = m + log Σ e^{a_i − m} with m = max a_i. Example: log(e⁻¹⁰⁰⁰ + e⁻¹⁰⁰¹). Naively, e⁻¹⁰⁰⁰ underflows to 0 and the answer is −∞; the stable form gives −1000 + log(1 + e⁻¹) = **−999.687**.

In `core.py` every recursion is one `torch.logsumexp` over a broadcast sum:

```python
# forward: (K,1) + (K,K) -> entry [i, j] = log alpha_{t-1}(i) + log A_ij ; sum over i (dim 0)
log_alpha[t] = torch.logsumexp(log_alpha[t - 1].unsqueeze(1) + log_A, dim=0) + log_b[t]
# backward: (K,K) + (1,K) -> entry [i, j] = log A_ij + log b_j + log beta_j ; sum over j (dim 1)
log_beta[t] = torch.logsumexp(log_A + (log_b[t + 1] + log_beta[t + 1]).unsqueeze(0), dim=1)
```

### 4.10 Keeping covariances valid

A weighted covariance S_k is positive *semi*-definite (it can have zero eigenvalues). Adding εI shifts every eigenvalue up by ε, making it positive *definite* (`reg_covar`, default 1e-6; scripts use 1e-4 for finance and 1e-3 for EEG). The symmetry projection ½(S + Sᵀ) removes floating-point asymmetry. If Cholesky still fails, `_cholesky` adds jitter ×10 per attempt. A separate floor (`transition_floor = 1e-12`) keeps probabilities from reaching exactly 0, whose log is −∞.

### 4.11 Initialisation (k-means++)

`_kmeans` picks the first centre at random and each next centre with probability proportional to its squared distance from the nearest chosen centre (spreading them out), then runs Lloyd iterations (assign → recompute centres). The cluster labels give initial means and covariances; the *sequence* of labels gives initial transition counts (+1 smoothing). `n_init > 1` repeats with seeds `seed, seed+1, …` and keeps the run with the highest final log-likelihood, which guards against bad local optima.

### 4.12 BIC — choosing the number of states

$$
\mathrm{BIC} = p\ln T - 2\ln \hat L.
$$

Lower is better. The first term punishes complexity, the second rewards fit. *Caveat:* BIC assumes independent observations; time-series data are dependent, so treat it as a **guide**, not a theorem. Also, never choose K by looking at *test-set* performance.

### 4.13 The Hungarian algorithm — fixing label switching

EM's state 3 may correspond to clinician stage N2. Build the **contingency matrix** C (rows = HMM state, columns = true stage, entries = number of epochs in both). Example:

| | Wake | N2 | N3 |
|---|---|---|---|
| State A | 90 | 5 | 5 |
| State B | 10 | 70 | 20 |
| State C | 0 | 25 | 75 |

Choose a one-to-one matching maximising the total on the matched cells: A→Wake (90), B→N2 (70), C→N3 (75), total **235 of 300 = 78.3 %** agreement. `scipy.optimize.linear_sum_assignment(C, maximize=True)` finds it efficiently (brute force would try K! matchings). Important honesty point: this uses the labels, so it is **evaluation only**; the HMM never saw them. If there are more HMM states than true classes, the leftover states are mapped to whichever class they overlap most.

### 4.14 Accuracy, macro-F1 and Cohen's kappa

- **Accuracy** = fraction correct. Misleading when one class (N2) dominates.
- **F1 of a class** = 2·TP / (2·TP + FP + FN); **macro-F1** = unweighted mean over classes, so a rare stage (N1) counts as much as a common one.
- **Cohen's kappa** corrects for agreement by chance: κ = (pₒ − pₑ) / (1 − pₑ), with pₒ = observed agreement, pₑ = agreement expected if both raters labelled independently with their own label frequencies. Example: pₒ = 0.70, pₑ = 0.30 → κ = 0.40 / 0.70 = **0.571**. κ = 1 is perfect, κ = 0 is chance. Sleep-scoring literature quotes kappa for human-human agreement (commonly reported in the 0.7–0.8 range — check a source before citing it).

### 4.15 Finance features

- **Log return:** r_t = ln(P_t / P_{t−1}). Additive over time, symmetric for ups and downs.
- **Parkinson volatility** (uses the day's high H and low L): σ_P = √( (ln(H/L))² / (4 ln 2) ). The range of a day carries more information about volatility than the close-to-close change.
- **10-day realised volatility:** √( Σ_{i=0}^{9} r_{t−i}² ).
- **z-score:** z = (x − μ_train) / σ_train. Using μ and σ from the **training window only** matters: full-sample statistics would leak information about the test period into the inputs.

### 4.16 The strategy and its statistics

- Position on day t: 1 (invested) if P(Crash on day t−1 given data up to t−1) ≤ 0.5, else 0 (cash). The one-day lag exists because the probability uses day t−1's close.
- Strategy log-return = position · r_t − cost · |Δposition|. Wealth = exp(cumulative sum of log returns).
- **CAGR** = W_end^{252/n} − 1. **Annual volatility** = std(daily simple returns)·√252. **Sharpe** (risk-free = 0) = mean·252 / annual vol. **Max drawdown** = min over time of (wealth / running maximum − 1).
- The "Crash" state is *defined after fitting* as the state with the highest mean z-scored volatility (Parkinson and realised), so it is meaningful regardless of how EM numbered the states.

### 4.17 EEG features and Welch's method

A 30-second epoch at 100 Hz has 3,000 samples. To measure how much power sits at each frequency:

1. Cut it into overlapping 4-second Hann-windowed segments (400 samples, 50 % overlap → 14 segments, frequency resolution 0.25 Hz).
2. Compute each segment's periodogram (squared magnitude of its Fourier transform).
3. **Average** them (Welch's method) — this trades frequency resolution for a much less noisy estimate.

Then sum the PSD bins inside five bands: delta 0.5–4, theta 4–8, alpha 8–12, sigma 12–16 (spindles), beta 16–30 Hz; divide by the total over the five bands (**relative power**, sums to 1); take the **natural log**; z-score each feature across the night. Rough physiology (simplified): alpha rises in relaxed wake with eyes closed; theta in light sleep; sigma bursts (spindles) in N2; large delta in N3 (slow-wave sleep); beta/low-voltage fast activity in active wake and REM. With **one EEG channel and no muscle/eye channels**, REM vs. light sleep is intrinsically hard.

The dataset: Sleep-EDF Expanded has 197 whole-night recordings; EEG and EOG are sampled at 100 Hz, and hypnograms were scored manually by trained technicians following the 1968 Rechtschaffen & Kales manual (✔ PhysioNet dataset page). The project merges R&K stages 3 and 4 into N3 to match the modern AASM convention.

---

## Part 5 — The code, file by file

### 5.0 Layout and data flow

```
regime-hmm-engine/
├── pyproject.toml             dependencies and pytest settings
├── src/hmm_engine/
│   ├── core.py                GaussianHMM  (the engine)
│   ├── metrics.py             BIC, Hungarian alignment, accuracy / F1 / kappa
│   ├── finance_pipeline.py    SPY  -> Polars features -> tensor (T, 3)
│   └── neuro_pipeline.py      EDF  -> MNE epochs -> Welch -> Polars -> tensor (T, 5)
├── scripts/
│   ├── run_finance_regimes.py fit K=3, causal test, strategy, figure
│   └── run_sleep_eeg.py       fit K=5, Viterbi, alignment, scores, figure
└── tests/                     19 tests
```

```
FINANCE:  yfinance -> OHLC (Polars) -> features -> z-score(train stats) -> X (T,3)
              -> GaussianHMM.fit(X_train) -> predict_proba(causal=True) -> P(Crash)
              -> lag 1 day -> positions -> returns -> stats + figure

SLEEP:    EDF + hypnogram -> MNE Raw -> 30 s epochs + stage labels -> Welch PSD
              -> log relative band power -> z-score -> X (T,5)
              -> GaussianHMM.fit(X) -> decode_viterbi -> Hungarian -> metrics + figure
```

### 5.1 `src/hmm_engine/core.py` — `GaussianHMM`

Read it in this order.

| Piece | What it does | Math |
|---|---|---|
| `_kmeans`, `_quantile_labels` | Hard initial labels for each time step | 4.11 |
| `__init__` | Stores settings (`n_states`, `reg_covar`, `init`, `n_init`, `seed`, `transition_floor`, …) and checks them | — |
| `_prepare` | Converts input to a 2-D `float64` tensor, rejects NaN/inf, checks the feature count | — |
| `_symmetrize_regularize`, `_cholesky` | Keep Σ valid; return Σ and its Cholesky factor L | 4.10 |
| `_store` | Floors probabilities, renormalises in log space, saves all parameters | 4.10 |
| `set_params` | Set parameters by hand (used by tests and `sample`) | — |
| `_initialize` | k-means labels → initial μ, Σ, A, π | 4.11 |
| `_log_emission` | `(T, K)` matrix of log b_k(x_t) | 4.3 |
| `_forward` | log α and log-likelihood | 4.4 |
| `_backward` | log β | 4.5 |
| `_e_step` | log γ and the time-summed log ξ | 4.6 |
| `_m_step` | New π, A, μ, Σ | 4.8 |
| `_run_em` | The EM loop with convergence test; returns the log-likelihood history | 4.8 |
| `fit` | Runs `n_init` restarts, keeps the best, stores results | 4.11 |
| `score` | log p(x₁:T) | 4.4 |
| `predict_proba(X, causal=...)` | Filtering (`True`) or smoothing (`False`) | 4.4, 4.6 |
| `decode_viterbi` | Most likely path | 4.7 |
| `sample` | Simulate data from the model (used for tests) | 4.2 |

Details worth understanding:

- **`log_likelihood_history_[i]`** is the log-likelihood after `i` M-steps; entry 0 is the starting guess. The monotonicity test reads this list.
- **`_e_step` clamps/renormalises** `log_gamma` rows with another `logsumexp`, which removes tiny floating-point drift.
- **`_m_step` clamps `log_gamma` at −745.** If a state ends up with essentially zero responsibility everywhere, the weights fall back to uniform instead of producing NaN (that state then inherits the global mean/covariance). Real runs rarely hit this; it is a safety net.
- **Python loops over time.** Each pass loops over t in Python (the recursion is sequential) but is vectorised over states. That is why thousands of steps take about a second per EM iteration on a CPU.
- **Memory.** `log_xi` has shape `(T−1, K, K)`; fine for K ≤ 10 and T ≤ 10⁵.

### 5.2 `src/hmm_engine/metrics.py`

- `compute_bic(model, X, log_likelihood=None)` → p·ln T − 2·LL using `model.n_parameters`.
- `confusion_matrix(y_true, y_pred, labels=None)` → `(matrix, labels)`; rows true, columns predicted.
- `align_states_hungarian(y_true, y_pred)` → `(aligned_predictions, mapping)`; builds the contingency matrix (4.13).
- `calculate_classification_metrics(y_true, y_pred_aligned)` → dict with `accuracy`, `macro_f1`, `cohen_kappa`, computed from the confusion matrix (4.14).

### 5.3 `src/hmm_engine/finance_pipeline.py`

| Function | Role |
|---|---|
| `download_ohlc` | yfinance → adjusted daily OHLCV as a Polars frame (`auto_adjust=True`, so splits/dividends are handled) |
| `compute_features` | Adds `log_return`, `parkinson_vol`, `realized_vol_10d`; drops warm-up rows and bad prices |
| `zscore_features` | Adds `z_*` columns with mean/std from `date ≤ train_end` |
| `prepare_finance_data` | Chains the three and returns `(DataFrame, float64 tensor (T,3))`. Accepts `ohlc=` so you can inject data (the tests do) |

### 5.4 `src/hmm_engine/neuro_pipeline.py`

| Function | Role |
|---|---|
| `fetch_sleep_edf` | MNE's downloader for one subject/night; returns file paths |
| `epochs_from_raw` | Keeps the `EEG Fpz-Cz` channel, converts hypnogram annotations to 30-s events (`chunk_duration`), maps stages (W→0, 1→1, 2→2, 3 and 4→3, R→4), drops unscored epochs, trims long Wake periods to 30 min each side, returns `(epochs, labels, sfreq)` |
| `load_epoched_eeg` | Reads the two EDF files, attaches annotations, calls `epochs_from_raw` |
| `compute_bandpower_features` | Welch PSD → band sums → relative → log (4.17) |
| `build_sleep_dataframe` | Polars table with raw and z-scored features; returns `(DataFrame, tensor (T,5))` |
| `prepare_sleep_data` | The whole chain |

### 5.5 `scripts/run_finance_regimes.py`

1. `prepare_finance_data(...)` and split at `--train-end`.
2. Fit `GaussianHMM(K=3, n_init=3, reg_covar=1e-4)` on the training rows.
3. `name_states_by_volatility`: sort states by mean z-volatility → "Calm", "Transition", "Crash".
4. Filter the **concatenated** train+test series with frozen parameters (still causal; lets the test period start from a sensible belief).
5. `build_strategy_frame`: lag the signal one day, compute positions, turnover and returns (Polars).
6. `performance_stats` for buy-and-hold and the strategy; `plot_figure` for the two-panel figure; save `artifacts/finance_metrics.json`.

Useful flags: `--threshold`, `--cost-bps`, `--k`, `--bic-sweep`, `--train-end`.

### 5.6 `scripts/run_sleep_eeg.py`

1. `prepare_sleep_data(...)`.
2. Fit `GaussianHMM(K=5, n_init=5, reg_covar=1e-3)`.
3. `decode_viterbi` → `align_states_hungarian` → `calculate_classification_metrics`; print the confusion matrix.
4. `plot_figure`: band-power heat map, clinician hypnogram, HMM hypnogram (stages drawn in the usual order Wake, REM, N1, N2, N3 from top to bottom).

### 5.7 The tests, and the bug each one would catch

| Test | What it proves | Typical bug it catches |
|---|---|---|
| `test_monotonic_log_likelihood` | EM never lowers the likelihood | Wrong M-step formula, un-normalised weights |
| `test_synthetic_recovery` | Learns the true A (0.95 diagonal) and means from simulated data | Mixed-up transpose in A, label errors |
| `test_causality` | Filtering ignores the future; smoothing does not | Accidentally using β in "causal" mode |
| `test_forward_matches_brute_force_and_viterbi_is_optimal` | Likelihood and Viterbi equal exhaustive enumeration of all 2⁸ paths | Off-by-one in recursions |
| `test_posteriors_are_distributions` | Rows are probabilities; filter = smoother at the last step | Normalisation errors |
| `test_covariances_stay_positive_definite_on_degenerate_data` | Perfectly collinear features do not crash | Singular matrices |
| `test_invalid_inputs` | NaN, bad K, unfitted model give clear errors | Silent garbage |
| `tests/test_metrics.py` (6 tests) | BIC formula, Hungarian recovers a permutation, F1/kappa vs. hand computation | Metric mistakes |
| `tests/test_pipelines.py` (6 tests) | Feature formulas, train-only z-scores, MNE epoching, an offline end-to-end sleep run | Leakage, wrong stage mapping |

**Honest limitation of the pipeline tests:** they use *synthetic* signals whose spectra differ strongly by stage, so a perfect score there only proves the plumbing works. Real EEG is far messier.

---

## Part 6 — Running it and reading the results

```bash
pip install -e .
pytest                                    # 19 tests, offline
python scripts/run_finance_regimes.py --bic-sweep
python scripts/run_sleep_eeg.py --bic-sweep
```

**Finance: what to look at**

1. The printed regimes: is "Crash" the one with the highest volatility? Is its expected duration (1/(1−A_ii) days) plausible (days to weeks, not months)?
2. The bottom panel: does P(Crash) rise during known stress episodes in the test period *at or after* they begin (a causal filter cannot anticipate them)? Is it noisy or stable?
3. The table: compare drawdown and Sharpe, not just final wealth. Being in cash lowers volatility but can miss rebounds.
4. Sanity check: change `--threshold` to 0.3 and 0.7. If conclusions flip wildly, the result is fragile.

**Sleep: what to look at**

1. The Hungarian mapping: is each HMM state matched to a *different* stage? Duplicated or unused stages mean the model merged or split stages.
2. The confusion matrix: expect confusion among N1/N2/REM and between Wake/N1.
3. Compare accuracy with the **majority-class baseline** (the share of the most common stage); accuracy alone can be flattering.
4. The heat map: delta should be bright in N3 stretches, sigma in N2, alpha in Wake. If not, the features, not the HMM, are the problem.

**Red flags**

- Log-likelihood that decreases (should never happen).
- A state with a tiny covariance that explains only a few points (over-fitting; raise `reg_covar` or reduce K).
- Results that change a lot with `--seed` (bad local optima; raise `--n-init`).
- Great test performance after you tuned K or the threshold *on the test set* (that is overfitting the backtest).

---

## Part 7 — Exercises (learn by experiment)

1. **By hand:** reproduce the Part 3 numbers; then write them as a 6-line Python script with NumPy.
2. **Underflow:** compute the likelihood of a 1,000-step sequence in probability space (it prints 0.0) and in log space.
3. **Prove causality yourself:** modify one observation late in a sequence and plot filtered vs. smoothed probabilities at an earlier time.
4. **Sticky vs. flickery:** set the transition matrix to 0.6/0.4 on the diagonal and simulate. How do the filtered probabilities and Viterbi path change?
5. **Choose K:** run `--bic-sweep` on both datasets. Does BIC agree with K = 3 and K = 5? What does a disagreement mean?
6. **Initialisation:** compare `init="kmeans"`, `init="quantile"` and `n_init=1` vs. `5` — how much do the final log-likelihoods differ?
7. **Covariance structure:** change the code to diagonal covariances. How do BIC and the sleep scores change? (Diagonal needs fewer parameters but ignores feature correlations.)
8. **Features:** add a fourth finance feature (e.g. 20-day volatility) or replace volatilities by their logarithm; see how the regimes change.
9. **Strategy honesty:** add `--cost-bps 5` and 10. What happens to the advantage over buy-and-hold?
10. **Smoothed vs. filtered trading:** (for understanding only!) run the strategy with smoothed probabilities and see how much better it looks. That gap is the size of the look-ahead bias.

---

## Part 8 — Talking about this project honestly

### 8.1 What you can claim

- "I built / ran / evaluated a Gaussian HMM **without HMM libraries**, **with AI assistance**, verified by 19 automated tests."
- "I applied it to two domains and evaluated them with leakage-safe protocols."
- "I understand the forward–backward algorithm, EM and Viterbi well enough to explain them" — **only after you can**. Use Part 3 and Part 7 to get there.

### 8.2 What not to claim

- Do not say you invented the algorithm, or that you wrote it "from scratch" without mentioning AI assistance.
- Do not describe unit-test results on *synthetic* data as research results.
- Do not claim a profitable strategy or clinically useful sleep staging.
- Do not quote placeholder numbers in your CV/SOP. Replace them with your actual output.

Some programmes ask applicants to disclose AI assistance; a brief, accurate statement (see `SOP_and_CV_text.md`) is better than an evasive one, and it is easier to defend in an interview.

### 8.3 Questions a reader may ask — with short answers

1. **What is an HMM?** A latent Markov chain of discrete states, each emitting an observation from a state-specific distribution; we infer the hidden chain from the observations.
2. **Why log space?** Probabilities of long sequences underflow float64; logs and log-sum-exp keep everything finite and accurate.
3. **Filtering vs. smoothing — why does it matter?** Filtering uses data up to t only, which is the only legitimate choice for trading; smoothing uses the whole sequence, which is fine for retrospective analysis such as sleep scoring but would leak the future into a backtest.
4. **How do you know EM is implemented correctly?** The log-likelihood never decreases, it recovers known parameters from simulated data, and forward/Viterbi equal brute-force enumeration on small cases.
5. **How did you avoid look-ahead bias?** z-scores from training data only, filtered probabilities only, signal lagged one day, no tuning on the test set.
6. **Why K = 3 and K = 5?** K = 3 gives an interpretable calm/transition/crash story; K = 5 matches the five AASM stages. I check both with BIC, which is only a guide because observations are dependent.
7. **What is label switching?** EM state numbers are arbitrary; I match states to stages with the Hungarian algorithm for evaluation only.
8. **Why full covariance?** The features are correlated (e.g. two volatility measures); a full covariance models that at the cost of more parameters.
9. **Why might N1 and REM be poorly recovered?** Their single-channel spectra overlap with Wake and N2 and human scorers rely on eye and muscle signals this project does not use.
10. **What are the main limitations?** Gaussian emissions, fixed K, one subject, one asset and period, parameter stationarity after training.
11. **What did the AI do, and what did you do?** (Answer truthfully, e.g.: it generated the initial code from my specification; I ran the tests and scripts, studied the derivations, and evaluated the results.)
12. **What would you do next?** Student-t or mixture emissions, multiple subjects with cross-validation, adding EOG/EMG features, time-varying transition probabilities, comparing with a change-point detector.

---

## Part 9 — Glossary

| Term | Meaning |
|---|---|
| **AASM** | American Academy of Sleep Medicine scoring rules (stages W, N1, N2, N3, REM) |
| **Baum–Welch** | EM algorithm for HMMs |
| **BIC** | Bayesian Information Criterion; fit minus complexity penalty |
| **Causal filter** | Estimate at time t using data up to t only |
| **Cholesky factor** | Lower-triangular L with Σ = LLᵀ |
| **Cohen's kappa** | Agreement between two labellings corrected for chance |
| **Covariance matrix** | Matrix of variances (diagonal) and co-movements (off-diagonal) |
| **Emission** | Distribution of the observation given the hidden state |
| **EM** | Expectation–Maximisation |
| **Epoch** | A fixed time window (here 30 s) with one stage label |
| **Filtering / smoothing** | P(state_t given past) / P(state_t given all data) |
| **Hypnogram** | Plot or list of sleep stages over the night |
| **Label switching** | Arbitrary numbering of unsupervised states |
| **Look-ahead bias** | Using information that would not have been available at decision time |
| **Local optimum** | A solution that is best nearby but not globally |
| **Log-sum-exp** | Stable way to compute log of a sum of exponentials |
| **Markov property** | The future depends on the past only through the present state |
| **Parkinson estimator** | Volatility estimator from the daily high and low |
| **PSD** | Power spectral density: power per unit frequency |
| **Realised volatility** | Square root of the sum of squared returns over a window |
| **Regime** | Persistent condition of the system (calm/crash, sleep stage) |
| **Sharpe ratio** | Mean excess return divided by its standard deviation (annualised) |
| **Viterbi path** | Single most probable sequence of hidden states |
| **Welch's method** | PSD estimate by averaging periodograms of overlapping windows |
| **z-score** | (value − mean) / standard deviation |

---

## References and further reading (confirmed links)

- Andrew Moore, *Hidden Markov Models* tutorial (Auton Lab): https://autonlab.org/tutorials/hidden-markov-models.html
- MIT OCW 7.91J, Lecture 10 (video; Markov and hidden Markov models, Viterbi): https://ocw.mit.edu/courses/7-91j-foundations-of-computational-and-systems-biology-spring-2014/resources/lecture-10-markov-and-hidden-markov-models-of-genomic-and-protein-features/
- Wikipedia, *Hidden Markov model*: https://en.wikipedia.org/wiki/Hidden_Markov_model
- PhysioNet, *Sleep-EDF Database Expanded*: https://physionet.org/content/sleep-edfx/1.0.0/
- A classic tutorial to read once you know the basics: L. R. Rabiner (1989), *A tutorial on hidden Markov models and selected applications in speech recognition*, Proceedings of the IEEE (look it up by title).
- Parkinson, M. (1980), *The extreme value method for estimating the variance of the rate of return*, Journal of Business.
- Welch, P. D. (1967), *The use of fast Fourier transform for the estimation of power spectra*, IEEE Trans. Audio Electroacoustics.
- Kemp, B. et al. (2000), *Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG*, IEEE-BME (the Sleep-EDF paper).
- Rechtschaffen & Kales (1968), *A manual of standardized terminology, techniques and scoring system for sleep stages of human subjects*.

*(Bibliographic details for the papers are from memory; verify them before citing.)*
