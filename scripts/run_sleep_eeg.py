#!/usr/bin/env python
"""Unsupervised sleep staging on PhysioNet Sleep-EDF (subject 0, night 1).

* Features: z-scored log relative band power (delta, theta, alpha, sigma, beta) of
  the Fpz-Cz EEG channel in 30-second epochs.
* Model: K=5 full-covariance Gaussian HMM fitted without labels.
* Offline decoding: Viterbi path. States are then matched to the clinician's AASM
  stages with the Hungarian algorithm and scored with accuracy, macro-F1 and
  Cohen's kappa.

Outputs: artifacts/sleep_hypnogram.png and artifacts/sleep_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hmm_engine import (
    GaussianHMM,
    align_states_hungarian,
    calculate_classification_metrics,
    compute_bic,
    confusion_matrix,
    prepare_sleep_data,
)
from hmm_engine.neuro_pipeline import BAND_NAMES, EPOCH_SECONDS, STAGE_NAMES, Z_COLUMNS

# Conventional hypnogram vertical order (top -> bottom): Wake, REM, N1, N2, N3.
HYPNOGRAM_LEVEL = {0: 4, 4: 3, 1: 2, 2: 1, 3: 0}
LEVEL_LABELS = ["N3", "N2", "N1", "REM", "Wake"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", type=int, default=0)
    p.add_argument("--recording", type=int, default=1, help="night (1 or 2)")
    p.add_argument("--data-path", default=None, help="MNE data directory (default: ~/mne_data)")
    p.add_argument("--k", type=int, default=5, help="number of hidden states")
    p.add_argument("--n-init", type=int, default=5, help="EM restarts (best likelihood kept)")
    p.add_argument("--max-iters", type=int, default=300)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--reg-covar", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wake-margin", type=int, default=60, help="Wake epochs kept before/after sleep (30 min = 60)")
    p.add_argument("--bic-sweep", action="store_true", help="also report BIC for K = 2..8")
    p.add_argument("--out-dir", default=str(ROOT / "artifacts"))
    return p.parse_args(argv)


def hypnogram_steps(time_h: np.ndarray, stages: np.ndarray):
    """Step-plot coordinates (time in hours, vertical level) of a stage sequence."""
    level = np.array([HYPNOGRAM_LEVEL[int(s)] for s in stages], dtype=np.float64)
    t = np.append(time_h, time_h[-1] + EPOCH_SECONDS / 3600.0)
    return t, np.append(level, level[-1])


def plot_figure(
    path: Path,
    z: np.ndarray,
    time_h: np.ndarray,
    y_true: np.ndarray,
    y_hmm: np.ndarray,
    metrics: Dict[str, float],
    title: str,
    n_states: int,
) -> None:
    epoch_h = EPOCH_SECONDS / 3600.0
    fig, axes = plt.subplots(
        3, 1, figsize=(13, 8.5), sharex=True, gridspec_kw={"height_ratios": [1.4, 1.0, 1.0]}, layout="constrained"
    )

    # Panel 1: normalised band-power heat map (rows = bands, columns = epochs).
    im = axes[0].imshow(
        z.T,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-2.5,
        vmax=2.5,
        extent=(time_h[0], time_h[-1] + epoch_h, len(BAND_NAMES), 0),
    )
    axes[0].set_yticks(np.arange(len(BAND_NAMES)) + 0.5)
    axes[0].set_yticklabels([b.capitalize() for b in BAND_NAMES])
    axes[0].set_title(title)
    axes[0].set_ylabel("EEG band")
    cbar = fig.colorbar(im, ax=axes.tolist(), location="right", fraction=0.025, pad=0.01)
    cbar.set_label("z-scored log relative power")

    # Panel 2: clinician hypnogram.
    t, lvl = hypnogram_steps(time_h, y_true)
    axes[1].step(t, lvl, where="post", color="black", lw=1.2)
    axes[1].set_ylabel("Clinician (AASM)")
    axes[1].set_title("Clinician-scored hypnogram")

    # Panel 3: HMM Viterbi decoding aligned to the clinician label space.
    t, lvl = hypnogram_steps(time_h, y_hmm)
    axes[2].step(t, lvl, where="post", color="tab:blue", lw=1.2)
    axes[2].set_ylabel("HMM (Viterbi)")
    axes[2].set_title(
        f"HMM decoded hypnogram (K={n_states} states, Hungarian-aligned) - "
        f"accuracy {metrics['accuracy']:.1%}, macro-F1 {metrics['macro_f1']:.3f}, "
        f"Cohen's kappa {metrics['cohen_kappa']:.3f}"
    )
    axes[2].set_xlabel("Time since start of analysed segment (hours)")

    for ax in axes[1:]:
        ax.set_yticks(range(5))
        ax.set_yticklabels(LEVEL_LABELS)
        ax.set_ylim(-0.5, 4.5)
        ax.grid(alpha=0.25)
    axes[0].set_xlim(time_h[0], time_h[-1] + epoch_h)

    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv=None) -> Dict[str, object]:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching Sleep-EDF subject {args.subject}, night {args.recording} (first run downloads ~50 MB) ...")
    df, X = prepare_sleep_data(
        subject=args.subject,
        recording=args.recording,
        data_path=args.data_path,
        wake_margin_epochs=args.wake_margin,
    )
    y_true = df["stage"].to_numpy()
    time_h = df["time_h"].to_numpy()
    counts = {STAGE_NAMES[s]: int((y_true == s).sum()) for s in sorted(STAGE_NAMES)}
    print(f"  {df.height} epochs of {EPOCH_SECONDS:.0f} s | stage counts: {counts}")

    if args.bic_sweep:
        print("BIC sweep (lower is better):")
        for k in range(2, 9):
            m = GaussianHMM(k, reg_covar=args.reg_covar, seed=args.seed).fit(X, args.max_iters, args.tol)
            print(f"  K={k}: BIC = {compute_bic(m, X, m.log_likelihood_):,.1f}")

    print(f"Fitting K={args.k} Gaussian HMM on 5-D spectral features ({args.n_init} restarts) ...")
    model = GaussianHMM(
        n_states=args.k, reg_covar=args.reg_covar, init="kmeans", n_init=args.n_init, seed=args.seed
    )
    model.fit(X, max_iters=args.max_iters, tol=args.tol)
    print(
        f"  EM iterations: {model.n_iter_} | converged: {model.converged_} | "
        f"log-likelihood: {model.log_likelihood_:,.2f} | BIC: {compute_bic(model, X, model.log_likelihood_):,.1f}"
    )

    y_pred = model.decode_viterbi(X).numpy()
    y_aligned, mapping = align_states_hungarian(y_true, y_pred)
    metrics = calculate_classification_metrics(y_true, y_aligned)
    cm, labels = confusion_matrix(y_true, y_aligned, labels=sorted(STAGE_NAMES))

    print("\nHungarian state -> stage mapping:", {f"state {k}": STAGE_NAMES[v] for k, v in sorted(mapping.items())})
    print("\nConfusion matrix (rows = clinician, columns = HMM):")
    print("        " + " ".join(f"{STAGE_NAMES[int(l)]:>6s}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{STAGE_NAMES[int(l)]:>6s}  " + " ".join(f"{int(v):6d}" for v in cm[i]))

    print("\n| Accuracy | Macro-F1 | Cohen's kappa |")
    print("|---|---|---|")
    print(f"| {metrics['accuracy']:.3f} | {metrics['macro_f1']:.3f} | {metrics['cohen_kappa']:.3f} |")

    fig_path = out_dir / "sleep_hypnogram.png"
    plot_figure(
        fig_path,
        df.select(Z_COLUMNS).to_numpy(),
        time_h,
        y_true,
        y_aligned,
        metrics,
        f"Sleep-EDF subject {args.subject}, night {args.recording}: z-scored log relative EEG band power (Fpz-Cz)",
        args.k,
    )

    result = {
        "subject": args.subject,
        "recording": args.recording,
        "n_epochs": int(df.height),
        "k": args.k,
        "stage_counts": counts,
        "state_to_stage": {int(k): STAGE_NAMES[v] for k, v in mapping.items()},
        "confusion_matrix_rows_clinician_cols_hmm": cm.tolist(),
        "log_likelihood": float(model.log_likelihood_),
        "bic": compute_bic(model, X, model.log_likelihood_),
        **metrics,
    }
    with open(out_dir / "sleep_metrics.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved figure  -> {fig_path}")
    print(f"Saved metrics -> {out_dir / 'sleep_metrics.json'}")
    return result


if __name__ == "__main__":
    main()
