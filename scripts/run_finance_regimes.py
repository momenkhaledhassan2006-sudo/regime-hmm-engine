#!/usr/bin/env python
"""SPY market-regime detection with a causal Gaussian HMM.

* Fit a K=3 HMM on the training window (default 2005-01-01 .. 2018-12-31).
* Evaluate out of sample (2019 onwards) with *forward-filtered* posteriors
  P(z_t | x_{1:t}) only -- no smoothing, hence zero look-ahead.
* Strategy: stay in cash on day t+1 whenever P(Crash | x_{1:t}) > threshold
  (signal known at the close of day t, applied to the return of day t+1);
  otherwise hold SPY. Compared with buy-and-hold.

Outputs: artifacts/finance_regimes.png and artifacts/finance_metrics.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl
import torch
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from hmm_engine import GaussianHMM, compute_bic, prepare_finance_data
from hmm_engine.finance_pipeline import Z_COLUMNS

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2005-01-01")
    p.add_argument("--end", default="2026-12-31", help="exclusive end date passed to yfinance")
    p.add_argument("--train-end", default="2018-12-31", help="last training day (inclusive)")
    p.add_argument("--k", type=int, default=3, help="number of hidden states")
    p.add_argument("--n-init", type=int, default=3, help="EM restarts (best likelihood kept)")
    p.add_argument("--max-iters", type=int, default=200)
    p.add_argument("--tol", type=float, default=1e-6, help="EM tolerance on per-observation log-likelihood gain")
    p.add_argument("--reg-covar", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.5, help="go to cash when P(Crash) exceeds this")
    p.add_argument("--cost-bps", type=float, default=0.0, help="one-way transaction cost in basis points")
    p.add_argument("--bic-sweep", action="store_true", help="also report BIC for K = 2..6 on the training data")
    p.add_argument("--out-dir", default=str(ROOT / "artifacts"))
    return p.parse_args(argv)


def name_states_by_volatility(model: GaussianHMM, k: int):
    """Rank states by mean z-scored volatility (Parkinson + realised) -> names/colours."""
    vol_cols = [Z_COLUMNS.index("z_parkinson_vol"), Z_COLUMNS.index("z_realized_vol_10d")]
    score = model.means_[:, vol_cols].mean(dim=1)
    order = torch.argsort(score).tolist()  # ascending volatility

    names: Dict[int, str] = {}
    for rank, state in enumerate(order):
        if rank == 0:
            names[state] = "Calm"
        elif rank == k - 1:
            names[state] = "Crash"
        else:
            names[state] = "Transition" if k == 3 else f"Transition {rank}"

    if k == 3:
        palette = ["#2e9e5b", "#f0a30a", "#d62839"]
    else:
        palette = [matplotlib.colors.to_hex(c) for c in plt.get_cmap("RdYlGn_r")(np.linspace(0.1, 0.9, k))]
    colors = {state: palette[rank] for rank, state in enumerate(order)}
    return order, names, colors


def performance_stats(log_returns: np.ndarray, in_market: np.ndarray) -> Dict[str, float]:
    """Summary statistics of a daily log-return stream (risk-free rate = 0)."""
    simple = np.expm1(log_returns)
    equity = np.concatenate([[1.0], np.exp(np.cumsum(log_returns))])
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    n = len(log_returns)
    vol = float(simple.std(ddof=1) * np.sqrt(TRADING_DAYS))
    return {
        "final_wealth": float(equity[-1]),
        "cagr": float(equity[-1] ** (TRADING_DAYS / n) - 1.0),
        "annual_volatility": vol,
        "sharpe": float(simple.mean() * TRADING_DAYS / vol) if vol > 0 else float("nan"),
        "max_drawdown": float(drawdown.min()),
        "fraction_in_market": float(in_market.mean()),
    }


def build_strategy_frame(
    df: pl.DataFrame,
    p_crash: np.ndarray,
    state: np.ndarray,
    threshold: float,
    cost_bps: float,
) -> pl.DataFrame:
    """Daily positions and returns. The position on day t uses P(Crash) from day t-1."""
    frame = (
        df.select("date", "log_return")
        .with_columns(pl.Series("p_crash", p_crash), pl.Series("state", state))
        .with_columns(
            (pl.col("p_crash").shift(1).fill_null(0.0) <= threshold).cast(pl.Float64).alias("position")
        )
        .with_columns(pl.col("position").diff().abs().fill_null(0.0).alias("turnover"))
        .with_columns(
            (pl.col("position") * pl.col("log_return") - pl.col("turnover") * cost_bps * 1e-4).alias(
                "strategy_log_return"
            )
        )
    )
    return frame


def plot_figure(
    path: Path,
    dates: np.ndarray,
    bh_cum: np.ndarray,
    strat_cum: np.ndarray,
    state: np.ndarray,
    p_crash: np.ndarray,
    names: Dict[int, str],
    colors: Dict[int, str],
    crash_state: int,
    order: List[int],
    threshold: float,
) -> None:
    x = mdates.date2num(dates)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"height_ratios": [2.4, 1.0]}, layout="constrained"
    )

    # Top: buy-and-hold equity curve, each segment coloured by the causal (filtered) regime.
    pts = np.column_stack([x, bh_cum])
    segments = np.stack([pts[:-1], pts[1:]], axis=1)
    seg_colors = [colors[int(s)] for s in state[1:]]
    ax1.add_collection(LineCollection(segments, colors=seg_colors, linewidths=2.2))
    ax1.plot(x, strat_cum, color="black", lw=1.4, ls="--")
    ax1.set_yscale("log")
    ax1.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)))
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax1.yaxis.set_minor_locator(mticker.NullLocator())
    ax1.set_xlim(x[0], x[-1])
    lo = float(min(bh_cum.min(), strat_cum.min()))
    hi = float(max(bh_cum.max(), strat_cum.max()))
    ax1.set_ylim(lo * 0.93, hi * 1.07)
    ax1.set_ylabel("Cumulative return (growth of $1, log scale)")
    ax1.set_title("SPY out-of-sample: buy-and-hold coloured by causal HMM regime vs regime-filtered strategy")
    handles = [
        Line2D([0], [0], color=colors[s], lw=3, label=f"Buy & hold - {names[s]} regime (argmax of filtered P)")
        for s in order
    ]
    handles.append(
        Line2D([0], [0], color="black", lw=1.4, ls="--", label=f"Regime-filtered (cash when P(Crash) > {threshold:g})")
    )
    ax1.legend(handles=handles, loc="upper left", frameon=True, fontsize=9)
    ax1.grid(alpha=0.25, which="both")

    # Bottom: causal crash probability.
    ax2.fill_between(x, 0.0, p_crash, color=colors[crash_state], alpha=0.35)
    ax2.plot(x, p_crash, color=colors[crash_state], lw=1.0)
    ax2.axhline(threshold, color="black", ls=":", lw=1.0)
    ax2.set_ylim(0.0, 1.02)
    ax2.set_ylabel(r"$P(z_t=\mathrm{Crash}\mid x_{1:t})$")
    ax2.set_xlabel("Date")
    ax2.set_title("Forward-filtered (zero look-ahead) crash-regime probability")
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.grid(alpha=0.25)

    fig.savefig(path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> Dict[str, object]:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_end = dt.date.fromisoformat(args.train_end)

    print(f"Downloading SPY {args.start} -> {args.end} and building features ...")
    df, X = prepare_finance_data(start=args.start, end=args.end, train_end=train_end)
    n_train = int((df["date"] <= train_end).sum())
    n_test = df.height - n_train
    if n_train < 250 or n_test < 20:
        raise RuntimeError(f"Unusable split: {n_train} training and {n_test} test days.")
    print(f"  {df.height} trading days | train: {n_train} (<= {train_end}) | test: {n_test}")

    if args.bic_sweep:
        print("BIC sweep on the training window (lower is better):")
        for k in range(2, 7):
            m = GaussianHMM(k, reg_covar=args.reg_covar, seed=args.seed).fit(X[:n_train], args.max_iters, args.tol)
            print(f"  K={k}: BIC = {compute_bic(m, X[:n_train], m.log_likelihood_):,.1f}")

    print(f"Fitting K={args.k} Gaussian HMM (full covariance, {args.n_init} restarts) on the training window ...")
    model = GaussianHMM(
        n_states=args.k, reg_covar=args.reg_covar, init="kmeans", n_init=args.n_init, seed=args.seed
    )
    model.fit(X[:n_train], max_iters=args.max_iters, tol=args.tol)
    print(
        f"  EM iterations: {model.n_iter_} | converged: {model.converged_} | "
        f"train log-likelihood: {model.log_likelihood_:,.2f} | "
        f"BIC: {compute_bic(model, X[:n_train], model.log_likelihood_):,.1f}"
    )

    order, names, colors = name_states_by_volatility(model, args.k)
    crash_state = order[-1]

    # Parameters are frozen after training. Filtering the *concatenated* series with those
    # parameters is still causal (row t only depends on x_{1:t}) and lets the out-of-sample
    # filter start from the information available at the end of the training window.
    probs = model.predict_proba(X, causal=True).numpy()
    p_crash = probs[:, crash_state]
    state_hat = probs.argmax(axis=1)

    frame = build_strategy_frame(df, p_crash, state_hat, args.threshold, args.cost_bps)
    test = frame.filter(pl.col("date") > train_end)

    bh_lr = test["log_return"].to_numpy()
    st_lr = test["strategy_log_return"].to_numpy()
    in_mkt = test["position"].to_numpy()
    stats_bh = performance_stats(bh_lr, np.ones_like(in_mkt))
    stats_st = performance_stats(st_lr, in_mkt)

    fig_path = out_dir / "finance_regimes.png"
    plot_figure(
        fig_path,
        test["date"].to_numpy(),
        np.exp(np.cumsum(bh_lr)),
        np.exp(np.cumsum(st_lr)),
        test["state"].to_numpy(),
        test["p_crash"].to_numpy(),
        names,
        colors,
        crash_state,
        order,
        args.threshold,
    )

    A_sorted = model.transmat_[order][:, order].numpy()
    result = {
        "train_end": args.train_end,
        "n_train_days": n_train,
        "n_test_days": n_test,
        "k": args.k,
        "regime_order_by_volatility": [names[s] for s in order],
        "transition_matrix_sorted_by_volatility": A_sorted.tolist(),
        "expected_regime_duration_days": [float(1.0 / (1.0 - A_sorted[i, i])) for i in range(args.k)],
        "state_means_z_sorted_by_volatility": model.means_[order].numpy().tolist(),
        "train_log_likelihood": float(model.log_likelihood_),
        "train_bic": compute_bic(model, X[:n_train], model.log_likelihood_),
        "test_fraction_days_flagged_crash": float((test["p_crash"].to_numpy() > args.threshold).mean()),
        "buy_and_hold": stats_bh,
        "regime_filtered": stats_st,
        "threshold": args.threshold,
        "cost_bps": args.cost_bps,
    }
    with open(out_dir / "finance_metrics.json", "w") as fh:
        json.dump(result, fh, indent=2)

    print("\nLearned regimes (sorted by volatility):")
    for i, s in enumerate(order):
        print(
            f"  {names[s]:<10s} mean z-vol = {float(model.means_[s, [1, 2]].mean()):+.2f} | "
            f"P(stay) = {A_sorted[i, i]:.3f} | expected duration = {1.0 / (1.0 - A_sorted[i, i]):.1f} days"
        )
    print(f"\nOut-of-sample results ({test['date'][0]} -> {test['date'][-1]}):")
    print("| Strategy | Final wealth | CAGR | Ann. vol | Sharpe | Max drawdown | Time in market |")
    print("|---|---|---|---|---|---|---|")
    for label, s in (("Buy & hold SPY", stats_bh), ("Regime-filtered", stats_st)):
        print(
            f"| {label} | {s['final_wealth']:.2f}x | {s['cagr']:.2%} | {s['annual_volatility']:.2%} | "
            f"{s['sharpe']:.2f} | {s['max_drawdown']:.2%} | {s['fraction_in_market']:.1%} |"
        )
    print(f"\nSaved figure  -> {fig_path}")
    print(f"Saved metrics -> {out_dir / 'finance_metrics.json'}")
    return result


if __name__ == "__main__":
    main()
