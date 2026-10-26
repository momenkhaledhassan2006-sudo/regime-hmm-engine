"""Offline tests for the Polars feature pipelines (no network access needed)."""

import datetime as dt
import math

import numpy as np
import polars as pl
import pytest
import torch

from hmm_engine import (
    GaussianHMM,
    align_states_hungarian,
    calculate_classification_metrics,
)
from hmm_engine import finance_pipeline as fin
from hmm_engine import neuro_pipeline as neuro


# --------------------------------------------------------------------------- #
# Finance
# --------------------------------------------------------------------------- #
def _synthetic_ohlc(n: int = 400, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, size=n)))
    high = close * np.exp(np.abs(rng.normal(0.0, 0.005, size=n)))
    low = close * np.exp(-np.abs(rng.normal(0.0, 0.005, size=n)))
    dates = np.datetime64("2010-01-04") + np.arange(n).astype("timedelta64[D]")
    return pl.DataFrame(
        {
            "date": pl.Series("date", dates, dtype=pl.Date),
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 1e6),
        }
    )


def test_finance_features_match_definitions():
    ohlc = _synthetic_ohlc()
    feats = fin.compute_features(ohlc, rv_window=10)

    close = ohlc["close"].to_numpy()
    high = ohlc["high"].to_numpy()
    low = ohlc["low"].to_numpy()
    r = np.diff(np.log(close))  # r[i] is the return of day i+1

    # Warm-up: first return needs 1 row, first 10-day realised vol needs 10 returns.
    assert feats.height == ohlc.height - 10
    row = 25  # index inside the feature frame
    day = row + 10  # corresponding day index in the original frame

    assert feats["log_return"][row] == pytest.approx(r[day - 1])
    assert feats["parkinson_vol"][row] == pytest.approx(
        math.sqrt(math.log(high[day] / low[day]) ** 2 / (4.0 * math.log(2.0)))
    )
    assert feats["realized_vol_10d"][row] == pytest.approx(math.sqrt(np.sum(r[day - 10 : day] ** 2)))
    assert feats.select(fin.FEATURE_COLUMNS).null_count().to_numpy().sum() == 0


def test_finance_zscore_uses_training_window_only():
    ohlc = _synthetic_ohlc(n=600)
    train_end = dt.date(2011, 3, 31)
    df, X = fin.prepare_finance_data(ohlc=ohlc, train_end=train_end)

    assert X.dtype == torch.float64
    assert tuple(X.shape) == (df.height, 3)
    assert torch.isfinite(X).all()

    train = df.filter(pl.col("date") <= train_end)
    assert 50 < train.height < df.height
    for col in fin.Z_COLUMNS:
        assert abs(float(train[col].mean())) < 1e-9
        assert float(train[col].std()) == pytest.approx(1.0, abs=1e-9)
    # The out-of-sample window is *not* re-normalised (no leakage of its own statistics).
    test = df.filter(pl.col("date") > train_end)
    assert abs(float(test["z_log_return"].mean())) > 1e-6 or abs(float(test["z_parkinson_vol"].mean())) > 1e-6

    _, X_full = fin.prepare_finance_data(ohlc=ohlc, train_end=None)
    assert torch.allclose(X_full.mean(dim=0), torch.zeros(3, dtype=torch.float64), atol=1e-9)


def test_finance_pipeline_feeds_the_hmm():
    ohlc = _synthetic_ohlc(n=500, seed=3)
    _, X = fin.prepare_finance_data(ohlc=ohlc, train_end="2011-06-30")
    model = GaussianHMM(n_states=3, reg_covar=1e-4, seed=0).fit(X[:300], max_iters=30, tol=1e-6)
    p = model.predict_proba(X, causal=True)
    assert p.shape == (X.shape[0], 3)
    assert torch.isfinite(p).all()


# --------------------------------------------------------------------------- #
# Neuro
# --------------------------------------------------------------------------- #
def test_bandpower_features_identify_dominant_band():
    sfreq = 100.0
    t = np.arange(int(30 * sfreq)) / sfreq
    rng = np.random.default_rng(0)
    freq_to_band = {2.0: "delta", 6.0: "theta", 10.0: "alpha", 14.0: "sigma", 22.0: "beta"}
    epochs = np.stack(
        [np.sin(2 * np.pi * f * t) + 0.05 * rng.standard_normal(t.size) for f in freq_to_band]
    )
    feats = neuro.compute_bandpower_features(epochs, sfreq)
    assert feats.shape == (5, 5)
    # Relative powers of an epoch form a distribution.
    assert np.allclose(np.exp(feats).sum(axis=1), 1.0, atol=1e-6)
    for row, band in enumerate(freq_to_band.values()):
        assert neuro.BAND_NAMES[int(feats[row].argmax())] == band


def _synthetic_recording(sfreq: float = 100.0, wake_epochs: int = 40, seed: int = 0):
    """Raw with two channels and a hypnogram whose spectra depend on the stage."""
    import mne

    rng = np.random.default_rng(seed)
    # (AASM stage annotation, number of 30 s epochs, dominant frequency in Hz)
    plan = [
        ("Sleep stage W", wake_epochs, 10.0),
        ("Sleep stage 1", 6, 6.0),
        ("Sleep stage 2", 30, 14.0),
        ("Sleep stage 3", 10, 2.0),
        ("Sleep stage 4", 8, 1.5),  # merged into N3
        ("Sleep stage 2", 20, 14.0),
        ("Sleep stage R", 20, 22.0),
        ("Sleep stage W", wake_epochs, 10.0),
    ]
    samples_per_epoch = int(30 * sfreq)
    t = np.arange(samples_per_epoch) / sfreq
    chunks, onset, onsets, durations, descriptions = [], 0.0, [], [], []
    for desc, n_ep, freq in plan:
        for _ in range(n_ep):
            f = freq + rng.uniform(-0.4, 0.4)
            chunks.append(
                rng.uniform(0.8, 1.2) * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))
                + 0.15 * rng.standard_normal(samples_per_epoch)
            )
        onsets.append(onset)
        durations.append(30.0 * n_ep)
        descriptions.append(desc)
        onset += 30.0 * n_ep

    eeg = np.concatenate(chunks) * 1e-5
    eog = rng.standard_normal(eeg.size) * 1e-5
    info = mne.create_info(["EEG Fpz-Cz", "EOG horizontal"], sfreq, ch_types=["eeg", "eog"])
    raw = mne.io.RawArray(np.vstack([eeg, eog]), info, verbose="ERROR")
    raw.set_annotations(mne.Annotations(onsets, durations, descriptions))

    expected = []
    stage_id = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
                "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4}
    for desc, n_ep, _ in plan:
        expected += [stage_id[desc]] * n_ep
    return raw, np.array(expected), sfreq


def test_epoching_labels_merging_and_wake_cropping():
    pytest.importorskip("mne")
    raw, expected, sfreq = _synthetic_recording()

    epochs, labels, fs = neuro.epochs_from_raw(raw, wake_margin_epochs=None)
    assert fs == sfreq
    assert epochs.shape == (expected.size, int(30 * sfreq))
    assert np.array_equal(labels, expected)  # stage 4 merged into N3, order preserved

    margin = 10
    epochs_c, labels_c, _ = neuro.epochs_from_raw(raw, wake_margin_epochs=margin)
    first, last = np.flatnonzero(expected != 0)[[0, -1]]
    assert np.array_equal(labels_c, expected[first - margin : last + margin + 1])
    assert epochs_c.shape[0] == labels_c.size


def test_sleep_pipeline_end_to_end_on_synthetic_recording():
    pytest.importorskip("mne")
    raw, _, sfreq = _synthetic_recording(seed=1)
    epochs, labels, fs = neuro.epochs_from_raw(raw, wake_margin_epochs=20)
    df, X = neuro.build_sleep_dataframe(epochs, labels, fs)

    assert X.dtype == torch.float64
    assert tuple(X.shape) == (labels.size, 5)
    assert df.columns[:4] == ["epoch", "time_h", "stage", "stage_name"]
    for col in neuro.Z_COLUMNS:
        assert abs(float(df[col].mean())) < 1e-9
        assert float(df[col].std()) == pytest.approx(1.0, abs=1e-9)
    assert set(df["stage_name"].unique().to_list()) == {"Wake", "N1", "N2", "N3", "REM"}

    model = GaussianHMM(n_states=5, reg_covar=1e-3, init="kmeans", n_init=3, seed=0).fit(X, max_iters=100)
    pred = model.decode_viterbi(X).numpy()
    aligned, mapping = align_states_hungarian(df["stage"].to_numpy(), pred)
    scores = calculate_classification_metrics(df["stage"].to_numpy(), aligned)
    assert len(mapping) == 5
    assert scores["accuracy"] > 0.85
    assert scores["cohen_kappa"] > 0.8
