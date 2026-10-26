"""PhysioNet Sleep-EDF (sleep-cassette) pipeline: MNE -> Welch PSD -> Polars -> torch.

Steps
-----
1. ``fetch_sleep_edf``       download one PSG / hypnogram pair via MNE.
2. ``epochs_from_raw``       cut the ``EEG Fpz-Cz`` channel into 30-second epochs
                             labelled with AASM stages (Wake=0, N1=1, N2=2,
                             N3=3 [stage 3 and 4 merged], REM=4).
3. ``compute_bandpower_features``  Welch PSD -> log relative band power in
                             delta / theta / alpha / sigma / beta.
4. ``build_sleep_dataframe`` Polars frame with raw + z-scored features.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
from scipy.signal import welch

EEG_CHANNEL = "EEG Fpz-Cz"
EPOCH_SECONDS = 30.0

STAGE_NAMES: Dict[int, str] = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

# MNE annotation description -> internal event id (1-based; id 0 is avoided on purpose).
_ANNOTATION_TO_EVENT_ID: Dict[str, int] = {
    "Sleep stage W": 1,
    "Sleep stage 1": 2,
    "Sleep stage 2": 3,
    "Sleep stage 3": 4,
    "Sleep stage 4": 4,  # AASM: stages 3 and 4 are merged into N3
    "Sleep stage R": 5,
}

BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),  # sleep spindles
    "beta": (16.0, 30.0),
}
BAND_NAMES: List[str] = list(BANDS.keys())
FEATURE_COLUMNS: List[str] = [f"log_rel_{b}" for b in BAND_NAMES]
Z_COLUMNS: List[str] = [f"z_{c}" for c in FEATURE_COLUMNS]


def fetch_sleep_edf(
    subject: int = 0,
    recording: int = 1,
    data_path: Optional[str] = None,
) -> Tuple[str, str]:
    """Download (or locate) a Sleep-EDF PSG file and its clinician hypnogram."""
    from mne.datasets.sleep_physionet.age import fetch_data

    files = fetch_data(subjects=[subject], recording=[recording], path=data_path)
    psg_path, hypnogram_path = files[0]
    return str(psg_path), str(hypnogram_path)


def epochs_from_raw(
    raw,
    channel: str = EEG_CHANNEL,
    wake_margin_epochs: Optional[int] = 60,
    epoch_seconds: float = EPOCH_SECONDS,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Cut an annotated ``mne.io.Raw`` into labelled fixed-length epochs.

    ``raw`` must already carry the hypnogram annotations. Epochs whose
    annotation is not a scored sleep stage (``Sleep stage ?``, ``Movement
    time``) are dropped. Long Wake stretches before the first and after the last
    sleep epoch are trimmed to ``wake_margin_epochs`` epochs (30 min by default);
    pass ``None`` to keep the whole recording.

    Returns ``(epochs, labels, sfreq)`` with ``epochs`` of shape
    ``(n_epochs, samples_per_epoch)`` in volts and ``labels`` in ``{0..4}``.
    """
    import mne

    raw = raw.copy().pick([channel])
    raw.load_data(verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    samples = int(round(epoch_seconds * sfreq))

    events, _ = mne.events_from_annotations(
        raw,
        event_id=_ANNOTATION_TO_EVENT_ID,
        chunk_duration=epoch_seconds,
        verbose="ERROR",
    )
    if len(events) == 0:
        raise RuntimeError("No scored sleep-stage annotations were found.")

    data = raw.get_data()[0].astype(np.float64)
    starts = events[:, 0].astype(np.int64) - int(raw.first_samp)
    labels = (events[:, 2].astype(np.int64) - 1)

    order = np.argsort(starts, kind="stable")
    starts, labels = starts[order], labels[order]
    keep = (starts >= 0) & (starts + samples <= data.shape[0])
    starts, labels = starts[keep], labels[keep]

    if wake_margin_epochs is not None:
        asleep = np.flatnonzero(labels != 0)
        if asleep.size > 0:
            lo = max(int(asleep[0]) - int(wake_margin_epochs), 0)
            hi = min(int(asleep[-1]) + int(wake_margin_epochs) + 1, labels.shape[0])
            starts, labels = starts[lo:hi], labels[lo:hi]

    idx = starts[:, None] + np.arange(samples, dtype=np.int64)[None, :]
    return data[idx], labels, sfreq


def load_epoched_eeg(
    psg_path: str,
    hypnogram_path: str,
    channel: str = EEG_CHANNEL,
    wake_margin_epochs: Optional[int] = 60,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Read a PSG EDF + hypnogram EDF with MNE and return labelled epochs."""
    import mne

    raw = mne.io.read_raw_edf(psg_path, preload=False, verbose="ERROR")
    annotations = mne.read_annotations(hypnogram_path)
    raw.set_annotations(annotations, emit_warning=False)
    return epochs_from_raw(raw, channel=channel, wake_margin_epochs=wake_margin_epochs)


def compute_bandpower_features(
    epochs: np.ndarray,
    sfreq: float,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    window_seconds: float = 4.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Welch PSD -> natural-log relative band power, shape ``(n_epochs, n_bands)``.

    Each band power is the sum of PSD bins in ``[f_lo, f_hi)`` times the bin
    width; it is divided by the total power over all bands (so the relative
    powers of an epoch sum to one) before taking the logarithm.
    """
    bands = BANDS if bands is None else bands
    if sfreq / 2.0 <= max(hi for _, hi in bands.values()):
        raise ValueError("Sampling rate too low for the requested frequency bands.")

    nperseg = min(int(round(window_seconds * sfreq)), epochs.shape[1])
    freqs, psd = welch(
        epochs,
        fs=sfreq,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
        axis=-1,
    )
    df = float(freqs[1] - freqs[0])
    powers = np.stack(
        [psd[:, (freqs >= lo) & (freqs < hi)].sum(axis=1) * df for lo, hi in bands.values()],
        axis=1,
    )
    total = np.maximum(powers.sum(axis=1, keepdims=True), np.finfo(np.float64).tiny)
    rel = powers / total
    return np.log(np.clip(rel, eps, None))


def build_sleep_dataframe(
    epochs: np.ndarray,
    labels: np.ndarray,
    sfreq: float,
) -> Tuple[pl.DataFrame, torch.Tensor]:
    """Feature extraction + z-scoring (Polars). Returns ``(df, X)``."""
    log_rel = compute_bandpower_features(epochs, sfreq)
    n = log_rel.shape[0]

    data = {
        "epoch": np.arange(n, dtype=np.int64),
        "time_h": np.arange(n, dtype=np.float64) * EPOCH_SECONDS / 3600.0,
        "stage": labels.astype(np.int64),
        "stage_name": [STAGE_NAMES[int(s)] for s in labels],
    }
    for j, col in enumerate(FEATURE_COLUMNS):
        data[col] = log_rel[:, j]
    df = pl.DataFrame(data)

    z_exprs = []
    for col in FEATURE_COLUMNS:
        mu = float(df[col].mean())
        sd = float(df[col].std())
        if not np.isfinite(sd) or sd <= 0.0:
            raise ValueError(f"Feature '{col}' has zero variance.")
        z_exprs.append(((pl.col(col) - mu) / sd).alias(f"z_{col}"))
    df = df.with_columns(z_exprs)

    X = torch.as_tensor(
        np.ascontiguousarray(df.select(Z_COLUMNS).to_numpy(), dtype=np.float64),
        dtype=torch.float64,
    )
    return df, X


def prepare_sleep_data(
    subject: int = 0,
    recording: int = 1,
    data_path: Optional[str] = None,
    channel: str = EEG_CHANNEL,
    wake_margin_epochs: Optional[int] = 60,
) -> Tuple[pl.DataFrame, torch.Tensor]:
    """Full pipeline: fetch -> epoch -> Welch band power -> z-score.

    Returns
    -------
    df : pl.DataFrame
        ``epoch, time_h, stage, stage_name, log_rel_<band>..., z_log_rel_<band>...``
    X : torch.Tensor
        ``float64`` tensor ``(n_epochs, 5)`` of z-scored log relative band powers.
    """
    psg_path, hypnogram_path = fetch_sleep_edf(subject, recording, data_path)
    epochs, labels, sfreq = load_epoched_eeg(
        psg_path, hypnogram_path, channel=channel, wake_margin_epochs=wake_margin_epochs
    )
    return build_sleep_dataframe(epochs, labels, sfreq)
