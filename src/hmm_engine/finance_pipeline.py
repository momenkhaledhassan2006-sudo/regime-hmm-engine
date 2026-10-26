"""SPY daily feature pipeline (Polars) for market regime detection.

Features
--------
* ``log_return``       : ``ln(C_t / C_{t-1})``
* ``parkinson_vol``    : Parkinson (1980) range estimator,
                         ``sqrt( ln(H_t / L_t)^2 / (4 ln 2) )`` (daily, not annualised)
* ``realized_vol_10d`` : ``sqrt( sum_{i=0}^{9} r_{t-i}^2 )``

Z-scoring uses statistics from the *training window only* by default
(``train_end``), so no information from the out-of-sample period leaks into the
inputs of the model. Pass ``train_end=None`` for full-sample normalisation.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import polars as pl
import torch

TICKER = "SPY"
FEATURE_COLUMNS: List[str] = ["log_return", "parkinson_vol", "realized_vol_10d"]
Z_COLUMNS: List[str] = [f"z_{c}" for c in FEATURE_COLUMNS]

DateLike = Union[str, dt.date]


def _to_date(d: DateLike) -> dt.date:
    if isinstance(d, dt.datetime):
        return d.date()
    if isinstance(d, dt.date):
        return d
    return dt.date.fromisoformat(str(d))


def download_ohlc(
    ticker: str = TICKER,
    start: str = "2005-01-01",
    end: str = "2026-12-31",
) -> pl.DataFrame:
    """Download split/dividend-adjusted daily OHLCV with yfinance as a Polars frame.

    Columns: ``date`` (pl.Date), ``open``, ``high``, ``low``, ``close``, ``volume``.
    ``end`` is exclusive (yfinance convention) and may lie in the future.
    """
    import yfinance as yf  # imported lazily so offline code paths never need it

    hist = yf.Ticker(ticker).history(
        start=start, end=end, interval="1d", auto_adjust=True, actions=False
    )
    if hist is None or hist.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker} ({start} -> {end}).")

    index = hist.index
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)  # keep the exchange-local calendar date
    dates = np.asarray(index.values).astype("datetime64[D]")

    return pl.DataFrame(
        {
            "date": pl.Series("date", dates, dtype=pl.Date),
            "open": hist["Open"].to_numpy(dtype=np.float64),
            "high": hist["High"].to_numpy(dtype=np.float64),
            "low": hist["Low"].to_numpy(dtype=np.float64),
            "close": hist["Close"].to_numpy(dtype=np.float64),
            "volume": hist["Volume"].to_numpy(dtype=np.float64),
        }
    ).sort("date")


def compute_features(ohlc: pl.DataFrame, rv_window: int = 10) -> pl.DataFrame:
    """Add the three raw features to an OHLC frame and drop warm-up rows."""
    required = {"date", "high", "low", "close"}
    missing = required - set(ohlc.columns)
    if missing:
        raise ValueError(f"OHLC frame is missing columns: {sorted(missing)}")

    out = (
        ohlc.sort("date")
        .filter((pl.col("low") > 0) & (pl.col("high") >= pl.col("low")) & (pl.col("close") > 0))
        .with_columns(
            pl.col("close").log().diff().alias("log_return"),
            ((pl.col("high") / pl.col("low")).log().pow(2) / (4.0 * math.log(2.0)))
            .sqrt()
            .alias("parkinson_vol"),
        )
        .with_columns(
            pl.col("log_return").pow(2).rolling_sum(window_size=rv_window).sqrt().alias("realized_vol_10d")
        )
        .drop_nulls(subset=FEATURE_COLUMNS)
    )
    return out


def zscore_features(df: pl.DataFrame, train_end: Optional[DateLike] = None) -> pl.DataFrame:
    """Append ``z_<feature>`` columns.

    Mean and standard deviation (ddof = 1) are estimated on rows with
    ``date <= train_end`` (or on all rows if ``train_end`` is ``None``) and then
    applied to every row.
    """
    ref = df if train_end is None else df.filter(pl.col("date") <= _to_date(train_end))
    if ref.height < 2:
        raise ValueError("Not enough rows in the normalisation window.")

    exprs = []
    for col in FEATURE_COLUMNS:
        mu = float(ref[col].mean())
        sd = float(ref[col].std())
        if not np.isfinite(sd) or sd <= 0.0:
            raise ValueError(f"Feature '{col}' has zero variance in the normalisation window.")
        exprs.append(((pl.col(col) - mu) / sd).alias(f"z_{col}"))
    return df.with_columns(exprs)


def prepare_finance_data(
    start: str = "2005-01-01",
    end: str = "2026-12-31",
    train_end: Optional[DateLike] = "2018-12-31",
    rv_window: int = 10,
    ohlc: Optional[pl.DataFrame] = None,
    ticker: str = TICKER,
) -> Tuple[pl.DataFrame, torch.Tensor]:
    """Full pipeline: download -> features -> z-score.

    Parameters
    ----------
    ohlc:
        Optionally supply an OHLC frame (``date, high, low, close``) instead of
        downloading, e.g. for offline tests.

    Returns
    -------
    df : pl.DataFrame
        ``date, [open, high, low, close, volume,] log_return, parkinson_vol,
        realized_vol_10d, z_log_return, z_parkinson_vol, z_realized_vol_10d``.
    X : torch.Tensor
        ``float64`` tensor of shape ``(T, 3)`` holding the z-scored features
        (columns in the order of :data:`Z_COLUMNS`).
    """
    raw = download_ohlc(ticker=ticker, start=start, end=end) if ohlc is None else ohlc
    df = zscore_features(compute_features(raw, rv_window=rv_window), train_end=train_end)
    X = torch.as_tensor(
        np.ascontiguousarray(df.select(Z_COLUMNS).to_numpy(), dtype=np.float64),
        dtype=torch.float64,
    )
    return df, X
