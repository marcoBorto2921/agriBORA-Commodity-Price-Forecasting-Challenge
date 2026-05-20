"""
EDA Time Series — checks for time series data.
Usage: import and call run_timeseries_checks(train, test, target_col, time_col)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf, pacf


OUTPUT_DIR = Path("outputs/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_THRESHOLD = 100_000


def _flag(level: str, msg: str) -> None:
    print(f"[{level}] {msg}")


def _savefig(name: str) -> None:
    path = OUTPUT_DIR / name
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.show()
    plt.close()
    _flag("INFO", f"Plot saved: {path}")


def detect_time_column(df: pd.DataFrame) -> str | None:
    """Auto-detect a datetime column by dtype or name heuristic."""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    for col in df.columns:
        if any(
            kw in col.lower()
            for kw in [
                "date",
                "time",
                "timestamp",
                "dt",
                "day",
                "week",
                "month",
                "year",
            ]
        ):
            try:
                pd.to_datetime(df[col])
                return col
            except Exception:
                pass
    return None


def check_time_index(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Parse time column, sort, check for gaps.

    Returns:
        DataFrame sorted by time_col with parsed datetime.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    diffs = df[time_col].diff().dropna()
    modal_diff = diffs.mode().iloc[0]
    gaps = diffs[diffs > modal_diff * 1.5]

    _flag("INFO", f"Time range: {df[time_col].min()} → {df[time_col].max()}")
    _flag("INFO", f"Modal time step: {modal_diff}")

    if not gaps.empty:
        _flag(
            "WARN",
            f"{len(gaps)} gaps detected (>{modal_diff * 1.5} between consecutive rows)",
        )
    else:
        _flag("INFO", "No significant time gaps detected.")

    return df


def check_target_over_time(
    df: pd.DataFrame,
    time_col: str,
    target_col: str,
) -> None:
    """Plot target value over time."""
    sample = (
        df
        if len(df) <= SAMPLE_THRESHOLD
        else df.iloc[:: len(df) // SAMPLE_THRESHOLD + 1]
    )
    plt.figure(figsize=(14, 4))
    plt.plot(sample[time_col], sample[target_col], linewidth=0.8, color="steelblue")
    plt.title(f"Target over time: {target_col}")
    plt.xlabel(time_col)
    plt.ylabel(target_col)
    plt.tight_layout()
    _savefig("target_over_time.png")


def check_stationarity(series: pd.Series, col_name: str = "target") -> None:
    """
    Augmented Dickey-Fuller test for stationarity.

    Args:
        series: Time series values.
        col_name: Label for output.
    """
    clean = series.dropna()
    adf_result = adfuller(clean, autolag="AIC")
    p_value = adf_result[1]
    if p_value < 0.05:
        _flag("INFO", f"{col_name}: stationary (ADF p={p_value:.4f})")
    else:
        _flag(
            "WARN",
            f"{col_name}: NOT stationary (ADF p={p_value:.4f}) — consider differencing or detrending",
        )


def check_acf_pacf(
    series: pd.Series,
    n_lags: int = 40,
    col_name: str = "target",
) -> None:
    """Plot ACF and PACF to identify relevant lags."""
    clean = series.dropna()
    acf_vals = acf(clean, nlags=n_lags, fft=True)
    pacf_vals = pacf(clean, nlags=n_lags)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    lags = np.arange(n_lags + 1)
    axes[0].bar(lags, acf_vals, color="steelblue")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].axhline(
        1.96 / np.sqrt(len(clean)), color="red", linestyle="--", linewidth=0.8
    )
    axes[0].axhline(
        -1.96 / np.sqrt(len(clean)), color="red", linestyle="--", linewidth=0.8
    )
    axes[0].set_title(f"ACF — {col_name}")
    axes[0].set_xlabel("Lag")

    axes[1].bar(lags, pacf_vals, color="steelblue")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].axhline(
        1.96 / np.sqrt(len(clean)), color="red", linestyle="--", linewidth=0.8
    )
    axes[1].axhline(
        -1.96 / np.sqrt(len(clean)), color="red", linestyle="--", linewidth=0.8
    )
    axes[1].set_title(f"PACF — {col_name}")
    axes[1].set_xlabel("Lag")

    plt.tight_layout()
    _savefig("acf_pacf.png")

    significant_lags = [
        i for i, v in enumerate(acf_vals[1:], 1) if abs(v) > 1.96 / np.sqrt(len(clean))
    ]
    if significant_lags:
        _flag("INFO", f"Significant ACF lags: {significant_lags[:10]}")


def check_train_test_temporal_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    time_col: str,
) -> None:
    """Verify test is always after train (no temporal leak)."""
    train_max = pd.to_datetime(train[time_col]).max()
    test_min = pd.to_datetime(test[time_col]).min()
    test_max = pd.to_datetime(test[time_col]).max()

    _flag("INFO", f"Train ends: {train_max} | Test: {test_min} → {test_max}")

    if test_min <= train_max:
        _flag(
            "CRITICAL",
            f"Temporal leak: test starts at {test_min}, before train ends at {train_max}. "
            "Features built from past data may see the future.",
        )
    else:
        gap = test_min - train_max
        _flag(
            "INFO", f"Clean temporal split. Gap between train end and test start: {gap}"
        )


# ── entry point ───────────────────────────────────────────────────────────────


def run_timeseries_checks(
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    target_col: str,
    time_col: str | None = None,
    n_lags: int = 40,
) -> None:
    """
    Run all time series EDA checks.

    Args:
        train: Training dataframe.
        test: Test dataframe (optional).
        target_col: Name of the target column.
        time_col: Name of the datetime column. Auto-detected if None.
        n_lags: Number of lags for ACF/PACF plots.
    """
    print("=" * 60)
    print("TIME SERIES CHECKS")
    print("=" * 60)

    if time_col is None:
        time_col = detect_time_column(train)
        if time_col is None:
            _flag(
                "WARN",
                "Could not auto-detect a time column. Skipping time series checks.",
            )
            return
        _flag("INFO", f"Auto-detected time column: {time_col}")

    train = check_time_index(train, time_col)
    check_target_over_time(train, time_col, target_col)
    check_stationarity(train[target_col], col_name=target_col)
    check_acf_pacf(train[target_col], n_lags=n_lags, col_name=target_col)

    if test is not None and time_col in test.columns:
        check_train_test_temporal_split(train, test, time_col)
    print()
