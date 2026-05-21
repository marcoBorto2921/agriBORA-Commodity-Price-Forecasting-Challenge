"""
E04b — County-specific blend: alpha_c × XGB_tuned + (1-alpha_c) × carryforward.

Optimise alpha per county on wk46-51 validation window (in-sample but the only
available calibration signal), then apply to wk52 + wk1 submission.

Usage:
    .venv/Scripts/python blend.py
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize_scalar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRAIN_PATH = "outputs/features/v2/train_fe.parquet"
MODEL_PATH = "outputs/models/xgb_tuned/xgboost/full_model.pkl"
XGB_SUB_PATH = "outputs/submissions/xgb_tuned/xgboost.csv"
GT_PATH = "agribora-commodity-price-forecasting-challenge20260114-30357-1miwudk/agriBORA_Final_Weeks_maize_price.csv"
OUT_PATH = "outputs/submissions/blend/blend.csv"

FEATURE_COLS = [
    "lag1_price",
    "lag2_price",
    "rolling_4w_mean",
    "deviation_from_rolling_mean",
    "rolling_4w_max",
    "price_vs_4w_max",
    "price_momentum",
    "consecutive_up_weeks",
    "week_of_year",
    "is_lean_season",
    "weeks_in_lean_season",
    "is_short_harvest_window",
    "weeks_to_long_harvest",
    "nairobi_lag1",
    "ug_lag1",
    "national_mean_lag1",
    "nairobi_ug_spread",
    "kamis_wholesale_lag1",
    "kamis_retail_lag1",
    "kamis_ug_supply_lag1",
    "kamis_national_supply_lag1",
    "is_kamis_augmented",
    "County",
]
TARGET_COL = "WholeSale"
TIME_COL = "Year_Week"

# Weeks used for alpha calibration (in-sample validation window)
CALIB_WEEKS = ["2025-46", "2025-47", "2025-48", "2025-49", "2025-50", "2025-51"]
COUNTIES = ["Kiambu", "Kirinyaga", "Mombasa", "Nairobi", "Uasin-Gishu"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def competition_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return 0.5 * mae + 0.5 * rmse


def encode_county(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df_enc = df.copy()
    unique_vals = sorted(df["County"].dropna().unique())
    enc = {v: i for i, v in enumerate(unique_vals)}
    df_enc["County"] = df["County"].map(enc).fillna(-1).astype(int)
    return df_enc, enc


def predict_xgb(model: xgb.Booster, df_enc: pd.DataFrame) -> np.ndarray:
    X = df_enc[FEATURE_COLS].values.astype(float)
    dm = xgb.DMatrix(X, feature_names=FEATURE_COLS)
    return model.predict(dm)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # ── Load data + model ──────────────────────────────────────────────────
    train = pd.read_parquet(TRAIN_PATH)

    with open(MODEL_PATH, "rb") as f:
        model: xgb.Booster = pickle.load(f)

    train_enc, _ = encode_county(train)

    # ── Calibration: predict on wk46-51 (in-sample) ───────────────────────
    calib = train[train[TIME_COL].isin(CALIB_WEEKS)].copy()
    calib_enc = train_enc[train_enc[TIME_COL].isin(CALIB_WEEKS)].copy()
    calib["xgb_pred"] = predict_xgb(model, calib_enc)

    # Build "previous week actual" carryforward for each row
    # Sort by county + week, then lag the actual price
    all_actuals = train[[TIME_COL, "County", TARGET_COL]].sort_values(
        ["County", TIME_COL]
    )
    all_actuals["prev_actual"] = all_actuals.groupby("County")[TARGET_COL].shift(1)
    calib = calib.merge(
        all_actuals[[TIME_COL, "County", "prev_actual"]],
        on=[TIME_COL, "County"],
        how="left",
    )
    # Drop rows where prev_actual is missing (first obs per county)
    calib = calib.dropna(subset=["prev_actual", TARGET_COL])

    log.info("Calibration rows: %d", len(calib))

    # ── Optimise alpha per county ──────────────────────────────────────────
    alphas: dict[str, float] = {}
    log.info("%-15s  %6s  %8s  %8s  %8s", "County", "alpha", "XGB", "Carry", "Blend")
    for county in COUNTIES:
        sub = calib[calib["County"] == county]
        if len(sub) == 0:
            alphas[county] = 1.0
            continue

        y_true = sub[TARGET_COL].values
        xgb_pred = sub["xgb_pred"].values
        carry = sub["prev_actual"].values

        xgb_score = competition_metric(y_true, xgb_pred)
        carry_score = competition_metric(y_true, carry)

        def objective(alpha: float) -> float:
            blended = alpha * xgb_pred + (1 - alpha) * carry
            return competition_metric(y_true, blended)

        result = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")
        alpha = float(result.x)
        alphas[county] = alpha

        blend_score = competition_metric(y_true, alpha * xgb_pred + (1 - alpha) * carry)
        log.info(
            "%-15s  %6.3f  %8.4f  %8.4f  %8.4f",
            county,
            alpha,
            xgb_score,
            carry_score,
            blend_score,
        )

    log.info("Optimised alphas: %s", alphas)

    # ── Apply to wk52 + wk1 ───────────────────────────────────────────────
    # Carryforward base: wk51 actual (last observed before test window)
    wk51_actual: dict[str, float] = {}
    wk51 = train[train[TIME_COL] == "2025-51"][["County", TARGET_COL]]
    for _, row in wk51.iterrows():
        wk51_actual[row["County"]] = float(row[TARGET_COL])
    log.info("Wk51 actuals: %s", wk51_actual)

    # XGB test predictions
    xgb_sub = pd.read_csv(XGB_SUB_PATH)
    # Parse county and week from ID
    xgb_sub["County"] = xgb_sub["ID"].str.rsplit("_Week_", n=1).str[0]
    xgb_sub["week"] = xgb_sub["ID"].str.rsplit("_Week_", n=1).str[1].astype(int)

    # Build blended predictions
    records = []
    wk52_blended: dict[str, float] = {}

    for _, row in xgb_sub.sort_values(["County", "week"]).iterrows():
        county = row["County"]
        week = int(row["week"])
        alpha = alphas.get(county, 1.0)
        xgb_p = float(row["Target_RMSE"])

        if week == 52:
            carry = wk51_actual.get(county, xgb_p)
        else:
            # wk1: carryforward = blended wk52 prediction
            carry = wk52_blended.get(county, wk51_actual.get(county, xgb_p))

        blended = alpha * xgb_p + (1 - alpha) * carry

        if week == 52:
            wk52_blended[county] = blended

        records.append(
            {
                "ID": row["ID"],
                "County": county,
                "week": week,
                "xgb_pred": xgb_p,
                "carry": carry,
                "alpha": alpha,
                "blend": blended,
            }
        )

    result_df = pd.DataFrame(records)
    log.info(
        "\nBlend vs XGB predictions:\n%s",
        result_df[["ID", "xgb_pred", "carry", "alpha", "blend"]].to_string(index=False),
    )

    # ── Compute local score ────────────────────────────────────────────────
    gt = pd.read_csv(GT_PATH)
    merged = result_df.merge(gt[["ID", "WholeSale"]], on="ID")
    y_true = merged["WholeSale"].values
    y_xgb = merged["xgb_pred"].values
    y_blend = merged["blend"].values

    score_xgb = competition_metric(y_true, y_xgb)
    score_blend = competition_metric(y_true, y_blend)
    log.info(
        "Local score — XGB alone: %.4f | Blend: %.4f | Delta: %.4f",
        score_xgb,
        score_blend,
        score_xgb - score_blend,
    )

    # Per-county breakdown
    log.info("\nPer-county errors:")
    for county in COUNTIES:
        sub = merged[merged["County"] == county]
        err_xgb = sub["WholeSale"].values - sub["xgb_pred"].values
        err_blend = sub["WholeSale"].values - sub["blend"].values
        log.info(
            "  %-15s  XGB errors: %s  |  Blend errors: %s",
            county,
            [f"{e:+.2f}" for e in err_xgb],
            [f"{e:+.2f}" for e in err_blend],
        )

    # ── Write submission ───────────────────────────────────────────────────
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    sub_out = result_df[["ID"]].copy()
    sub_out["Target_RMSE"] = result_df["blend"].round(6)
    sub_out["Target_MAE"] = result_df["blend"].round(6)
    sub_out.to_csv(OUT_PATH, index=False)
    log.info("Submission saved: %s", OUT_PATH)


if __name__ == "__main__":
    main()
