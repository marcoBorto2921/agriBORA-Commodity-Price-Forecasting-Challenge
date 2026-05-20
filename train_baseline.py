"""
Baseline model training — agriBORA Commodity Price Forecasting.

Trains LightGBM, XGBoost, and a trend extrapolation baseline using
walk-forward time series CV. Reads all config from configs/baseline.yaml.

Usage:
    .venv/Scripts/python train_baseline.py --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import time
import warnings
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def competition_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """0.5 * MAE + 0.5 * RMSE — lower is better."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return 0.5 * mae + 0.5 * rmse


# ---------------------------------------------------------------------------
# Walk-forward CV splits
# ---------------------------------------------------------------------------

def get_walk_forward_splits(
    df: pd.DataFrame,
    time_col: str,
    n_folds: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Walk-forward CV: sorted unique weeks, last 30% as validation window.
    Each fold: train = all weeks up to cutoff, val = next 2 weeks after cutoff.

    Returns list of (train_idx, val_idx) index arrays.
    """
    sorted_weeks = sorted(df[time_col].unique())
    n_weeks = len(sorted_weeks)

    # Use last 30% of timeline as the rolling validation region
    val_start = int(n_weeks * 0.70)
    val_weeks = sorted_weeks[val_start:]

    # Create n_folds cutpoints within the validation region
    cutpoints = np.linspace(val_start - 1, n_weeks - 3, n_folds, dtype=int)
    cutpoints = np.unique(cutpoints)  # deduplicate if linspace produces repeats

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for cutpoint in cutpoints:
        cutoff_week = sorted_weeks[cutpoint]
        next_two = sorted_weeks[cutpoint + 1 : cutpoint + 3]
        if not next_two:
            continue

        train_mask = df[time_col] <= cutoff_week
        val_mask = df[time_col].isin(next_two)

        train_idx = df.index[train_mask].to_numpy()
        val_idx = df.index[val_mask].to_numpy()

        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        splits.append((train_idx, val_idx))

    return splits


# ---------------------------------------------------------------------------
# LightGBM training
# ---------------------------------------------------------------------------

def train_lightgbm(
    df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    target_col: str,
    time_col: str,
    cfg: dict[str, Any],
    model_dir: Path,
    oof_dir: Path,
) -> dict[str, Any]:
    """Train LightGBM with walk-forward CV. Returns results dict."""
    lgb_cfg = cfg["models"]["lightgbm"]
    params = lgb_cfg["params"].copy()
    early_stopping = lgb_cfg.get("early_stopping_rounds", 50)
    n_folds = cfg["n_folds"]

    splits = get_walk_forward_splits(df, time_col, n_folds)
    logger.info("LightGBM: %d CV splits", len(splits))

    oof_records: list[dict] = []
    fold_scores: list[float] = []
    models: list[lgb.Booster] = []

    # LightGBM requires integer-encoded categoricals (not string dtype)
    cat_features = [c for c in categorical_cols if c in feature_cols]
    cat_encoders: dict[str, dict] = {}
    df_enc = df.copy()
    for col in cat_features:
        unique_vals = sorted(df[col].dropna().unique())
        cat_encoders[col] = {v: i for i, v in enumerate(unique_vals)}
        df_enc[col] = df[col].map(cat_encoders[col]).fillna(-1).astype("int32")

    n_estimators = params.pop("n_estimators", 500)

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_df = df_enc.loc[train_idx]
        val_df = df_enc.loc[val_idx]
        val_df_orig = df.loc[val_idx]

        X_train = train_df[feature_cols]
        y_train = train_df[target_col].values
        X_val = val_df[feature_cols]
        y_val = val_df[target_col].values

        dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
        dval = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_features, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(early_stopping, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        val_pred = model.predict(X_val)
        score = competition_metric(y_val, val_pred)
        fold_scores.append(score)
        logger.info("  Fold %d: score=%.4f (n_train=%d, n_val=%d)", fold_idx + 1, score, len(train_idx), len(val_idx))

        for i, idx in enumerate(val_idx):
            row = val_df_orig.loc[idx]
            oof_records.append({
                "index": idx,
                "County": row.get("County", ""),
                "Year_Week": row.get(time_col, ""),
                "y_true": float(y_val[i]),
                "y_pred": float(val_pred[i]),
                "fold": fold_idx + 1,
            })

        model_path = model_dir / "lightgbm" / f"fold_{fold_idx + 1}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        models.append(model)

    cv_mean = float(np.mean(fold_scores))
    cv_std = float(np.std(fold_scores))
    logger.info("LightGBM CV: %.4f +/- %.4f  | folds: %s", cv_mean, cv_std, fold_scores)

    oof_df = pd.DataFrame(oof_records)
    oof_path = oof_dir / "lightgbm_oof.parquet"
    oof_df.to_parquet(oof_path, index=False)

    # Train-set in-sample score (leakage check)
    full_model = lgb.train(
        params,
        lgb.Dataset(df_enc[feature_cols], label=df_enc[target_col].values, categorical_feature=cat_features),
        num_boost_round=int(np.mean([m.num_trees() for m in models])),
        callbacks=[lgb.log_evaluation(period=-1)],
    )
    train_pred = full_model.predict(df_enc[feature_cols])
    train_score = competition_metric(df_enc[target_col].values, train_pred)
    logger.info("LightGBM train (in-sample): %.4f", train_score)

    # Save full model for test inference
    full_model_path = model_dir / "lightgbm" / "full_model.pkl"
    with open(full_model_path, "wb") as f:
        pickle.dump(full_model, f)

    return {
        "model": "lightgbm",
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "fold_scores": fold_scores,
        "train_score": train_score,
        "oof_path": str(oof_path),
        "full_model": full_model,
        "cat_encoders": cat_encoders,
        "feature_importances": dict(zip(feature_cols, full_model.feature_importance(importance_type="gain").tolist())),
    }


# ---------------------------------------------------------------------------
# XGBoost training
# ---------------------------------------------------------------------------

def train_xgboost(
    df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    target_col: str,
    time_col: str,
    cfg: dict[str, Any],
    model_dir: Path,
    oof_dir: Path,
) -> dict[str, Any]:
    """Train XGBoost with walk-forward CV. Returns results dict."""
    xgb_cfg = cfg["models"]["xgboost"]
    params = xgb_cfg["params"].copy()
    early_stopping = xgb_cfg.get("early_stopping_rounds", 50)
    n_folds = cfg["n_folds"]

    splits = get_walk_forward_splits(df, time_col, n_folds)
    logger.info("XGBoost: %d CV splits", len(splits))

    # XGBoost needs numeric encoding for categoricals
    cat_encoders: dict[str, dict] = {}
    df_enc = df.copy()
    for col in categorical_cols:
        if col in feature_cols:
            unique_vals = sorted(df[col].dropna().unique())
            cat_encoders[col] = {v: i for i, v in enumerate(unique_vals)}
            df_enc[col] = df[col].map(cat_encoders[col]).fillna(-1).astype(int)

    oof_records: list[dict] = []
    fold_scores: list[float] = []
    models: list[xgb.Booster] = []

    n_estimators = params.pop("n_estimators", 500)
    eval_metric = params.pop("eval_metric", "mae")

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_df = df_enc.loc[train_idx]
        val_df_enc = df_enc.loc[val_idx]
        val_df_orig = df.loc[val_idx]

        X_train = train_df[feature_cols].values.astype(float)
        y_train = train_df[target_col].values
        X_val = val_df_enc[feature_cols].values.astype(float)
        y_val = val_df_orig[target_col].values

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)

        evals_result: dict = {}
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dval, "val")],
            early_stopping_rounds=early_stopping,
            evals_result=evals_result,
            verbose_eval=False,
        )

        val_pred = model.predict(dval)
        score = competition_metric(y_val, val_pred)
        fold_scores.append(score)
        logger.info("  Fold %d: score=%.4f (n_train=%d, n_val=%d)", fold_idx + 1, score, len(train_idx), len(val_idx))

        for i, idx in enumerate(val_idx):
            row = val_df_orig.loc[idx]
            oof_records.append({
                "index": idx,
                "County": row.get("County", ""),
                "Year_Week": row.get(time_col, ""),
                "y_true": float(y_val[i]),
                "y_pred": float(val_pred[i]),
                "fold": fold_idx + 1,
            })

        model_path = model_dir / "xgboost" / f"fold_{fold_idx + 1}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        models.append(model)

    cv_mean = float(np.mean(fold_scores))
    cv_std = float(np.std(fold_scores))
    logger.info("XGBoost CV: %.4f +/- %.4f  | folds: %s", cv_mean, cv_std, fold_scores)

    oof_df = pd.DataFrame(oof_records)
    oof_path = oof_dir / "xgboost_oof.parquet"
    oof_df.to_parquet(oof_path, index=False)

    # Full model for test inference
    dtrain_full = xgb.DMatrix(
        df_enc[feature_cols].values.astype(float),
        label=df[target_col].values,
        feature_names=feature_cols,
    )
    avg_best_round = int(np.mean([m.best_iteration for m in models]))
    full_model = xgb.train(params, dtrain_full, num_boost_round=avg_best_round + 1, verbose_eval=False)

    train_pred = full_model.predict(dtrain_full)
    train_score = competition_metric(df[target_col].values, train_pred)
    logger.info("XGBoost train (in-sample): %.4f", train_score)

    full_model_path = model_dir / "xgboost" / "full_model.pkl"
    with open(full_model_path, "wb") as f:
        pickle.dump(full_model, f)

    return {
        "model": "xgboost",
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "fold_scores": fold_scores,
        "train_score": train_score,
        "oof_path": str(oof_path),
        "full_model": full_model,
        "cat_encoders": cat_encoders,
    }


# ---------------------------------------------------------------------------
# Trend extrapolation baseline
# ---------------------------------------------------------------------------

def train_trend_extrapolation(
    df: pd.DataFrame,
    target_col: str,
    time_col: str,
    cfg: dict[str, Any],
    oof_dir: Path,
) -> dict[str, Any]:
    """
    Trend extrapolation baseline per county.

    For each validation fold:
    - Compute avg weekly price change over the last 6 weeks of training data
    - Predict val[0] = last_obs + slope, val[1] = val[0] + slope

    This directly addresses the fold 5 uptrend underprediction issue.
    """
    n_folds = cfg["n_folds"]
    splits = get_walk_forward_splits(df, time_col, n_folds)
    logger.info("Trend extrapolation: %d CV splits", len(splits))

    oof_records: list[dict] = []
    fold_scores: list[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_df = df.loc[train_idx]
        val_df = df.loc[val_idx]

        val_preds: list[float] = []
        y_val: list[float] = []

        for _, val_row in val_df.iterrows():
            county = val_row["County"]
            week = val_row[time_col]
            y_val.append(float(val_row[target_col]))

            county_train = train_df[train_df["County"] == county].sort_values(time_col)
            if len(county_train) < 2:
                # Fallback: predict last known value
                pred = float(county_train[target_col].iloc[-1]) if len(county_train) >= 1 else float(train_df[target_col].mean())
                val_preds.append(pred)
                continue

            # Slope = avg weekly change over last 6 weeks
            recent = county_train.tail(6)
            diffs = recent[target_col].diff().dropna()
            slope = float(diffs.mean()) if len(diffs) > 0 else 0.0
            last_obs = float(county_train[target_col].iloc[-1])

            # Determine horizon (1=first step, 2=second step) per county
            county_val_sorted = val_df[val_df["County"] == county].sort_values(time_col)
            county_val_weeks = county_val_sorted[time_col].tolist()
            horizon = county_val_weeks.index(week) + 1 if week in county_val_weeks else 1
            pred = last_obs + slope * horizon
            val_preds.append(pred)

        y_val_arr = np.array(y_val)
        val_pred_arr = np.array(val_preds)
        score = competition_metric(y_val_arr, val_pred_arr)
        fold_scores.append(score)
        logger.info("  Fold %d: score=%.4f", fold_idx + 1, score)

        for i, idx in enumerate(val_idx):
            row = val_df.loc[idx]
            oof_records.append({
                "index": idx,
                "County": row.get("County", ""),
                "Year_Week": row.get(time_col, ""),
                "y_true": float(y_val[i]),
                "y_pred": float(val_preds[i]),
                "fold": fold_idx + 1,
            })

    cv_mean = float(np.mean(fold_scores))
    cv_std = float(np.std(fold_scores))
    logger.info("Trend extrapolation CV: %.4f +/- %.4f  | folds: %s", cv_mean, cv_std, fold_scores)

    oof_df = pd.DataFrame(oof_records)
    oof_path = oof_dir / "trend_extrapolation_oof.parquet"
    oof_df.to_parquet(oof_path, index=False)

    return {
        "model": "trend_extrapolation",
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "fold_scores": fold_scores,
        "train_score": None,
        "oof_path": str(oof_path),
    }


# ---------------------------------------------------------------------------
# Test inference
# ---------------------------------------------------------------------------

def _encode_categoricals(
    df: pd.DataFrame,
    feature_cols: list[str],
    cat_encoders: dict,
    dtype: str = "int32",
) -> pd.DataFrame:
    """Apply integer encoding to categorical columns in-place on a copy."""
    df_enc = df.copy()
    for col, enc in cat_encoders.items():
        if col in feature_cols:
            df_enc[col] = df_enc[col].map(enc).fillna(-1).astype(dtype)
    return df_enc


def _predict_batch(
    model_name: str,
    full_model: Any,
    df: pd.DataFrame,
    feature_cols: list[str],
    cat_encoders: dict,
) -> np.ndarray:
    """Predict a batch of rows for a given model."""
    if model_name == "lightgbm":
        df_enc = _encode_categoricals(df, feature_cols, cat_encoders, dtype="int32")
        return full_model.predict(df_enc[feature_cols])
    elif model_name == "xgboost":
        df_enc = _encode_categoricals(df, feature_cols, cat_encoders, dtype="int32")
        dmat = xgb.DMatrix(df_enc[feature_cols].values.astype(float), feature_names=feature_cols)
        return full_model.predict(dmat)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def predict_test(
    test_df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    results: list[dict[str, Any]],
    submissions_dir: Path,
) -> None:
    """
    Generate submission CSVs using recursive 2-step inference.

    wk52: predict with observed wk51 features (lag1_price = wk51 actual).
    wk1:  lag1_price was NaN — fill with wk52 prediction before predicting.
          This fixes the null-routing artifact where LGBM defaulted to historical mean.
    """
    wk52_mask = test_df["Year_Week"].str.endswith("-52")
    wk1_mask = ~wk52_mask

    for result in results:
        model_name = result["model"]
        full_model = result.get("full_model")
        if full_model is None:
            continue

        cat_encoders = result.get("cat_encoders", {})

        # Step 1: predict wk52 (all features observed)
        wk52_df = test_df[wk52_mask].copy()
        wk52_preds = _predict_batch(model_name, full_model, wk52_df, feature_cols, cat_encoders)
        wk52_pred_by_county = dict(zip(wk52_df["County"].values, wk52_preds))
        logger.info("%s wk52 preds: %s", model_name, {k: round(v, 2) for k, v in wk52_pred_by_county.items()})

        # Step 2: fill wk1 lag1_price with wk52 prediction, then predict
        wk1_df = test_df[wk1_mask].copy()
        wk1_df["lag1_price"] = wk1_df["County"].map(wk52_pred_by_county)

        # Also update derived features that depend on lag1
        if "price_momentum" in feature_cols:
            # momentum = lag1 - lag2; lag2 for wk1 = wk51 actual (already in lag2_price col)
            wk1_df["price_momentum"] = wk1_df["lag1_price"] - wk1_df["lag2_price"]

        if "deviation_from_rolling_mean" in feature_cols and "rolling_4w_mean" in feature_cols:
            wk1_df["deviation_from_rolling_mean"] = wk1_df["lag1_price"] - wk1_df["rolling_4w_mean"]

        wk1_preds = _predict_batch(model_name, full_model, wk1_df, feature_cols, cat_encoders)
        logger.info("%s wk1 preds: %s", model_name, dict(zip(wk1_df["County"].values, [round(p, 2) for p in wk1_preds])))

        # Assemble final submission
        all_counties = list(wk52_df["County"].values) + list(wk1_df["County"].values)
        all_year_weeks = list(wk52_df["Year_Week"].values) + list(wk1_df["Year_Week"].values)
        all_preds = list(wk52_preds) + list(wk1_preds)

        submission = pd.DataFrame({
            "County": all_counties,
            "Year_Week": all_year_weeks,
            "WholeSale_pred": all_preds,
        })
        submission["ID"] = submission.apply(
            lambda r: f"{r['County']}_Week_{int(r['Year_Week'].split('-')[1])}", axis=1
        )
        submission = submission[["ID", "WholeSale_pred"]].rename(
            columns={"WholeSale_pred": "Target_RMSE"}
        )
        submission["Target_MAE"] = submission["Target_RMSE"]

        out_path = submissions_dir / f"{model_name}.csv"
        submission.to_csv(out_path, index=False, encoding="utf-8")
        logger.info("Submission saved: %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline model training")
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    experiment_id = cfg["experiment_id"]
    logger.info("Experiment: %s", experiment_id)

    # --- Paths ---
    oof_dir = Path(cfg["oof_dir"])
    model_dir = Path(cfg["models_dir"])
    submissions_dir = Path(cfg["submissions_dir"])
    results_path = Path(cfg["results_path"])
    for p in [oof_dir, model_dir, submissions_dir, results_path.parent]:
        p.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    logger.info("Loading train: %s", cfg["train_path"])
    df = pd.read_parquet(cfg["train_path"])
    logger.info("Train shape: %s", df.shape)

    test_df = None
    if cfg.get("test_path") and Path(cfg["test_path"]).exists():
        test_df = pd.read_parquet(cfg["test_path"])
        logger.info("Test shape: %s", test_df.shape)

    feature_cols: list[str] = cfg["feature_cols"]
    categorical_cols: list[str] = cfg["categorical_cols"]
    target_col: str = cfg["target_col"]
    time_col: str = cfg["time_col"]

    # Validate features exist in dataframe
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning("Missing feature columns: %s", missing)
        feature_cols = [c for c in feature_cols if c in df.columns]

    logger.info("Features: %d | Target: %s | Time: %s", len(feature_cols), target_col, time_col)

    all_results: list[dict[str, Any]] = []
    start_total = time.time()

    # --- LightGBM ---
    if cfg["models"]["lightgbm"].get("enabled", True):
        logger.info("=" * 50)
        logger.info("Training LightGBM...")
        t0 = time.time()
        lgb_result = train_lightgbm(df, feature_cols, categorical_cols, target_col, time_col, cfg, model_dir, oof_dir)
        lgb_result["train_time_s"] = round(time.time() - t0, 1)
        all_results.append(lgb_result)

    # --- XGBoost ---
    if cfg["models"]["xgboost"].get("enabled", True):
        logger.info("=" * 50)
        logger.info("Training XGBoost...")
        t0 = time.time()
        xgb_result = train_xgboost(df, feature_cols, categorical_cols, target_col, time_col, cfg, model_dir, oof_dir)
        xgb_result["train_time_s"] = round(time.time() - t0, 1)
        all_results.append(xgb_result)

    # --- Trend extrapolation ---
    if cfg["models"]["trend_extrapolation"].get("enabled", True):
        logger.info("=" * 50)
        logger.info("Training Trend Extrapolation...")
        t0 = time.time()
        trend_result = train_trend_extrapolation(df, target_col, time_col, cfg, oof_dir)
        trend_result["train_time_s"] = round(time.time() - t0, 1)
        all_results.append(trend_result)

    total_time = round(time.time() - start_total, 1)

    # --- Test predictions ---
    if test_df is not None:
        logger.info("=" * 50)
        logger.info("Generating test predictions...")
        predict_test(test_df, feature_cols, categorical_cols, all_results, submissions_dir)

    # --- Summary ---
    summary: dict[str, Any] = {
        "experiment_id": experiment_id,
        "feature_version": cfg.get("feature_version", "v2"),
        "n_train": len(df),
        "n_features": len(feature_cols),
        "naive_baseline": 2.907,
        "models": [],
    }

    logger.info("=" * 50)
    logger.info("RESULTS SUMMARY")
    logger.info("%-25s %10s %10s %12s", "Model", "CV Mean", "CV Std", "Train Time")
    logger.info("-" * 60)

    for r in all_results:
        row = {
            "model": r["model"],
            "cv_mean": round(r["cv_mean"], 4),
            "cv_std": round(r["cv_std"], 4),
            "fold_scores": [round(s, 4) for s in r["fold_scores"]],
            "train_score": round(r["train_score"], 4) if r.get("train_score") is not None else None,
            "train_time_s": r.get("train_time_s", 0),
            "oof_path": r["oof_path"],
        }
        summary["models"].append(row)
        logger.info(
            "%-25s %10.4f %10.4f %11.1fs",
            r["model"], r["cv_mean"], r["cv_std"], r.get("train_time_s", 0)
        )

    logger.info("-" * 60)
    logger.info("Total time: %.1fs", total_time)

    best = min(all_results, key=lambda x: x["cv_mean"])
    logger.info("Best model: %s (CV %.4f vs naive %.3f)", best["model"], best["cv_mean"], 2.907)
    summary["best_model"] = best["model"]
    summary["best_cv"] = round(best["cv_mean"], 4)
    summary["total_time_s"] = total_time

    # Save results JSON
    serializable_summary = {
        k: v for k, v in summary.items()
        if k != "feature_importances"
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(serializable_summary, f, indent=2)
    logger.info("Results saved: %s", results_path)

    # Print machine-parseable summary line
    print(f"\nEXPERIMENT_SUMMARY: {json.dumps(serializable_summary)}")


if __name__ == "__main__":
    main()
