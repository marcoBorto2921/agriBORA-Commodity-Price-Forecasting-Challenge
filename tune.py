"""
Phase 5a — Hyperparameter Tuning (Optuna).
Walk-forward CV adaptation for agriBORA Commodity Price Forecasting.

Usage:
    .venv/Scripts/python tune.py --config configs/tuning_lgbm.yaml
    .venv/Scripts/python tune.py --config configs/tuning_xgb.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Config + data
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def competition_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """0.5 * MAE + 0.5 * RMSE — lower is better."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return 0.5 * mae + 0.5 * rmse


def get_walk_forward_splits(
    df: pd.DataFrame,
    time_col: str,
    n_folds: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Identical to train_baseline.py — do not modify."""
    sorted_weeks = sorted(df[time_col].unique())
    n_weeks = len(sorted_weeks)
    val_start = int(n_weeks * 0.70)
    cutpoints = np.linspace(val_start - 1, n_weeks - 3, n_folds, dtype=int)
    cutpoints = np.unique(cutpoints)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for cutpoint in cutpoints:
        cutoff_week = sorted_weeks[cutpoint]
        next_two = sorted_weeks[cutpoint + 1 : cutpoint + 3]
        if not next_two:
            continue
        train_idx = df.index[df[time_col] <= cutoff_week].to_numpy()
        val_idx = df.index[df[time_col].isin(next_two)].to_numpy()
        if len(train_idx) > 0 and len(val_idx) > 0:
            splits.append((train_idx, val_idx))
    return splits


def encode_categoricals(
    df: pd.DataFrame,
    cat_cols: list[str],
    feature_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Integer-encode categorical columns in feature_cols. Returns (df_enc, encoders)."""
    df_enc = df.copy()
    cat_encoders: dict[str, dict] = {}
    for col in cat_cols:
        if col not in feature_cols:
            continue
        unique_vals = sorted(df[col].dropna().unique())
        cat_encoders[col] = {v: i for i, v in enumerate(unique_vals)}
        df_enc[col] = df[col].map(cat_encoders[col]).fillna(-1).astype("int32")
    return df_enc, cat_encoders


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------


def make_objective(train_df: pd.DataFrame, cfg: dict[str, Any]):
    """Returns an Optuna objective function (minimize competition_metric)."""
    feature_cols: list[str] = cfg["feature_cols"]
    target_col: str = cfg["target_col"]
    time_col: str = cfg["time_col"]
    cat_cols: list[str] = cfg.get("categorical_cols", [])
    ss: dict[str, Any] = cfg["search_space"]
    model_name: str = cfg["model"]

    df_enc, _ = encode_categoricals(train_df, cat_cols, feature_cols)
    cat_in_features = [c for c in cat_cols if c in feature_cols]
    splits = get_walk_forward_splits(train_df, time_col, cfg["n_folds"])
    log.info("Walk-forward splits: %d", len(splits))

    def objective(trial: optuna.Trial) -> float:
        # ── MODEL SWITCH — edit params here if needed ──────────────────────
        if model_name == "lightgbm":
            import lightgbm as lgb

            params = {
                "objective": trial.suggest_categorical(
                    "objective",
                    ss.get("objective", ["regression_l1", "regression"]),
                ),
                "metric": "mae",
                "learning_rate": trial.suggest_float(
                    "learning_rate", *ss["learning_rate"], log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", *ss["num_leaves"]),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples", *ss["min_child_samples"]
                ),
                "feature_fraction": trial.suggest_float(
                    "feature_fraction", *ss["feature_fraction"]
                ),
                "bagging_fraction": trial.suggest_float(
                    "bagging_fraction", *ss["bagging_fraction"]
                ),
                "bagging_freq": 1,
                "reg_alpha": trial.suggest_float(
                    "reg_alpha", *ss["reg_alpha"], log=True
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", *ss["reg_lambda"], log=True
                ),
                "n_estimators": 2000,
                "random_state": cfg["seed"],
                "verbose": -1,
                "device": "cpu",
            }

            fold_scores = []
            for train_idx, val_idx in splits:
                X_tr = df_enc.loc[train_idx][feature_cols]
                y_tr = df_enc.loc[train_idx][target_col].values
                X_val = df_enc.loc[val_idx][feature_cols]
                y_val = df_enc.loc[val_idx][target_col].values

                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        lgb.early_stopping(50, verbose=False),
                        lgb.log_evaluation(-1),
                    ],
                    categorical_feature=cat_in_features if cat_in_features else "auto",
                )
                val_pred = model.predict(X_val)
                fold_scores.append(competition_metric(y_val, val_pred))

        elif model_name == "xgboost":
            import xgboost as xgb

            objective_choice = trial.suggest_categorical(
                "objective",
                ss.get("objective", ["reg:absoluteerror", "reg:squarederror"]),
            )
            params = {
                "objective": objective_choice,
                "eval_metric": "mae",
                "tree_method": "hist",
                "device": "cuda",
                "learning_rate": trial.suggest_float(
                    "learning_rate", *ss["learning_rate"], log=True
                ),
                "max_depth": trial.suggest_int("max_depth", *ss["max_depth"]),
                "min_child_weight": trial.suggest_int(
                    "min_child_weight", *ss["min_child_weight"]
                ),
                "subsample": trial.suggest_float("subsample", *ss["subsample"]),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", *ss["colsample_bytree"]
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha", *ss["reg_alpha"], log=True
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", *ss["reg_lambda"], log=True
                ),
                "n_estimators": 2000,
                "early_stopping_rounds": 100,
                "random_state": cfg["seed"],
                "verbosity": 0,
            }

            fold_scores = []
            for train_idx, val_idx in splits:
                X_tr = df_enc.loc[train_idx][feature_cols]
                y_tr = df_enc.loc[train_idx][target_col].values
                X_val = df_enc.loc[val_idx][feature_cols]
                y_val = df_enc.loc[val_idx][target_col].values

                model = xgb.XGBRegressor(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                val_pred = model.predict(X_val)
                fold_scores.append(competition_metric(y_val, val_pred))

        else:
            raise ValueError(f"Unknown model: {model_name}")
        # ── END MODEL SWITCH ────────────────────────────────────────────────

        return float(np.mean(fold_scores))

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tuning_lgbm.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log.info(
        "Config: %s | Experiment: %s | Model: %s",
        args.config,
        cfg["experiment_id"],
        cfg["model"],
    )

    train_df = pd.read_parquet(cfg["train_path"])
    log.info("Train shape: %s", train_df.shape)

    objective = make_objective(train_df, cfg)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg["seed"]),
    )

    optuna_cfg = cfg.get("optuna", {})
    n_trials = optuna_cfg.get("n_trials", 100)
    timeout = optuna_cfg.get("timeout", None)

    log.info(
        "Starting Optuna: %d trials | timeout: %ss | direction: minimize",
        n_trials,
        timeout,
    )
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    elapsed = time.time() - t0

    best = study.best_trial
    log.info(
        "Best CV: %.6f (trial %d) | Time: %.0fs", best.value, best.number, elapsed
    )
    log.info("Best params: %s", best.params)

    results_dir = Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "experiment_id": cfg["experiment_id"],
        "model": cfg["model"],
        "feature_version": cfg.get("feature_version", ""),
        "best_cv_score": best.value,
        "best_params": best.params,
        "n_trials_completed": len(study.trials),
        "direction": "minimize",
        "train_time_seconds": round(elapsed, 1),
    }
    out_path = results_dir / f"tuning_{cfg['model']}_best.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info("Best params saved: %s", out_path)

    summary = {
        "experiment_id": cfg["experiment_id"],
        "model": cfg["model"],
        "feature_version": cfg.get("feature_version", ""),
        "cv_score": round(best.value, 6),
        "best_params": best.params,
        "n_trials": len(study.trials),
        "train_time_seconds": round(elapsed, 1),
        "timestamp": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
    }
    print("EXPERIMENT_SUMMARY:" + json.dumps(summary))


if __name__ == "__main__":
    main()
