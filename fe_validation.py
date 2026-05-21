"""
FE Validation — agriBORA Commodity Price Forecasting (Phase 3.5)
Walk-forward time-series CV + leakage check + feature importances.
Run: .venv/Scripts/python fe_validation.py
"""

import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────

TRAIN_PATH = "outputs/features/v2/train_fe.parquet"
TEST_PATH = "outputs/features/v2/test_fe.parquet"
TARGET_COL = "WholeSale"
RANDOM_STATE = 42

# Number of walk-forward validation splits
# Each fold: train on all weeks before cutoff, validate on next 2 weeks
N_FOLDS = 5
# Naive 2-step baseline score from EDA
NAIVE_BASELINE = 2.907

OUTPUT_DIR = Path("outputs/fe_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
]

LGBM_PARAMS = {
    "objective": "regression_l1",  # MAE-optimal base (metric blends MAE+RMSE)
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "verbose": -1,
    "n_jobs": -1,
}


# ── helpers ───────────────────────────────────────────────────────────────────


def competition_score(actual: np.ndarray, pred: np.ndarray) -> float:
    """0.5 * MAE + 0.5 * RMSE — competition metric."""
    mae = np.mean(np.abs(actual - pred))
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    return 0.5 * mae + 0.5 * rmse


def _flag(level: str, msg: str) -> None:
    print(f"[{level}] {msg}")


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in FEATURE_COLS if c in df.columns]


# ── load ──────────────────────────────────────────────────────────────────────


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    # County as categorical
    train["County"] = train["County"].astype("category")
    test["County"] = test["County"].astype("category")

    _flag("INFO", f"Train: {train.shape} | Test: {test.shape}")
    _flag(
        "INFO",
        f"Year_Week range: {train['Year_Week'].min()} to {train['Year_Week'].max()}",
    )
    return train, test


# ── Check 1: Walk-forward CV (predictive signal) ──────────────────────────────


def walk_forward_cv(
    train: pd.DataFrame,
) -> tuple[float, float, list[float], np.ndarray, np.ndarray]:
    """
    Walk-forward CV: train on all weeks before cutoff, validate on next 2 weeks.
    Uses N_FOLDS equally spaced cutoffs in the last ~30% of the timeline.
    Returns: mean_score, std_score, fold_scores, oof_preds, oof_actuals.
    """
    feat_cols = get_feature_cols(train) + ["County"]
    sorted_weeks = sorted(train["Year_Week"].unique())
    n_weeks = len(sorted_weeks)

    # Define cutoff weeks: last 30% of timeline, N_FOLDS folds
    start_idx = int(n_weeks * 0.65)
    cutoff_indices = np.linspace(start_idx, n_weeks - 3, N_FOLDS, dtype=int)
    cutoff_weeks = [sorted_weeks[i] for i in cutoff_indices]

    fold_scores = []
    oof_preds_list = []
    oof_actuals_list = []

    for fold_i, cutoff in enumerate(cutoff_weeks):
        val_weeks = sorted_weeks[
            sorted_weeks.index(cutoff) + 1 : sorted_weeks.index(cutoff) + 3
        ]  # next 2 weeks
        if not val_weeks:
            continue

        train_mask = train["Year_Week"] <= cutoff
        val_mask = train["Year_Week"].isin(val_weeks)

        X_tr = train.loc[train_mask, feat_cols]
        y_tr = train.loc[train_mask, TARGET_COL]
        X_val = train.loc[val_mask, feat_cols]
        y_val = train.loc[val_mask, TARGET_COL]

        if len(X_tr) < 10 or len(X_val) == 0:
            continue

        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)

        score = competition_score(y_val.values, preds)
        fold_scores.append(score)
        oof_preds_list.extend(preds)
        oof_actuals_list.extend(y_val.values)

        _flag(
            "INFO",
            f"  Fold {fold_i + 1}: cutoff={cutoff}, val_weeks={val_weeks}, "
            f"n_val={len(y_val)}, score={score:.4f}",
        )

    mean_score = float(np.mean(fold_scores)) if fold_scores else float("nan")
    std_score = float(np.std(fold_scores)) if fold_scores else float("nan")
    oof_preds = np.array(oof_preds_list)
    oof_actuals = np.array(oof_actuals_list)

    return mean_score, std_score, fold_scores, oof_preds, oof_actuals


# ── Check 2: Leakage detection ────────────────────────────────────────────────


def check_leakage(train: pd.DataFrame, cv_score: float) -> tuple[float, str]:
    """
    Fit on full train, score in-sample. Compare to CV.
    Large train/CV gap = potential leakage.
    """
    feat_cols = get_feature_cols(train) + ["County"]
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(train[feat_cols], train[TARGET_COL])
    train_preds = model.predict(train[feat_cols])
    train_score = competition_score(train[TARGET_COL].values, train_preds)

    gap = cv_score - train_score  # positive = CV worse than train (expected)
    gap_pct = (cv_score - train_score) / cv_score * 100 if cv_score > 0 else 0

    if gap_pct > 50:
        verdict = "CRITICAL"
    elif gap_pct > 25:
        verdict = "WARN"
    else:
        verdict = "PASS"

    _flag(
        verdict,
        f"  Train score={train_score:.4f}, CV score={cv_score:.4f}, "
        f"gap={gap:.4f} ({gap_pct:.1f}%) -> {verdict}",
    )
    return train_score, verdict


# ── Check 3: Adversarial validation ──────────────────────────────────────────


def adversarial_validation(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[float, str]:
    """
    Binary classifier: can LightGBM distinguish train from test?
    AUC ~ 0.5 = indistinguishable (PASS), AUC > 0.85 = distribution shift (CRITICAL).
    Note: test has only 10 rows — result is directional only.
    """
    from sklearn.metrics import roc_auc_score

    feat_cols = get_feature_cols(train)

    # Align columns between train and test
    common_feats = [c for c in feat_cols if c in test.columns]
    if len(common_feats) < 3:
        _flag("WARN", "  Too few common features for adversarial validation — skipped")
        return float("nan"), "SKIP"

    tr = train[common_feats].copy()
    te = test[common_feats].copy()

    tr["_label"] = 0
    te["_label"] = 1
    combined = pd.concat([tr, te], ignore_index=True)

    X = combined[common_feats].fillna(-9999)
    y = combined["_label"]

    if y.sum() < 2:
        _flag("WARN", "  Test too small for adversarial validation — skipped")
        return float("nan"), "SKIP"

    # Single stratified fold (test is tiny)
    model = lgb.LGBMClassifier(
        n_estimators=100, num_leaves=8, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1
    )
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]

    try:
        auc = roc_auc_score(y, proba)
    except Exception:
        auc = float("nan")

    if np.isnan(auc):
        verdict = "SKIP"
    elif auc > 0.85:
        verdict = "CRITICAL"
    elif auc > 0.70:
        verdict = "WARN"
    else:
        verdict = "PASS"

    _flag(
        verdict if verdict != "SKIP" else "INFO",
        f"  Adversarial AUC={auc:.4f} (test n={len(te)}) -> {verdict}",
    )
    _flag("INFO", "  Note: test=10 rows only — AUC directional, not conclusive.")
    return auc, verdict


# ── Check 4: Feature importances ─────────────────────────────────────────────


def feature_importance_check(train: pd.DataFrame) -> tuple[list[str], int]:
    """Fit on full train, extract LightGBM feature importances."""
    feat_cols = get_feature_cols(train) + ["County"]
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(train[feat_cols], train[TARGET_COL])

    importance = pd.Series(model.feature_importances_, index=feat_cols).sort_values(
        ascending=False
    )
    dead = (importance == 0).sum()
    top5 = importance.head(5).index.tolist()

    _flag("INFO", f"  Top 5 features: {top5}")
    _flag(
        "INFO" if dead == 0 else "WARN",
        f"  Dead features (importance=0): {dead}/{len(feat_cols)}",
    )

    if dead > 0:
        dead_list = importance[importance == 0].index.tolist()
        _flag("INFO", f"  Dead: {dead_list}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, max(4, len(feat_cols) * 0.35)))
    importance.sort_values().plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("LightGBM Feature Importance (gain)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=120, bbox_inches="tight")
    plt.close()
    _flag("INFO", f"  Plot saved: {OUTPUT_DIR}/feature_importance.png")

    return top5, dead


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 60)
    print("FE VALIDATION — agriBORA v1")
    print("=" * 60)

    train, test = load_data()

    # ── Check 1: Predictive signal ──
    print("\n--- Check 1: Walk-forward CV (predictive signal) ---")
    cv_mean, cv_std, fold_scores, oof_preds, oof_actuals = walk_forward_cv(train)
    _flag("INFO", f"CV score: {cv_mean:.4f} ± {cv_std:.4f}")
    _flag("INFO", f"Naive 2-step baseline: {NAIVE_BASELINE:.4f}")

    if np.isnan(cv_mean):
        signal_verdict = "CRITICAL"
        _flag("CRITICAL", "CV failed — no valid folds")
    elif cv_mean >= NAIVE_BASELINE:
        signal_verdict = "WARN"
        _flag(
            "WARN",
            f"CV score {cv_mean:.4f} >= naive baseline {NAIVE_BASELINE:.4f} — not beating baseline yet",
        )
    elif cv_mean < NAIVE_BASELINE * 0.5:
        signal_verdict = "PASS"
        _flag(
            "INFO",
            f"PASS — CV score {cv_mean:.4f} well below baseline {NAIVE_BASELINE:.4f}",
        )
    else:
        signal_verdict = "PASS"
        _flag(
            "INFO", f"PASS — CV score {cv_mean:.4f} beats baseline {NAIVE_BASELINE:.4f}"
        )

    # ── Check 2: Leakage ──
    print("\n--- Check 2: Leakage detection ---")
    train_score, leakage_verdict = check_leakage(train, cv_mean)

    # ── Check 3: Adversarial validation ──
    print("\n--- Check 3: Adversarial validation ---")
    adv_auc, adv_verdict = adversarial_validation(train, test)

    # ── Check 4: Feature importances ──
    print("\n--- Check 4: Feature importances ---")
    top5, n_dead = feature_importance_check(train)
    n_total_feats = len(get_feature_cols(train)) + 1  # +1 for County
    if n_dead / n_total_feats > 0.5:
        importance_verdict = "CRITICAL"
    elif n_dead / n_total_feats > 0.3:
        importance_verdict = "WARN"
    else:
        importance_verdict = "PASS"

    # ── Overall verdict ──
    print("\n" + "=" * 60)
    print("OVERALL VERDICT")
    print("=" * 60)
    criticals = [
        v
        for v in [signal_verdict, leakage_verdict, adv_verdict, importance_verdict]
        if v == "CRITICAL"
    ]
    overall = "STOP — fix criticals" if criticals else "GO"
    _flag("INFO" if not criticals else "CRITICAL", f"Overall: {overall}")
    _flag("INFO", f"  Check 1 (signal):      {signal_verdict}")
    _flag("INFO", f"  Check 2 (leakage):     {leakage_verdict}")
    _flag("INFO", f"  Check 3 (adversarial): {adv_verdict}")
    _flag("INFO", f"  Check 4 (importance):  {importance_verdict}")

    # ── Write report ──
    report_path = Path(".claude/features/FE_VALIDATION_REPORT.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"""# FE Validation Report — agriBORA Commodity Price Forecasting
**Date:** 2026-05-19
**Train shape:** {train.shape[0]} rows x {train.shape[1]} cols
**Test shape:** {test.shape[0]} rows
**Feature version:** v1

---

## Check 1 — Predictive Signal
**CV score (walk-forward, {N_FOLDS} folds):** {cv_mean:.4f} +/- {cv_std:.4f}
**Naive 2-step baseline:** {NAIVE_BASELINE:.4f}
**Fold scores:** {[round(s, 4) for s in fold_scores]}
**Result:** {signal_verdict}

## Check 2 — Leakage Detection
**Train score (in-sample):** {train_score:.4f}
**CV score (out-of-fold):** {cv_mean:.4f}
**Gap (CV - train):** {cv_mean - train_score:.4f} ({(cv_mean - train_score) / cv_mean * 100:.1f}% of CV)
**Result:** {leakage_verdict}

## Check 3 — Adversarial Validation
**Adversarial AUC (train vs test):** {adv_auc:.4f}
**Note:** test=10 rows only — AUC is directional, not conclusive.
**Result:** {adv_verdict}

## Check 4 — Feature Importances
**Top 5 features:** {", ".join(top5)}
**Dead features (importance=0):** {n_dead}/{n_total_feats}
**Plot:** outputs/fe_validation/feature_importance.png
**Result:** {importance_verdict}

---

## Overall Verdict
**{overall}**

## Issues Requiring Attention
""")
        if signal_verdict != "PASS":
            f.write(
                f"- [Check 1] CV score {cv_mean:.4f} vs baseline {NAIVE_BASELINE:.4f} — model not yet beating naive\n"
            )
        if leakage_verdict == "CRITICAL":
            f.write(
                f"- [Check 2] Leakage suspect — train score {train_score:.4f} much lower than CV {cv_mean:.4f}\n"
            )
        if adv_verdict == "CRITICAL":
            f.write(
                f"- [Check 3] Adversarial AUC {adv_auc:.4f} — distribution shift between train and test\n"
            )
        if importance_verdict in ("WARN", "CRITICAL"):
            f.write(
                f"- [Check 4] {n_dead}/{n_total_feats} features have zero importance\n"
            )
        if not criticals and signal_verdict == "PASS":
            f.write("- None.\n")

    _flag("INFO", f"\nReport saved: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
