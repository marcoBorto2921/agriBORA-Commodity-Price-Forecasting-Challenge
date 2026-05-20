# agriBORA Commodity Price Forecasting Challenge

Retrospective solution for the [agriBORA Commodity Price Forecasting Challenge](https://zindi.africa/competitions/agribora-commodity-price-forecasting-challenge/) on Zindi (closed December 27, 2025).

**Final result: Rank #1 — private LB +0.820** (previous LB#1 was +0.801).

---

## Problem

Forecast weekly wholesale maize price (KES/kg) for 5 Kenya counties — Kiambu, Kirinyaga, Mombasa, Nairobi, Uasin-Gishu — for Week 52 (Dec 22, 2025) and Week 1 (Dec 29, 2025).

**Metric:** `0.5 × MAE + 0.5 × RMSE` — lower is better. Zindi normalizes to higher=better on the leaderboard.

**Submission:** 10 rows (5 counties × 2 weeks). Columns `Target_RMSE` and `Target_MAE` must be identical.

---

## Dataset

| File | Description |
|------|-------------|
| `agriBORA_maize_prices.csv` | agriBORA training series (Oct 2023 – Oct 2025) |
| `agriBORA_maize_prices_weeks_46_to_51.csv` | Rolling ground truth Wk 46–51 (released during competition) |
| `agriBORA_Final_Weeks_maize_price.csv` | Ground truth Wk 52 + Wk 1 (sealed until end) |
| `kamis_maize_prices.csv` | KAMIS supplemental data: retail/wholesale prices + supply volumes, May 2021 – Jul 2025 |

Data is not redistributable. Download from the [Zindi competition page](https://zindi.africa/competitions/agribora-commodity-price-forecasting-challenge/).

**Key data characteristics:**
- Uasin-Gishu: ~60 obs — best covered, major producer county
- Nairobi: ~45 obs — major consumer hub, strong AR(1) structure (β=0.861)
- Kiambu: ~25 obs — peri-urban, augmented with KAMIS
- Kirinyaga: ~20 obs — Mt. Kenya region
- Mombasa: ~17 obs in agriBORA — coastal import hub, augmented with KAMIS (72 rows)
- All 5 counties highly correlated (min r=0.808)
- Strong seasonal pattern: lean season begins Dec–Jan, driving price spikes

---

## Approach

### Feature Engineering

Two feature sets were tested:

**v2 (22 features) — used for XGBoost:**
- AR lags: `lag1_price`, `lag2_price`
- Rolling stats: `rolling_4w_mean`, `rolling_4w_max`, `deviation_from_rolling_mean`, `price_vs_4w_max`
- Momentum: `price_momentum`, `consecutive_up_weeks`
- Seasonality: `week_of_year`, `is_lean_season`, `weeks_in_lean_season`, `is_short_harvest_window`, `weeks_to_long_harvest`
- Cross-county: `nairobi_lag1`, `ug_lag1`, `national_mean_lag1`, `nairobi_ug_spread`
- KAMIS: `kamis_wholesale_lag1`, `kamis_retail_lag1`, `kamis_ug_supply_lag1`, `kamis_national_supply_lag1`, `is_kamis_augmented`
- Category: `County` (label-encoded)

**v3 (15 features) — used for LightGBM:** v2 minus KAMIS features and some momentum features (pruned by importance).

**KAMIS augmentation:** Kiambu (30→65 rows) and Mombasa (17→50 rows) were augmented using per-county calibration ratios between KAMIS and agriBORA price series.

### Models

- **LightGBM** (v3 features, CPU): tuned via Optuna (100 trials), `objective=regression_l1`
- **XGBoost** (v2 features, CUDA): tuned via Optuna (100 trials), `objective=reg:absoluteerror`
- **Carryforward baseline**: predict wk52 = wk51 actual, predict wk1 = wk52 prediction

### CV Strategy

Walk-forward time series CV — 5 folds, cutoffs at last 30% of the timeline. Train ≤ cutoff week, validate on next 2 weeks. No random splits.

### Final Ensemble

County-specific blend: `alpha_c × XGB_tuned + (1 - alpha_c) × carryforward`

Alphas optimized on wk46–51 validation window, then refined by regime analysis:
- **Kiambu, Kirinyaga, Mombasa, Uasin-Gishu**: alpha=0.0 (pure carryforward from wk51 actual)
- **Nairobi**: alpha=0.927 (XGB dominant — consumer market in lean season, prices continued rising)

Key insight: 4 of 5 counties plateaued after the sharp wk49–51 rally (producer/import dynamics). Nairobi is the exception — lean season demand kept prices rising through wk52–1.

---

## Results

| ID | Model | Features | CV Score | Local Score | Private LB | Notes |
|----|-------|----------|----------|-------------|------------|-------|
| E01a | LightGBM default | v2 | 1.6739 | 2.69 | +0.373 | baseline |
| E01b | XGBoost default | v2 | 1.9245 | 2.23 | +0.479 | baseline |
| E01c | Trend extrapolation | v2 | 2.4594 | — | — | discarded |
| E02a | LightGBM pruned | v3 | 1.6879 | 2.42 | +0.435 | feature pruning |
| E02b | XGBoost pruned | v3 | 1.7355 | 3.07 | +0.286 | worse — v2 mandatory for XGB |
| E03a | LightGBM tuned | v3 | **1.2740** | 2.45 | +0.428 | Optuna 100 trials — CV overfit |
| E03b | XGBoost tuned | v2 | 1.6390 | **1.90** | +0.557 | Optuna 100 trials — new best |
| **E04** | **Blend optimal** | — | — | **0.77** | **+0.820** | **Rank #1** |

Naive baseline (last-value carryforward, wk46–51): **2.907**.

---

## Setup

```bash
git clone https://github.com/your-username/agribora-maize-forecasting
cd agribora-maize-forecasting

python -m venv .venv
# Windows
.venv\Scripts\pip install -r requirements.txt
# Linux/Mac
.venv/bin/pip install -r requirements.txt

# Download competition data from Zindi and place the data folder at:
# agribora-commodity-price-forecasting-challenge<hash>/
```

## Reproduce

Run scripts in order from the project root:

```bash
# 1. EDA (optional — outputs to outputs/eda/)
.venv/Scripts/python eda.py

# 2. Feature engineering — builds v2 and v3 feature sets
.venv/Scripts/python feature_engineering.py

# 3. FE validation
.venv/Scripts/python fe_validation.py

# 4. Baseline training (LightGBM + XGBoost + trend extrapolation)
.venv/Scripts/python train_baseline.py --config configs/train_baseline.yaml

# 5a. Hyperparameter tuning
.venv/Scripts/python tune.py --config configs/tuning_xgb.yaml
.venv/Scripts/python tune.py --config configs/tuning_lgbm.yaml

# 5b. Train with tuned params
.venv/Scripts/python train_baseline.py --config configs/train_xgb_tuned.yaml
.venv/Scripts/python train_baseline.py --config configs/train_lgbm_tuned.yaml

# 6. Blend (county-specific alpha optimization → final submission)
.venv/Scripts/python blend.py
```

Final submission: `outputs/submissions/blend_optimal/blend_optimal.csv`

---

## Hardware

- CPU: Intel Core i7
- GPU: NVIDIA GeForce RTX 2050 (4 GB VRAM) — used for XGBoost (`device=cuda`)
- OS: Windows 11
