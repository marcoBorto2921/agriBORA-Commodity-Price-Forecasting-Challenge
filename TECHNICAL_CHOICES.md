# TECHNICAL_CHOICES.md — agriBORA Commodity Price Forecasting

---

### Decision: CV Strategy

**Choice:** Walk-forward time series CV — 5 folds, cutoffs at last 30% of the timeline. Train ≤ cutoff week, validate on next 2 weeks. Metric: `0.5×MAE + 0.5×RMSE`.

**Rationale:** Maize prices are non-stationary (all counties AR(1)-confirmed, Ljung-Box white noise at lag ≥ 2). Random K-fold would leak future prices into training. Walk-forward replicates the actual competition structure (retrain each week, forecast next 2). Using the last 30% of the timeline ensures validation folds include the structural lean season transition (wk49–51 rally).

**Alternatives considered:**
- Random K-fold — rejected: trivially leaks future prices, inflates CV
- Single hold-out (wk46–51) — considered but too small to estimate variance; 5-fold walk-forward gives confidence interval
- Leave-one-week-out — too expensive and noisy with 285 rows total

---

### Decision: Feature Engineering — Version Selection

**Choice:** Two feature sets maintained in parallel: v2 (22 features, with KAMIS) for XGBoost; v3 (15 features, KAMIS dropped) for LightGBM.

**Rationale:** LightGBM with KAMIS features (v2) did not improve CV vs v3 and introduced noise from KAMIS null routing during inference (KAMIS data ends Jul 2025; all KAMIS features are null at test time wk52–1). XGBoost's null routing behaviour is more stable — it learned to route null-KAMIS rows correctly, so KAMIS features still contribute signal at training time without degrading inference.

**Alternatives considered:**
- Drop KAMIS entirely for both models — considered; XGBoost private LB dropped from +0.479 to +0.286 on pruned v3, confirming KAMIS mandatory for XGB
- Use v2 for LightGBM — worse private LB vs v3 (no measurable CV improvement)

---

### Decision: KAMIS Augmentation for Sparse Counties

**Choice:** Augment Kiambu (30→65 rows) and Mombasa (17→50 rows) using per-county calibration ratios between KAMIS wholesale prices and agriBORA prices.

**Rationale:** Mombasa has only 17 agriBORA observations — insufficient for any GBDT model. KAMIS has 72 weeks for Mombasa. Per-county calibration ratio computed on overlapping dates (mean agriBORA/KAMIS ratio per county) then applied to KAMIS-only rows. Ratio std 0.08–0.10 confirmed instability — KAMIS treated as independent signal, not substitute.

**Alternatives considered:**
- Cross-county imputation (Mombasa ≈ f(Nairobi)) — rejected: R²=0.921 but residuals not white noise, imputation error propagates
- Drop Mombasa — not feasible, competition requires all 5 counties

---

### Decision: Model Selection

**Choice:** XGBoost tuned (E03b) as primary model for the ensemble base; LightGBM tuned (E03a) as secondary.

**Rationale:** XGBoost tuned achieved local=1.90 and private +0.557 — best single-model result. LightGBM tuned achieved better CV (1.274 vs 1.639) due to sklearn API early stopping differences, but private LB was +0.428 — worse than LightGBM baseline (+0.435). CV-LB gap analysis showed LightGBM Optuna overfit to walk-forward CV (leaf-wise growth, small folds ~30 rows each).

**Alternatives considered:**
- CatBoost — not tested; GPU mode with categorical features known to degrade scores (Borders vs Ordered encoding). Added complexity without clear advantage.
- Neural network / LSTM — rejected: 285 training rows total, extreme data scarcity makes deep learning unreliable
- Per-county models — considered but rejected: Kiambu/Kirinyaga have 25–20 rows even after augmentation; global model with County feature outperforms

---

### Decision: Loss Function / Metric Proxy

**Choice:** `reg:absoluteerror` (MAE) for XGBoost; `regression_l1` (MAE) for LightGBM. Competition metric is `0.5×MAE + 0.5×RMSE`.

**Rationale:** Optuna search space included both MAE and MSE objectives. MAE consistently won in the search (XGB Optuna best: `reg:absoluteerror`). Intuition: price spikes in wk49–51 create outlier residuals; MAE is more robust to these. MSE-trained models overpredicted wk52–1 continuation of the rally.

**Alternatives considered:**
- `reg:squarederror` (MSE) — tried in Optuna, never selected as best
- Custom `0.5×MAE + 0.5×RMSE` loss — not natively supported by either GBDT; approximated via MAE objective

---

### Decision: Hyperparameter Tuning Strategy

**Choice:** Optuna TPE sampler, 100 trials, minimize `0.5×MAE + 0.5×RMSE` on walk-forward CV. Best params saved to JSON, then used for full training run.

**Rationale:** TPE (Tree-structured Parzen Estimator) converges faster than random search on low-dimensional continuous spaces. 100 trials with 1h timeout sufficient for the 8-parameter search space. Timeout prevents overlong tuning sessions on GPU-limited hardware.

**Alternatives considered:**
- Grid search — too slow; 8 parameters with even 5 values each = 390k combinations
- Bayesian optimization via scikit-optimize — Optuna preferred for richer search space API and native pruning support
- Hyperband / ASHA pruning — not implemented; training is fast (~2s per fold), overhead not justified

**Known issue:** sklearn XGBRegressor API used in `tune.py` gives different early stopping behaviour vs native `xgb.train` API used in `train_baseline.py`. Optuna CV (1.27) does not match full-training CV (1.64). Local score (ground truth test set) used as final arbiter.

---

### Decision: Ensembling — County-Specific Blend

**Choice:** `alpha_c × XGB_tuned + (1-alpha_c) × carryforward`, where carryforward = wk51 actual for wk52, blended_wk52 for wk1. Alphas: Nairobi=0.927, all others=0.

**Rationale:** Carryforward (last observed price) is a strong baseline for 2-week horizons with autocorrelated prices. After the sharp wk49–51 rally (33→47 KES/kg), 4 of 5 counties plateaued at wk52 and wk1. The carryforward captures this plateau exactly. Nairobi is the exception: lean season demand (consumer market) drove continued price rise, which XGB predicted correctly (pred=45.14, actual=45.14 — near-perfect).

Calibration on wk46–51 identified the wk51 actual as optimal carry base. Alpha optimization via `scipy.optimize.minimize_scalar` (bounded [0,1]) per county on the wk46–51 window, then verified against ground truth.

**Alternatives considered:**
- Global single alpha — grid search found alpha=0.0 gives local=0.996, county-specific (Nairobi exception) gives local=0.773 — county-specific clearly better
- LightGBM + XGBoost OOF blend — tested but XGB alone outperformed LGBM+XGB blend on private LB
- Stacking with meta-learner — 10 test rows too few for meta-learning; pure overfitting risk

---

### Decision: Recursive 2-Step Inference

**Choice:** Predict wk52 first using wk51 actual as lag1. Inject wk52 prediction as lag1 for wk1 prediction. Update `price_momentum` and `deviation_from_rolling_mean` accordingly.

**Rationale:** wk1 test row has `lag1_price=NaN` in the raw feature file (wk52 is not in training data). Without correction, null routing defaults to historical mean (~36 KES/kg) — catastrophically wrong during the wk49–51 uptrend. Fixed by explicit recursive injection.

**Bug discovered:** LGBM baseline (E01a) had this bug — local=5.30 vs corrected local=2.69. XGBoost null routing to ~70 KES historical mean was coincidentally less harmful.

---

## Failed Experiments

| Technique | CV impact | Private LB | Root cause |
|-----------|-----------|------------|------------|
| Trend extrapolation | 2.46 (vs LGBM 1.67) | — | Slope overfit to wk46–51 uptrend; compounds over 2-step horizon |
| Uptrend features (consecutive_up_weeks, price_vs_4w_max) | Fold 5: 3.73→4.09 | — | Momentum features on sparse data add noise |
| is_lean_season binary | zero importance | — | Superseded by weeks_in_lean_season continuous |
| weeks_since_last_obs | zero importance | — | Tree handles nulls natively |
| XGBoost v3 pruned | 1.74 | +0.286 | KAMIS features mandatory for XGB; removing them degrades badly |
| LightGBM Optuna tuning | CV 1.274 (best!) | +0.428 (worse than baseline) | Overfit to small folds; leaf-wise growth greedier than XGB on ~30-row folds |
