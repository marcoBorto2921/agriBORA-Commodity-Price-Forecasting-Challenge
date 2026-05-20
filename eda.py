"""
EDA — agriBORA Commodity Price Forecasting
Run: .venv/Scripts/python eda.py
Outputs: stdout flags + plots in outputs/eda/
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "eda"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, adfuller, pacf, grangercausalitytests

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────

DATA_DIR = "agribora-commodity-price-forecasting-challenge20260114-30357-1miwudk"
TRAIN_PATH = f"{DATA_DIR}/agriBORA_maize_prices.csv"
ROLLING_PATH = f"{DATA_DIR}/agriBORA_maize_prices_weeks_46_to_51.csv"
KAMIS_PATH = f"{DATA_DIR}/kamis_maize_prices.csv"
TARGET_COL = "WholeSale"
TIME_COL = "Date"
TARGET_COUNTIES = ["Kiambu", "Kirinyaga", "Mombasa", "Nairobi", "Uasin-Gishu"]
TASK = "regression"
MODALITY = "time_series"

OUTPUT_DIR = Path("outputs/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────


def _flag(level: str, msg: str) -> None:
    print(f"[{level}] {msg}")


def _savefig(name: str) -> None:
    path = OUTPUT_DIR / name
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    _flag("INFO", f"Plot saved: {path}")


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


# ── load ──────────────────────────────────────────────────────────────────────


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load agriBORA train, rolling ground truth, and KAMIS datasets."""
    train = pd.read_csv(TRAIN_PATH, encoding="utf-8")
    train[TIME_COL] = pd.to_datetime(train[TIME_COL])
    train = train.sort_values([TIME_COL, "County"]).reset_index(drop=True)

    rolling = pd.read_csv(ROLLING_PATH, encoding="utf-8")
    rolling[TIME_COL] = pd.to_datetime(rolling[TIME_COL])

    kamis = pd.read_csv(KAMIS_PATH, encoding="utf-8")
    kamis["Date"] = pd.to_datetime(kamis["Date"])

    _flag("INFO", f"agriBORA train: {train.shape} | rolling wk46-51: {rolling.shape} | KAMIS: {kamis.shape}")
    return train, rolling, kamis


# ── Section 1: Time Series Structure ──────────────────────────────────────────


def check_time_series_structure(train: pd.DataFrame, rolling: pd.DataFrame) -> None:
    _section("1. TIME SERIES STRUCTURE")

    # Combine train + rolling for full picture
    full = pd.concat([train, rolling], ignore_index=True)
    full[TIME_COL] = pd.to_datetime(full[TIME_COL])
    full = full.drop_duplicates(subset=["County", TIME_COL])

    # 1a. Line plot per target county
    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
    colors = ["steelblue", "darkorange", "green", "red", "purple"]
    for ax, county, color in zip(axes, TARGET_COUNTIES, colors):
        df_c = full[full["County"] == county].sort_values(TIME_COL)
        ax.plot(df_c[TIME_COL], df_c[TARGET_COL], marker="o", markersize=3,
                linewidth=1.2, color=color, label=county)
        ax.axvline(pd.Timestamp("2025-11-10"), color="black", linestyle="--",
                   linewidth=0.8, alpha=0.7, label="Wk 46 (comp start)")
        ax.set_ylabel("KES/kg")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Date")
    fig.suptitle("agriBORA WholeSale price per county (train + rolling wk46-51)", fontsize=12)
    plt.tight_layout()
    _savefig("1a_price_per_county.png")

    # 1b. Obs count per target county
    _flag("INFO", "Observation count per target county (train only):")
    train_target = train[train["County"].isin(TARGET_COUNTIES)]
    counts = train_target.groupby("County").size().reindex(TARGET_COUNTIES)
    for county, n in counts.items():
        level = "CRITICAL" if n < 15 else "WARN" if n < 30 else "INFO"
        _flag(level, f"  {county}: {n} obs")

    # 1c. Week-over-week % change per county
    _flag("INFO", "\nWeek-over-week WholeSale % change (train only):")
    for county in TARGET_COUNTIES:
        df_c = train[train["County"] == county].sort_values(TIME_COL)[TARGET_COL]
        if len(df_c) < 3:
            _flag("WARN", f"  {county}: too few obs for WoW analysis")
            continue
        wow = df_c.pct_change().dropna()
        _flag("INFO", f"  {county}: mean={wow.mean()*100:.1f}%, std={wow.std()*100:.1f}%, "
              f"max_spike={wow.max()*100:.1f}%")

    # 1d. ACF/PACF per county (combined all counties as one series — KAMIS richer)
    _flag("INFO", "\nACF/PACF on Nairobi (most complete county):")
    nairobi = train[train["County"] == "Nairobi"].sort_values(TIME_COL)[TARGET_COL].values
    if len(nairobi) >= 12:
        n_lags = min(12, len(nairobi) // 2 - 1)
        acf_vals = acf(nairobi, nlags=n_lags, fft=True)
        pacf_vals = pacf(nairobi, nlags=n_lags)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        lags = np.arange(n_lags + 1)
        conf = 1.96 / np.sqrt(len(nairobi))
        axes[0].bar(lags, acf_vals, color="steelblue")
        axes[0].axhline(conf, color="red", linestyle="--", linewidth=0.8)
        axes[0].axhline(-conf, color="red", linestyle="--", linewidth=0.8)
        axes[0].set_title("ACF — Nairobi WholeSale")
        axes[1].bar(lags, pacf_vals, color="steelblue")
        axes[1].axhline(conf, color="red", linestyle="--", linewidth=0.8)
        axes[1].axhline(-conf, color="red", linestyle="--", linewidth=0.8)
        axes[1].set_title("PACF — Nairobi WholeSale")
        plt.tight_layout()
        _savefig("1d_acf_pacf_nairobi.png")
        sig_lags = [i for i, v in enumerate(acf_vals[1:], 1) if abs(v) > conf]
        _flag("INFO", f"  Significant ACF lags (Nairobi): {sig_lags}")


# ── Section 2: Cross-dataset Calibration ──────────────────────────────────────


def check_calibration(train: pd.DataFrame, kamis: pd.DataFrame) -> None:
    _section("2. CROSS-DATASET CALIBRATION (agriBORA vs KAMIS)")

    kamis_white = kamis[kamis["Commodity_Classification"] == "Dry_White_Maize"].copy()
    kamis_agg = (
        kamis_white.groupby(["County", "Year_Week"])["Wholesale"]
        .mean()
        .reset_index()
        .rename(columns={"Wholesale": "KAMIS_Wholesale"})
    )

    train_agg = (
        train[train["County"].isin(TARGET_COUNTIES)]
        .groupby(["County", "Year_Week"])[TARGET_COL]
        .mean()
        .reset_index()
        .rename(columns={TARGET_COL: "agriBORA_WholeSale"})
    )

    merged = train_agg.merge(kamis_agg, on=["County", "Year_Week"], how="inner")
    _flag("INFO", f"Matched (County, Year_Week) pairs agriBORA x KAMIS: {len(merged)}")

    if len(merged) < 5:
        _flag("WARN", "Too few matched pairs for calibration analysis — KAMIS counties may differ")
        return

    # Scatter
    fig, axes = plt.subplots(1, min(len(TARGET_COUNTIES), 5), figsize=(16, 4))
    for ax, county in zip(axes, TARGET_COUNTIES):
        sub = merged[merged["County"] == county]
        if len(sub) < 2:
            ax.set_title(f"{county}\n(no overlap)")
            continue
        ax.scatter(sub["KAMIS_Wholesale"], sub["agriBORA_WholeSale"], alpha=0.7)
        ax.plot([sub["KAMIS_Wholesale"].min(), sub["KAMIS_Wholesale"].max()],
                [sub["KAMIS_Wholesale"].min(), sub["KAMIS_Wholesale"].max()],
                "r--", linewidth=0.8, label="1:1")
        ax.set_xlabel("KAMIS Wholesale")
        ax.set_ylabel("agriBORA WholeSale")
        ax.set_title(county)
        ax.legend(fontsize=7)
    plt.suptitle("agriBORA vs KAMIS Wholesale (matched weeks)", fontsize=11)
    plt.tight_layout()
    _savefig("2a_calibration_scatter.png")

    # Ratio per county
    merged["ratio"] = merged["agriBORA_WholeSale"] / merged["KAMIS_Wholesale"]
    _flag("INFO", "\nagriBORA / KAMIS ratio per county:")
    for county in TARGET_COUNTIES:
        sub = merged[merged["County"] == county]["ratio"]
        if len(sub) == 0:
            _flag("WARN", f"  {county}: no overlap")
        else:
            level = "WARN" if sub.std() > 0.05 else "INFO"
            _flag(level, f"  {county}: mean={sub.mean():.3f}, std={sub.std():.3f} "
                  f"({'unstable' if sub.std() > 0.05 else 'stable'})")

    # KAMIS commodity type comparison
    _flag("INFO", "\nKAMIS commodity type wholesale comparison:")
    for ctype in ["Dry_White_Maize", "Dry_Maize_Mixed_Traditional", "Dry_Yellow_Maize"]:
        sub = kamis[kamis["Commodity_Classification"] == ctype]["Wholesale"].dropna()
        if len(sub) > 0:
            _flag("INFO", f"  {ctype}: mean={sub.mean():.2f}, median={sub.median():.2f}, n={len(sub)}")


# ── Section 3: Sparsity & Coverage ────────────────────────────────────────────


def check_sparsity(train: pd.DataFrame, rolling: pd.DataFrame, kamis: pd.DataFrame) -> None:
    _section("3. SPARSITY & COVERAGE")

    # agriBORA heatmap (5 target counties)
    train_target = train[train["County"].isin(TARGET_COUNTIES)].copy()
    train_target["YearWeek_int"] = train_target["Year_Week"].str.replace("-", "").astype(int)
    pivot = train_target.groupby(["County", "Year_Week"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(TARGET_COUNTIES)

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) // 3), 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="Blues")
    ax.set_yticks(range(len(TARGET_COUNTIES)))
    ax.set_yticklabels(TARGET_COUNTIES)
    ax.set_xticks(range(0, len(pivot.columns), max(1, len(pivot.columns) // 15)))
    ax.set_xticklabels(list(pivot.columns)[::max(1, len(pivot.columns) // 15)], rotation=45, ha="right", fontsize=7)
    ax.set_title("agriBORA obs count: County × Year_Week (train)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    _savefig("3a_agribora_heatmap.png")

    # KAMIS coverage for target counties
    kamis_target = kamis[kamis["County"].isin(TARGET_COUNTIES)]
    _flag("INFO", f"\nKAMIS obs for target counties: {len(kamis_target)}")
    kamis_coverage = kamis_target.groupby("County")["Year_Week"].nunique().reindex(TARGET_COUNTIES, fill_value=0)
    for county, n in kamis_coverage.items():
        level = "WARN" if n < 10 else "INFO"
        _flag(level, f"  {county}: {n} weeks in KAMIS")

    # KAMIS market count per county
    _flag("INFO", "\nKAMIS distinct markets per target county:")
    market_counts = kamis_target.groupby("County")["Market"].nunique().reindex(TARGET_COUNTIES, fill_value=0)
    for county, n in market_counts.items():
        _flag("INFO", f"  {county}: {n} markets")


# ── Section 4: Seasonality & Harvest Calendar ─────────────────────────────────


def check_seasonality(train: pd.DataFrame, kamis: pd.DataFrame) -> None:
    _section("4. SEASONALITY & HARVEST CALENDAR")

    # KAMIS white maize aggregated by week for target counties
    kamis_white = kamis[
        (kamis["Commodity_Classification"] == "Dry_White_Maize") &
        (kamis["County"].isin(TARGET_COUNTIES))
    ].groupby(["County", "WeekofYear"])["Wholesale"].median().reset_index()

    agribora_woy = train[train["County"].isin(TARGET_COUNTIES)].groupby(
        ["County", "WeekofYear"])[TARGET_COL].median().reset_index()

    # Combine for richer seasonality
    combined = pd.concat([
        agribora_woy.rename(columns={TARGET_COL: "price"}),
        kamis_white.rename(columns={"Wholesale": "price"})
    ], ignore_index=True)
    seasonality = combined.groupby(["County", "WeekofYear"])["price"].median().reset_index()

    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
    harvest_weeks = [7, 8, 28, 29, 30, 48, 49, 50, 51]  # long rains Jul-Aug + short rains Dec-Jan
    for ax, county in zip(axes, TARGET_COUNTIES):
        sub = seasonality[seasonality["County"] == county].sort_values("WeekofYear")
        if len(sub) < 3:
            ax.set_title(f"{county} — insufficient data")
            continue
        ax.plot(sub["WeekofYear"], sub["price"], marker="o", markersize=4, linewidth=1.5)
        for wk in harvest_weeks:
            ax.axvline(wk, color="green", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_ylabel("KES/kg")
        ax.set_title(f"{county} — median price by week-of-year (agriBORA + KAMIS)")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Week of Year")
    fig.text(0.01, 0.5, "Green dashes = harvest periods", va="center", rotation="vertical", fontsize=8)
    plt.tight_layout()
    _savefig("4a_seasonality_by_county.png")

    # Producer vs consumer comparison
    _flag("INFO", "\nSeasonality — Uasin-Gishu (producer) vs Nairobi (consumer):")
    for county in ["Uasin-Gishu", "Nairobi"]:
        sub = seasonality[seasonality["County"] == county]
        if len(sub) > 0:
            peak_wk = sub.loc[sub["price"].idxmax(), "WeekofYear"]
            trough_wk = sub.loc[sub["price"].idxmin(), "WeekofYear"]
            _flag("INFO", f"  {county}: peak wk={peak_wk}, trough wk={trough_wk}, "
                  f"range={sub['price'].max()-sub['price'].min():.1f} KES/kg")


# ── Section 5: Cross-county Dynamics ──────────────────────────────────────────


def check_cross_county(train: pd.DataFrame, rolling: pd.DataFrame) -> None:
    _section("5. CROSS-COUNTY DYNAMICS")

    full = pd.concat([train, rolling], ignore_index=True)
    full[TIME_COL] = pd.to_datetime(full[TIME_COL])

    # Pivot: one column per county, aligned by date
    pivot = full[full["County"].isin(TARGET_COUNTIES)].pivot_table(
        index="Year_Week", columns="County", values=TARGET_COL, aggfunc="mean"
    ).reindex(columns=TARGET_COUNTIES)

    _flag("INFO", f"County pivot shape (weeks × counties): {pivot.shape}")
    _flag("INFO", f"Non-null counts per county: {pivot.notna().sum().to_dict()}")

    # Correlation matrix
    corr = pivot.corr()
    _flag("INFO", "\nCorrelation matrix (WholeSale, weekly aligned):")
    print(corr.round(3).to_string())

    mombasa_corr = corr["Mombasa"].drop("Mombasa")
    if mombasa_corr.max() < 0.5:
        _flag("WARN", f"Mombasa max correlation with other counties: {mombasa_corr.max():.3f} — treat as separate model")
    else:
        _flag("INFO", f"Mombasa max correlation: {mombasa_corr.max():.3f}")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(TARGET_COUNTIES)))
    ax.set_yticks(range(len(TARGET_COUNTIES)))
    ax.set_xticklabels(TARGET_COUNTIES, rotation=45, ha="right")
    ax.set_yticklabels(TARGET_COUNTIES)
    for i in range(len(TARGET_COUNTIES)):
        for j in range(len(TARGET_COUNTIES)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax)
    ax.set_title("Cross-county correlation (WholeSale)")
    plt.tight_layout()
    _savefig("5a_cross_county_corr.png")

    # Price spread: Uasin-Gishu vs Nairobi over time
    spread = pivot["Nairobi"] - pivot["Uasin-Gishu"]
    _flag("INFO", f"\nNairobi - Uasin-Gishu spread: mean={spread.mean():.2f}, std={spread.std():.2f} KES/kg")
    if spread.std() > 3:
        _flag("WARN", "Spread is volatile — producer-consumer gap not stable enough to use as a fixed offset")
    else:
        _flag("INFO", "Spread relatively stable — can model spread directly")

    # Granger causality (Uasin-Gishu -> Nairobi, lag 1-2)
    granger_data = pivot[["Uasin-Gishu", "Nairobi"]].dropna()
    if len(granger_data) >= 10:
        _flag("INFO", "\nGranger causality: Uasin-Gishu -> Nairobi (lags 1-2):")
        try:
            results = grangercausalitytests(granger_data[["Nairobi", "Uasin-Gishu"]], maxlag=2, verbose=False)
            for lag, res in results.items():
                p = res[0]["ssr_ftest"][1]
                _flag("INFO", f"  lag {lag}: p={p:.4f} ({'significant' if p < 0.05 else 'not significant'})")
        except Exception as e:
            _flag("WARN", f"Granger test failed: {e}")

    # Mombasa R² vs other counties
    _flag("INFO", "\nMombasa R² vs each county:")
    for county in ["Kiambu", "Kirinyaga", "Nairobi", "Uasin-Gishu"]:
        sub = pivot[["Mombasa", county]].dropna()
        if len(sub) < 5:
            _flag("WARN", f"  vs {county}: too few overlap points")
            continue
        r2 = np.corrcoef(sub["Mombasa"], sub[county])[0, 1] ** 2
        level = "WARN" if r2 < 0.3 else "INFO"
        _flag(level, f"  vs {county}: R²={r2:.3f}")


# ── Section 6: KAMIS as Leading Indicator ─────────────────────────────────────


def check_kamis_leading(train: pd.DataFrame, kamis: pd.DataFrame) -> None:
    _section("6. KAMIS AS LEADING INDICATOR")

    kamis_white = kamis[
        (kamis["Commodity_Classification"] == "Dry_White_Maize") &
        (kamis["County"].isin(TARGET_COUNTIES))
    ].copy()

    # SupplyVolume nulls
    sv_nulls = kamis_white["SupplyVolume"].isna().sum()
    sv_total = len(kamis_white)
    _flag("INFO" if sv_nulls / sv_total < 0.05 else "WARN",
          f"KAMIS SupplyVolume nulls: {sv_nulls}/{sv_total} ({sv_nulls/sv_total*100:.1f}%)")

    kamis_agg = kamis_white.groupby(["County", "WeekofYear"]).agg(
        SupplyVolume=("SupplyVolume", "sum"),
        Retail=("Retail", "mean"),
        Wholesale=("Wholesale", "mean")
    ).reset_index()

    # Cross-correlation SupplyVolume vs agriBORA WholeSale per county
    _flag("INFO", "\nCross-correlation KAMIS SupplyVolume vs agriBORA WholeSale (lags 0-4):")
    for county in TARGET_COUNTIES:
        df_a = train[train["County"] == county].groupby("WeekofYear")[TARGET_COL].mean()
        df_k = kamis_agg[kamis_agg["County"] == county].set_index("WeekofYear")["SupplyVolume"]
        common = df_a.index.intersection(df_k.index)
        if len(common) < 6:
            _flag("WARN", f"  {county}: too few overlap points ({len(common)}) for cross-corr")
            continue
        a_vals = df_a[common].values
        k_vals = df_k[common].values
        corrs = [np.corrcoef(k_vals[:len(k_vals)-lag if lag > 0 else len(k_vals)],
                             a_vals[lag:])[0, 1] if lag > 0
                 else np.corrcoef(k_vals, a_vals)[0, 1]
                 for lag in range(5)]
        best_lag = int(np.argmax(np.abs(corrs)))
        _flag("INFO", f"  {county}: corrs={[round(c, 2) for c in corrs]}, best_lag={best_lag}")

    # KAMIS Retail vs agriBORA WholeSale correlation
    _flag("INFO", "\nKAMIS Retail vs agriBORA WholeSale correlation per county:")
    for county in TARGET_COUNTIES:
        df_a = train[train["County"] == county].groupby("WeekofYear")[TARGET_COL].mean()
        df_k = kamis_agg[kamis_agg["County"] == county].set_index("WeekofYear")["Retail"]
        common = df_a.index.intersection(df_k.index)
        if len(common) < 5:
            _flag("WARN", f"  {county}: too few overlap ({len(common)})")
            continue
        r = np.corrcoef(df_a[common].values, df_k[common].values)[0, 1]
        _flag("INFO", f"  {county}: r={r:.3f}")


# ── Section 7: Baseline Benchmark ─────────────────────────────────────────────


def check_baseline(train: pd.DataFrame, rolling: pd.DataFrame) -> None:
    _section("7. BASELINE BENCHMARK")

    # Rolling wk46-51 has 6 weeks per county — use as validation
    # Simulate: predict wk N from wk N-1 (naive last value), and from wk N-2 (2-step naive)
    rolling_sorted = rolling[rolling["County"].isin(TARGET_COUNTIES)].sort_values([TIME_COL, "County"])

    def competition_score(actual: np.ndarray, pred: np.ndarray) -> float:
        mae = np.mean(np.abs(actual - pred))
        rmse = np.sqrt(np.mean((actual - pred) ** 2))
        return 0.5 * mae + 0.5 * rmse

    _flag("INFO", "\nNaive last-value (predict wk N from wk N-1):")
    results = {}
    for county in TARGET_COUNTIES:
        df_c = rolling_sorted[rolling_sorted["County"] == county].sort_values(TIME_COL)
        if len(df_c) < 2:
            continue
        actual = df_c[TARGET_COL].values[1:]
        pred_naive = df_c[TARGET_COL].values[:-1]
        score = competition_score(actual, pred_naive)
        results[county] = score
        _flag("INFO", f"  {county}: score={score:.3f} (MAE={np.mean(np.abs(actual-pred_naive)):.3f}, "
              f"RMSE={np.sqrt(np.mean((actual-pred_naive)**2)):.3f})")

    if results:
        _flag("INFO", f"  MEAN across counties: {np.mean(list(results.values())):.3f}")

    # 2-week-ahead naive (predict wk N+2 from wk N)
    _flag("INFO", "\n2-week-ahead naive (predict wk N+2 from wk N) — actual submission horizon:")
    results_2 = {}
    for county in TARGET_COUNTIES:
        df_c = rolling_sorted[rolling_sorted["County"] == county].sort_values(TIME_COL)
        if len(df_c) < 3:
            continue
        actual = df_c[TARGET_COL].values[2:]
        pred_naive2 = df_c[TARGET_COL].values[:-2]
        score = competition_score(actual, pred_naive2)
        results_2[county] = score
        _flag("INFO", f"  {county}: score={score:.3f}")
    if results_2:
        _flag("INFO", f"  MEAN: {np.mean(list(results_2.values())):.3f}")

    # Last-4-weeks mean
    _flag("INFO", "\n4-week rolling mean forecast:")
    results_4 = {}
    for county in TARGET_COUNTIES:
        # Use train to build rolling mean, eval on first 2 wk46-47
        df_train_c = train[train["County"] == county].sort_values(TIME_COL)
        df_roll_c = rolling_sorted[rolling_sorted["County"] == county].sort_values(TIME_COL)
        if len(df_train_c) < 4 or len(df_roll_c) < 2:
            continue
        last4_mean = df_train_c[TARGET_COL].tail(4).mean()
        actual = df_roll_c[TARGET_COL].values[:2]
        pred = np.full(len(actual), last4_mean)
        score = competition_score(actual, pred)
        results_4[county] = score
        _flag("INFO", f"  {county}: score={score:.3f} (last-4-mean={last4_mean:.2f})")
    if results_4:
        _flag("INFO", f"  MEAN: {np.mean(list(results_4.values())):.3f}")

    # Summary bar chart
    if results and results_2:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(TARGET_COUNTIES))
        w = 0.3
        scores_1 = [results.get(c, np.nan) for c in TARGET_COUNTIES]
        scores_2 = [results_2.get(c, np.nan) for c in TARGET_COUNTIES]
        scores_4 = [results_4.get(c, np.nan) for c in TARGET_COUNTIES]
        ax.bar(x - w, scores_1, w, label="naive 1-step", color="steelblue")
        ax.bar(x, scores_2, w, label="naive 2-step", color="darkorange")
        ax.bar(x + w, scores_4, w, label="last-4-mean", color="green")
        ax.set_xticks(x)
        ax.set_xticklabels(TARGET_COUNTIES, rotation=20, ha="right")
        ax.set_ylabel("0.5×MAE + 0.5×RMSE")
        ax.set_title("Baseline benchmark per county (wk46-51 validation)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        _savefig("7a_baseline_scores.png")


# ── Section 8: ADF Stationarity Test ─────────────────────────────────────────


def check_stationarity(train: pd.DataFrame, rolling: pd.DataFrame) -> None:
    _section("8. ADF STATIONARITY TEST")

    full = pd.concat([train, rolling], ignore_index=True)
    full[TIME_COL] = pd.to_datetime(full[TIME_COL])

    _flag("INFO", "ADF test: H0 = unit root (non-stationary). p < 0.05 = stationary.")
    for county in TARGET_COUNTIES:
        series = (
            full[full["County"] == county]
            .sort_values(TIME_COL)[TARGET_COL]
            .dropna()
            .values
        )
        if len(series) < 10:
            _flag("WARN", f"  {county}: too few obs ({len(series)}) for ADF")
            continue
        # Levels
        adf_stat, p_level, _, _, _, _ = adfuller(series, autolag="AIC")
        level_label = "stationary" if p_level < 0.05 else "NON-STATIONARY"
        # First difference
        diff = np.diff(series)
        adf_stat_d, p_diff, _, _, _, _ = adfuller(diff, autolag="AIC")
        diff_label = "stationary" if p_diff < 0.05 else "NON-STATIONARY"
        flag_level = "WARN" if p_level >= 0.05 else "INFO"
        _flag(flag_level, f"  {county}: levels p={p_level:.4f} ({level_label}) | "
              f"diff p={p_diff:.4f} ({diff_label})")


# ── Section 9: Rolling Volatility ─────────────────────────────────────────────


def check_rolling_volatility(train: pd.DataFrame, rolling: pd.DataFrame) -> None:
    _section("9. ROLLING VOLATILITY (4-week window)")

    full = pd.concat([train, rolling], ignore_index=True)
    full[TIME_COL] = pd.to_datetime(full[TIME_COL])

    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
    colors = ["steelblue", "darkorange", "green", "red", "purple"]

    for ax, county, color in zip(axes, TARGET_COUNTIES, colors):
        df_c = (
            full[full["County"] == county]
            .sort_values(TIME_COL)
            .set_index(TIME_COL)[TARGET_COL]
            .dropna()
        )
        if len(df_c) < 6:
            ax.set_title(f"{county} — insufficient data")
            continue
        roll_mean = df_c.rolling(4, min_periods=2).mean()
        roll_std = df_c.rolling(4, min_periods=2).std()
        ax.plot(df_c.index, df_c.values, alpha=0.4, color=color, linewidth=0.8, label="price")
        ax.plot(roll_mean.index, roll_mean.values, color=color, linewidth=1.5, label="4w mean")
        ax.fill_between(
            roll_mean.index,
            (roll_mean - roll_std).values,
            (roll_mean + roll_std).values,
            alpha=0.2, color=color
        )
        ax.set_ylabel("KES/kg")
        ax.set_title(f"{county} — 4-week rolling mean +/- std")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
        # Print max std period
        if roll_std.notna().any():
            max_std_date = roll_std.idxmax()
            _flag("INFO", f"  {county}: max rolling std={roll_std.max():.2f} at {max_std_date.date()}")

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    _savefig("9a_rolling_volatility.png")


# ── Section 10: WoW Change Distribution ───────────────────────────────────────


def check_wow_distribution(train: pd.DataFrame, rolling: pd.DataFrame) -> None:
    _section("10. WoW PRICE CHANGE DISTRIBUTION (differenced series)")

    full = pd.concat([train, rolling], ignore_index=True)
    full[TIME_COL] = pd.to_datetime(full[TIME_COL])

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for ax, county in zip(axes, TARGET_COUNTIES):
        df_c = (
            full[full["County"] == county]
            .sort_values(TIME_COL)[TARGET_COL]
            .dropna()
        )
        if len(df_c) < 5:
            ax.set_title(f"{county}\n(too few obs)")
            continue
        diff = df_c.diff().dropna()
        ax.hist(diff, bins=15, edgecolor="white", color="steelblue", alpha=0.8)
        ax.axvline(diff.mean(), color="red", linestyle="--", linewidth=1.2, label=f"mean={diff.mean():.2f}")
        ax.axvline(diff.median(), color="orange", linestyle="--", linewidth=1.2, label=f"med={diff.median():.2f}")
        ax.set_title(f"{county}")
        ax.legend(fontsize=7)
        ax.set_xlabel("KES/kg change")
        skew = diff.skew()
        _flag("INFO" if abs(skew) < 1 else "WARN",
              f"  {county}: mean={diff.mean():.2f}, std={diff.std():.2f}, "
              f"skew={skew:.2f}, p5/p95={diff.quantile(0.05):.2f}/{diff.quantile(0.95):.2f}")

    plt.suptitle("WoW price change distribution per county", fontsize=11)
    plt.tight_layout()
    _savefig("10a_wow_distribution.png")


# ── Section 11: AR(1) Residual Diagnostics ────────────────────────────────────


def check_ar1_residuals(train: pd.DataFrame, rolling: pd.DataFrame) -> None:
    _section("11. AR(1) RESIDUAL DIAGNOSTICS + LJUNG-BOX")

    full = pd.concat([train, rolling], ignore_index=True)
    full[TIME_COL] = pd.to_datetime(full[TIME_COL])

    _flag("INFO", "Fitting AR(1): price_t = c + b*price_t-1 + e_t")
    residuals_dict: dict[str, np.ndarray] = {}

    for county in TARGET_COUNTIES:
        series = (
            full[full["County"] == county]
            .sort_values(TIME_COL)[TARGET_COL]
            .dropna()
            .values
        )
        if len(series) < 8:
            _flag("WARN", f"  {county}: too few obs for AR(1)")
            continue
        y = series[1:]
        x = series[:-1]
        # OLS AR(1)
        b = np.cov(x, y)[0, 1] / np.var(x)
        c = y.mean() - b * x.mean()
        pred = c + b * x
        resid = y - pred
        residuals_dict[county] = resid
        resid_std = resid.std()
        # Ljung-Box on residuals
        lb_lags = min(5, len(resid) // 4)
        try:
            lb_result = acorr_ljungbox(resid, lags=[lb_lags], return_df=True)
            lb_p = lb_result["lb_pvalue"].values[0]
            lb_label = "white noise" if lb_p > 0.05 else "RESIDUAL STRUCTURE REMAINS"
            flag_level = "INFO" if lb_p > 0.05 else "WARN"
        except Exception:
            lb_p, lb_label, flag_level = float("nan"), "LB failed", "WARN"
        _flag(flag_level, f"  {county}: AR(1) b={b:.3f}, c={c:.2f}, "
              f"resid_std={resid_std:.2f} | Ljung-Box(lag={lb_lags}) p={lb_p:.4f} -> {lb_label}")

    # Cross-county residual correlations
    if len(residuals_dict) >= 2:
        _flag("INFO", "\nCross-county AR(1) residual correlations:")
        counties_with_resid = list(residuals_dict.keys())
        for i, c1 in enumerate(counties_with_resid):
            for c2 in counties_with_resid[i + 1:]:
                r1, r2 = residuals_dict[c1], residuals_dict[c2]
                n = min(len(r1), len(r2))
                if n < 5:
                    continue
                r = np.corrcoef(r1[:n], r2[:n])[0, 1]
                flag_level = "WARN" if abs(r) > 0.4 else "INFO"
                _flag(flag_level, f"  {c1} vs {c2}: r={r:.3f}"
                      + (" — shared shock, add cross-county feature" if abs(r) > 0.4 else ""))


# ── Section 12: KAMIS County Name Audit ───────────────────────────────────────


def check_kamis_county_names(train: pd.DataFrame, kamis: pd.DataFrame) -> None:
    _section("12. KAMIS COUNTY NAME AUDIT")

    agribora_counties = set(train["County"].unique())
    kamis_counties = set(kamis["County"].unique())

    _flag("INFO", f"agriBORA counties ({len(agribora_counties)}): {sorted(agribora_counties)}")
    _flag("INFO", f"KAMIS counties ({len(kamis_counties)}): {sorted(kamis_counties)}")

    exact_match = agribora_counties & kamis_counties
    _flag("INFO", f"Exact matches: {sorted(exact_match)}")

    no_match = agribora_counties - kamis_counties
    _flag("WARN" if no_match else "INFO", f"agriBORA counties NOT in KAMIS: {sorted(no_match)}")

    # Fuzzy match: check if any KAMIS county contains the unmatched agriBORA county name
    _flag("INFO", "\nFuzzy search for unmatched counties in KAMIS:")
    for county in sorted(no_match):
        candidates = [k for k in kamis_counties if county.lower() in k.lower() or k.lower() in county.lower()]
        if candidates:
            _flag("WARN", f"  '{county}' -> possible KAMIS match: {candidates}")
            # Show volume in KAMIS for those candidates
            for cand in candidates:
                n = len(kamis[kamis["County"] == cand])
                _flag("INFO", f"    '{cand}': {n} KAMIS rows")
        else:
            _flag("WARN", f"  '{county}' -> no fuzzy match found in KAMIS")

    # KAMIS Dry_White_Maize coverage for target counties (including fuzzy names)
    _flag("INFO", "\nKAMIS Dry_White_Maize rows per target county (exact match):")
    kamis_white = kamis[kamis["Commodity_Classification"] == "Dry_White_Maize"]
    for county in TARGET_COUNTIES:
        n = len(kamis_white[kamis_white["County"] == county])
        flag_level = "WARN" if n == 0 else "INFO"
        _flag(flag_level, f"  {county}: {n} rows")


# ── Section 13: KAMIS SupplyVolume Null Pattern ────────────────────────────────


def check_supply_null_pattern(kamis: pd.DataFrame) -> None:
    _section("13. KAMIS SUPPLY VOLUME NULL PATTERN")

    kamis_white = kamis[
        (kamis["Commodity_Classification"] == "Dry_White_Maize") &
        (kamis["County"].isin(TARGET_COUNTIES))
    ].copy()

    if len(kamis_white) == 0:
        _flag("WARN", "No KAMIS Dry_White_Maize rows for target counties — skip")
        return

    # Null rate per county
    _flag("INFO", "SupplyVolume null rate per target county (KAMIS Dry_White_Maize):")
    for county in TARGET_COUNTIES:
        sub = kamis_white[kamis_white["County"] == county]["SupplyVolume"]
        if len(sub) == 0:
            _flag("WARN", f"  {county}: 0 rows")
            continue
        null_rate = sub.isna().mean()
        flag_level = "CRITICAL" if null_rate > 0.3 else "WARN" if null_rate > 0.1 else "INFO"
        _flag(flag_level, f"  {county}: {null_rate*100:.1f}% nulls ({sub.isna().sum()}/{len(sub)})")

    # Null heatmap: County × Year_Week
    try:
        pivot_null = kamis_white.groupby(["County", "Year_Week"])["SupplyVolume"].apply(
            lambda x: x.isna().mean()
        ).unstack(fill_value=0)
        pivot_null = pivot_null.reindex([c for c in TARGET_COUNTIES if c in pivot_null.index])
        if pivot_null.shape[1] > 5:
            fig, ax = plt.subplots(figsize=(max(10, pivot_null.shape[1] // 3), 3))
            im = ax.imshow(pivot_null.values, aspect="auto", cmap="Reds", vmin=0, vmax=1)
            ax.set_yticks(range(len(pivot_null)))
            ax.set_yticklabels(pivot_null.index.tolist())
            n_cols = pivot_null.shape[1]
            step = max(1, n_cols // 15)
            ax.set_xticks(range(0, n_cols, step))
            ax.set_xticklabels(list(pivot_null.columns)[::step], rotation=45, ha="right", fontsize=7)
            ax.set_title("KAMIS SupplyVolume null rate: County × Year_Week (1=all null)")
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            _savefig("13a_supply_null_heatmap.png")
    except Exception as e:
        _flag("WARN", f"Supply null heatmap failed: {e}")

    # National aggregate SupplyVolume (sum across non-null counties) per week
    national = (
        kamis_white.groupby("Year_Week")["SupplyVolume"]
        .sum()
        .reset_index()
        .rename(columns={"SupplyVolume": "national_supply"})
    )
    _flag("INFO", f"\nNational weekly supply (sum across counties): "
          f"mean={national['national_supply'].mean():.0f}, "
          f"std={national['national_supply'].std():.0f}, "
          f"weeks with data={national['national_supply'].gt(0).sum()}/{len(national)}")


# ── Section 14: Last-Known-Value Recency Check ────────────────────────────────


def check_last_value_recency(train: pd.DataFrame, rolling: pd.DataFrame) -> None:
    _section("14. LAST-KNOWN-VALUE RECENCY BEFORE FORECAST WINDOW (wk52)")

    full = pd.concat([train, rolling], ignore_index=True)
    full[TIME_COL] = pd.to_datetime(full[TIME_COL])

    # Forecast starts at wk52 2025; rolling ends at wk51 2025
    forecast_start = pd.Timestamp("2025-12-22")  # approx wk52 2025

    _flag("INFO", f"Forecast window starts: {forecast_start.date()} (approx wk52 2025)")
    _flag("INFO", "Checking most recent obs per county before forecast start:")
    for county in TARGET_COUNTIES:
        df_c = full[(full["County"] == county) & (full[TIME_COL] < forecast_start)].sort_values(TIME_COL)
        if len(df_c) == 0:
            _flag("CRITICAL", f"  {county}: NO observations before forecast window")
            continue
        last_obs_date = df_c[TIME_COL].max()
        last_price = df_c.loc[df_c[TIME_COL] == last_obs_date, TARGET_COL].mean()
        weeks_stale = (forecast_start - last_obs_date).days / 7
        flag_level = "CRITICAL" if weeks_stale > 4 else "WARN" if weeks_stale > 2 else "INFO"
        _flag(flag_level, f"  {county}: last obs={last_obs_date.date()}, "
              f"price={last_price:.2f}, {weeks_stale:.1f} weeks before forecast")


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 60)
    print("EDA START — agriBORA Commodity Price Forecasting")
    print("=" * 60)

    train, rolling, kamis = load_data()

    check_time_series_structure(train, rolling)
    check_calibration(train, kamis)
    check_sparsity(train, rolling, kamis)
    check_seasonality(train, kamis)
    check_cross_county(train, rolling)
    check_kamis_leading(train, kamis)
    check_baseline(train, rolling)
    # Opus review additions
    check_stationarity(train, rolling)
    check_rolling_volatility(train, rolling)
    check_wow_distribution(train, rolling)
    check_ar1_residuals(train, rolling)
    check_kamis_county_names(train, kamis)
    check_supply_null_pattern(kamis)
    check_last_value_recency(train, rolling)

    print("\n" + "=" * 60)
    print("EDA DONE — check outputs/eda/ for plots")
    print("=" * 60)


if __name__ == "__main__":
    main()
