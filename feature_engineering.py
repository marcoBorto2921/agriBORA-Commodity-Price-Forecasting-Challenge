"""
Feature Engineering — agriBORA Commodity Price Forecasting
Run: .venv/Scripts/python feature_engineering.py
Outputs: outputs/features/v1/train_fe.parquet, test_fe.parquet
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────

DATA_DIR = "agribora-commodity-price-forecasting-challenge20260114-30357-1miwudk"
TRAIN_PATH = f"{DATA_DIR}/agriBORA_maize_prices.csv"
ROLLING_PATH = f"{DATA_DIR}/agriBORA_maize_prices_weeks_46_to_51.csv"
KAMIS_PATH = f"{DATA_DIR}/kamis_maize_prices.csv"

TARGET_COL = "WholeSale"
TARGET_COUNTIES = ["Kiambu", "Kirinyaga", "Mombasa", "Nairobi", "Uasin-Gishu"]

# Per-county KAMIS→agriBORA calibration ratio (from EDA: agriBORA / KAMIS)
CALIBRATION_RATIOS: dict[str, float] = {
    "Nairobi": 0.805,
    "Uasin-Gishu": 0.818,
    "Kirinyaga": 0.984,
    # Kiambu and Mombasa have no agriBORA-KAMIS overlap → use mean of known ratios
    "Kiambu": 0.869,
    "Mombasa": 0.869,
}

# Lean season weeks (Nov–early Feb)
LEAN_SEASON_WEEKS = set(range(48, 53)) | set(range(1, 7))
# Short rains harvest window (Dec–Jan)
SHORT_HARVEST_WEEKS = set(range(49, 53)) | set(range(1, 3))

OUTPUT_DIR = Path("outputs/features/v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────


def _flag(level: str, msg: str) -> None:
    print(f"[{level}] {msg}")


def _weeks_to_long_harvest(woy: int) -> int:
    """Cyclic distance in weeks to week 27 (long rains harvest start, Uasin-Gishu)."""
    dist = abs(woy - 27)
    return min(dist, 52 - dist)


# ── load ──────────────────────────────────────────────────────────────────────


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all raw data files."""
    train = pd.read_csv(TRAIN_PATH, encoding="utf-8")
    train["Date"] = pd.to_datetime(train["Date"])

    rolling = pd.read_csv(ROLLING_PATH, encoding="utf-8")
    rolling["Date"] = pd.to_datetime(rolling["Date"])

    kamis = pd.read_csv(KAMIS_PATH, encoding="utf-8")
    kamis["Date"] = pd.to_datetime(kamis["Date"])

    _flag(
        "INFO",
        f"Raw — agriBORA train: {train.shape}, rolling: {rolling.shape}, KAMIS: {kamis.shape}",
    )
    return train, rolling, kamis


# ── Step 1: Build weekly agriBORA panel ───────────────────────────────────────


def build_agribora_panel(train: pd.DataFrame, rolling: pd.DataFrame) -> pd.DataFrame:
    """
    Merge train + rolling, filter to target counties, aggregate to weekly median.
    Returns a panel indexed by (County, Year_Week) sorted chronologically.
    """
    full = pd.concat([train, rolling], ignore_index=True)
    full["Date"] = pd.to_datetime(full["Date"])
    full = full[full["County"].isin(TARGET_COUNTIES)].copy()

    # Aggregate: multiple transactions per (County, Year_Week) → median
    panel = (
        full.groupby(["County", "Year_Week", "WeekofYear"])
        .agg(WholeSale=(TARGET_COL, "median"), Date=("Date", "min"))
        .reset_index()
        .sort_values(["County", "Year_Week"])
        .reset_index(drop=True)
    )

    _flag(
        "INFO",
        f"agriBORA panel: {panel.shape} rows ({panel['Year_Week'].nunique()} weeks × 5 counties approx)",
    )
    return panel


# ── Step 2: KAMIS augmentation for sparse counties ────────────────────────────


def build_kamis_augmentation(
    kamis: pd.DataFrame, agribora_panel: pd.DataFrame
) -> pd.DataFrame:
    """
    For Kiambu and Mombasa (most sparse), create synthetic agriBORA-scale rows
    from KAMIS Dry_White_Maize using per-county calibration ratios.
    Only adds rows for (County, Year_Week) not already in agriBORA panel.
    """
    kamis_white = kamis[kamis["Commodity_Classification"] == "Dry_White_Maize"].copy()

    kamis_county = (
        kamis_white[kamis_white["County"].isin(["Kiambu", "Mombasa"])]
        .groupby(["County", "Year_Week", "WeekofYear"])
        .agg(KAMIS_Wholesale=("Wholesale", "median"), Date=("Date", "min"))
        .reset_index()
    )

    if len(kamis_county) == 0:
        _flag("WARN", "No KAMIS rows for Kiambu/Mombasa — skipping augmentation")
        return agribora_panel

    # Convert to agriBORA scale
    kamis_county["WholeSale"] = kamis_county.apply(
        lambda r: r["KAMIS_Wholesale"] * CALIBRATION_RATIOS[r["County"]], axis=1
    )
    kamis_county["is_kamis_augmented"] = 1

    # Keep only rows not already covered by agriBORA
    existing = set(zip(agribora_panel["County"], agribora_panel["Year_Week"]))
    new_rows = kamis_county[
        ~kamis_county.apply(lambda r: (r["County"], r["Year_Week"]) in existing, axis=1)
    ][
        ["County", "Year_Week", "WeekofYear", "WholeSale", "Date", "is_kamis_augmented"]
    ].copy()

    _flag("INFO", f"KAMIS augmentation: added {len(new_rows)} rows for Kiambu/Mombasa")

    agribora_panel["is_kamis_augmented"] = 0
    result = pd.concat([agribora_panel, new_rows], ignore_index=True)
    result = result.sort_values(["County", "Year_Week"]).reset_index(drop=True)
    return result


# ── Step 3: KAMIS feature table ───────────────────────────────────────────────


def build_kamis_features(kamis: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate KAMIS Dry_White_Maize to weekly per-county features.
    Returns a table keyed on (County, Year_Week) with KAMIS wholesale, retail, UG supply.
    """
    kamis_white = kamis[kamis["Commodity_Classification"] == "Dry_White_Maize"].copy()

    # Per-county weekly aggregation
    kamis_agg = (
        kamis_white[kamis_white["County"].isin(TARGET_COUNTIES)]
        .groupby(["County", "Year_Week"])
        .agg(
            kamis_wholesale=("Wholesale", "median"),
            kamis_retail=("Retail", "median"),
        )
        .reset_index()
    )

    # Uasin-Gishu supply volume (0.2% nulls — near complete)
    ug_supply = (
        kamis_white[kamis_white["County"] == "Uasin-Gishu"]
        .groupby("Year_Week")["SupplyVolume"]
        .sum()
        .reset_index()
        .rename(columns={"SupplyVolume": "kamis_ug_supply"})
    )

    # National supply aggregate (all counties, per week)
    national_supply = (
        kamis_white.groupby("Year_Week")["SupplyVolume"]
        .sum()
        .reset_index()
        .rename(columns={"SupplyVolume": "kamis_national_supply"})
    )

    _flag(
        "INFO",
        f"KAMIS features: {len(kamis_agg)} county-week rows, "
        f"UG supply {ug_supply['kamis_ug_supply'].gt(0).sum()} non-zero weeks, "
        f"national supply {national_supply['kamis_national_supply'].gt(0).sum()} non-zero weeks",
    )

    return kamis_agg, ug_supply, national_supply


# ── Step 4: Compute lag + rolling features ────────────────────────────────────


def add_time_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling, and calendar features per county. Sorted by Year_Week."""
    result_parts = []

    for county in TARGET_COUNTIES:
        df = panel[panel["County"] == county].sort_values("Year_Week").copy()

        # ── Lag features ──
        df["lag1_price"] = df["WholeSale"].shift(1)
        df["lag2_price"] = df["WholeSale"].shift(2)

        # ── Rolling 4-week mean (on non-null price) ──
        df["rolling_4w_mean"] = (
            df["WholeSale"].rolling(window=4, min_periods=2).mean().shift(1)
        )
        df["deviation_from_rolling_mean"] = df["lag1_price"] - df["rolling_4w_mean"]

        # ── Rolling 4-week max ──
        df["rolling_4w_max"] = (
            df["WholeSale"].rolling(window=4, min_periods=2).max().shift(1)
        )
        df["price_vs_4w_max"] = df["lag1_price"] - df["rolling_4w_max"]

        # ── Consecutive weeks of price increase ──
        price_diff = df["WholeSale"].diff()
        is_up = (price_diff > 0).astype(int)
        # Count consecutive ups ending at t-1 (lagged)
        consec = []
        count = 0
        for up in is_up:
            if up == 1:
                count += 1
            else:
                count = 0
            consec.append(count)
        df["consecutive_up_weeks"] = pd.Series(consec, index=df.index).shift(1)

        # ── Price momentum ──
        df["price_momentum"] = df["lag1_price"] - df["lag2_price"]

        result_parts.append(df)

    panel_out = pd.concat(result_parts, ignore_index=True)
    panel_out = panel_out.sort_values(["County", "Year_Week"]).reset_index(drop=True)
    return panel_out


def add_calendar_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add week-of-year based calendar and seasonal features."""
    panel = panel.copy()
    panel["week_of_year"] = panel["WeekofYear"].astype(int)
    panel["is_lean_season"] = panel["week_of_year"].isin(LEAN_SEASON_WEEKS).astype(int)
    panel["is_short_harvest_window"] = (
        panel["week_of_year"].isin(SHORT_HARVEST_WEEKS).astype(int)
    )
    panel["weeks_to_long_harvest"] = panel["week_of_year"].apply(_weeks_to_long_harvest)

    # Weeks already into lean season (0 outside lean season)
    def _weeks_into_lean(woy: int) -> int:
        if woy >= 48:
            return woy - 47
        elif woy <= 6:
            return woy + (52 - 47)
        return 0

    panel["weeks_in_lean_season"] = panel["week_of_year"].apply(_weeks_into_lean)
    return panel


# ── Step 5: Cross-county features ─────────────────────────────────────────────


def add_cross_county_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add lagged Nairobi and UG prices + national mean as features for all counties."""
    # Build pivot of lag1 per county per week
    lag1_pivot = (
        panel[["Year_Week", "County", "lag1_price"]]
        .pivot(index="Year_Week", columns="County", values="lag1_price")
        .reset_index()
    )
    lag1_pivot.columns.name = None

    nairobi_col = "Nairobi" if "Nairobi" in lag1_pivot.columns else None
    ug_col = "Uasin-Gishu" if "Uasin-Gishu" in lag1_pivot.columns else None

    # National mean lag1 across all 5 counties
    county_cols = [c for c in TARGET_COUNTIES if c in lag1_pivot.columns]
    lag1_pivot["national_mean_lag1"] = lag1_pivot[county_cols].mean(axis=1)

    if nairobi_col:
        lag1_pivot = lag1_pivot.rename(columns={"Nairobi": "nairobi_lag1"})
    if ug_col:
        lag1_pivot = lag1_pivot.rename(columns={"Uasin-Gishu": "ug_lag1"})

    # Nairobi-UG spread
    if "nairobi_lag1" in lag1_pivot.columns and "ug_lag1" in lag1_pivot.columns:
        lag1_pivot["nairobi_ug_spread"] = (
            lag1_pivot["nairobi_lag1"] - lag1_pivot["ug_lag1"]
        )

    # Drop individual county columns (keep only cross-county features)
    drop_cols = [
        c
        for c in TARGET_COUNTIES
        if c in lag1_pivot.columns and c not in ("nairobi_lag1", "ug_lag1")
    ]
    lag1_pivot = lag1_pivot.drop(columns=drop_cols, errors="ignore")

    panel = panel.merge(lag1_pivot, on="Year_Week", how="left")
    return panel


# ── Step 6: Merge KAMIS features ──────────────────────────────────────────────


def merge_kamis_features(
    panel: pd.DataFrame,
    kamis_agg: pd.DataFrame,
    ug_supply: pd.DataFrame,
    national_supply: pd.DataFrame,
) -> pd.DataFrame:
    """Merge KAMIS wholesale, retail (per county) and supply features (global), lagged 1 week."""

    # Build lagged KAMIS per county
    kamis_lagged_parts = []
    for county in TARGET_COUNTIES:
        k = kamis_agg[kamis_agg["County"] == county].sort_values("Year_Week").copy()
        k["kamis_wholesale_lag1"] = k["kamis_wholesale"].shift(1)
        k["kamis_retail_lag1"] = k["kamis_retail"].shift(1)
        kamis_lagged_parts.append(
            k[["County", "Year_Week", "kamis_wholesale_lag1", "kamis_retail_lag1"]]
        )

    kamis_county_lag = pd.concat(kamis_lagged_parts, ignore_index=True)

    # Lag supply features (global → all counties share same value)
    ug_supply_lag = ug_supply.sort_values("Year_Week").copy()
    ug_supply_lag["kamis_ug_supply_lag1"] = ug_supply_lag["kamis_ug_supply"].shift(1)
    ug_supply_lag = ug_supply_lag[["Year_Week", "kamis_ug_supply_lag1"]]

    national_supply_lag = national_supply.sort_values("Year_Week").copy()
    national_supply_lag["kamis_national_supply_lag1"] = national_supply_lag[
        "kamis_national_supply"
    ].shift(1)
    national_supply_lag = national_supply_lag[
        ["Year_Week", "kamis_national_supply_lag1"]
    ]

    panel = panel.merge(kamis_county_lag, on=["County", "Year_Week"], how="left")
    panel = panel.merge(ug_supply_lag, on="Year_Week", how="left")
    panel = panel.merge(national_supply_lag, on="Year_Week", how="left")

    return panel


# ── Step 7: Build test rows (wk52 + wk1) ─────────────────────────────────────


def build_test_rows(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature rows for wk52 and wk1 (no target).
    wk52: lag1 = wk51 (observed), fully computable.
    wk1:  lag1 = wk52 (prediction, unknown) → set to NaN, filled recursively at inference.
    """
    test_rows = []

    for county in TARGET_COUNTIES:
        df_c = panel[panel["County"] == county].sort_values("Year_Week")
        last_row = df_c.iloc[-1]

        # wk52 row
        tail4 = df_c["WholeSale"].dropna().tail(4)
        roll_mean = tail4.mean()
        roll_max = tail4.max()
        lag1 = last_row["WholeSale"]
        lag2 = df_c.iloc[-2]["WholeSale"] if len(df_c) >= 2 else np.nan
        # Count consecutive up weeks into wk51
        recent = df_c["WholeSale"].dropna().tail(6).values
        consec = 0
        for i in range(len(recent) - 1, 0, -1):
            if recent[i] > recent[i - 1]:
                consec += 1
            else:
                break

        wk52 = {
            "County": county,
            "Year_Week": "2025-52",
            "WeekofYear": 52,
            "WholeSale": np.nan,
            "is_kamis_augmented": 0,
            "lag1_price": lag1,
            "lag2_price": lag2,
            "rolling_4w_mean": roll_mean,
            "deviation_from_rolling_mean": lag1 - roll_mean
            if pd.notna(lag1)
            else np.nan,
            "rolling_4w_max": roll_max,
            "price_vs_4w_max": lag1 - roll_max if pd.notna(lag1) else np.nan,
            "price_momentum": lag1 - lag2
            if pd.notna(lag1) and pd.notna(lag2)
            else np.nan,
            "consecutive_up_weeks": consec,
        }
        test_rows.append(wk52)

        # wk1 row — lag1 = wk52 prediction (unknown, filled recursively at inference)
        wk1 = {
            "County": county,
            "Year_Week": "2026-01",
            "WeekofYear": 1,
            "WholeSale": np.nan,
            "is_kamis_augmented": 0,
            "lag1_price": np.nan,
            "lag2_price": lag1,
            "rolling_4w_mean": roll_mean,
            "deviation_from_rolling_mean": np.nan,
            "rolling_4w_max": roll_max,
            "price_vs_4w_max": np.nan,
            "price_momentum": np.nan,
            "consecutive_up_weeks": consec + 1,  # approximate: assume trend continues
        }
        test_rows.append(wk1)

    test = pd.DataFrame(test_rows)
    return test


def add_calendar_and_cross_county_to_test(
    test: pd.DataFrame,
    panel: pd.DataFrame,
    kamis_agg: pd.DataFrame,
    ug_supply: pd.DataFrame,
    national_supply: pd.DataFrame,
) -> pd.DataFrame:
    """Add calendar + cross-county + KAMIS features to test rows."""
    test = add_calendar_features(test)

    # Cross-county features for wk52: use wk51 prices (lag1 of each county in wk52)
    wk51_prices = {
        county: panel[panel["County"] == county]
        .sort_values("Year_Week")["WholeSale"]
        .iloc[-1]
        for county in TARGET_COUNTIES
    }
    national_mean = np.mean(list(wk51_prices.values()))
    nairobi_lag1 = wk51_prices.get("Nairobi", np.nan)
    ug_lag1 = wk51_prices.get("Uasin-Gishu", np.nan)

    test.loc[test["Year_Week"] == "2025-52", "nairobi_lag1"] = nairobi_lag1
    test.loc[test["Year_Week"] == "2025-52", "ug_lag1"] = ug_lag1
    test.loc[test["Year_Week"] == "2025-52", "national_mean_lag1"] = national_mean
    test.loc[test["Year_Week"] == "2025-52", "nairobi_ug_spread"] = (
        nairobi_lag1 - ug_lag1
    )

    # wk1: lag1 for cross-county = wk52 prediction (unknown) → NaN, filled at inference
    for col in ["nairobi_lag1", "ug_lag1", "national_mean_lag1", "nairobi_ug_spread"]:
        test.loc[test["Year_Week"] == "2026-01", col] = np.nan

    # KAMIS features: KAMIS ends Jul 2025 — all null for wk52/wk1
    for col in [
        "kamis_wholesale_lag1",
        "kamis_retail_lag1",
        "kamis_ug_supply_lag1",
        "kamis_national_supply_lag1",
    ]:
        test[col] = np.nan

    return test


# ── Step 8: Final feature set ─────────────────────────────────────────────────


FEATURE_COLS = [
    # Core lag
    "lag1_price",
    "lag2_price",
    # Rolling
    "rolling_4w_mean",
    "deviation_from_rolling_mean",
    "rolling_4w_max",
    "price_momentum",
    # Calendar
    "week_of_year",
    "weeks_in_lean_season",
    "is_short_harvest_window",
    "weeks_to_long_harvest",
    # Cross-county
    "nairobi_lag1",
    "ug_lag1",
    "national_mean_lag1",
    "nairobi_ug_spread",
]
# Removed from v2→v3:
# - is_lean_season: importance=0 (superseded by weeks_in_lean_season)
# - kamis_*_lag1 (x4): all null at test time (KAMIS ends Jul 2025) — train-test mismatch
# - is_kamis_augmented: always 0 at test — misleads model
# - consecutive_up_weeks: hurt fold5 in v2 (3.73→4.09)
# - price_vs_4w_max: same — noise on sparse data

META_COLS = ["County", "Year_Week", "WeekofYear", "Date"]


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 60)
    print("FEATURE ENGINEERING — agriBORA v1")
    print("=" * 60)

    # Load
    train_raw, rolling_raw, kamis_raw = load_raw()

    # Build agriBORA panel
    panel = build_agribora_panel(train_raw, rolling_raw)

    # KAMIS augmentation for Kiambu + Mombasa
    panel = build_kamis_augmentation(kamis_raw, panel)

    # KAMIS feature tables
    kamis_agg, ug_supply, national_supply = build_kamis_features(kamis_raw)

    # Lag + rolling features
    panel = add_time_features(panel)

    # Calendar features
    panel = add_calendar_features(panel)

    # Cross-county features
    panel = add_cross_county_features(panel)

    # Merge KAMIS features
    panel = merge_kamis_features(panel, kamis_agg, ug_supply, national_supply)

    # Build test rows
    test = build_test_rows(panel)
    test = add_calendar_and_cross_county_to_test(
        test, panel, kamis_agg, ug_supply, national_supply
    )

    # ── Train: all rows with non-null target (WholeSale) and valid lag1 ──
    train_fe = panel[panel["WholeSale"].notna() & panel["lag1_price"].notna()].copy()

    _flag("INFO", f"\nTrain FE shape: {train_fe.shape}")
    _flag("INFO", f"Test FE shape: {test.shape}")

    # Feature null summary (train)
    print("\nNull counts in train features:")
    avail_cols = [c for c in FEATURE_COLS if c in train_fe.columns]
    null_summary = train_fe[avail_cols].isna().sum()
    for col, n in null_summary[null_summary > 0].items():
        pct = n / len(train_fe) * 100
        flag = "WARN" if pct > 20 else "INFO"
        _flag(flag, f"  {col}: {n} nulls ({pct:.1f}%)")
    if null_summary.sum() == 0:
        _flag("INFO", "  No nulls in feature columns.")

    # Obs per county in train
    print("\nTraining rows per county:")
    for county in TARGET_COUNTIES:
        n = len(train_fe[train_fe["County"] == county])
        aug = train_fe[
            (train_fe["County"] == county) & (train_fe["is_kamis_augmented"] == 1)
        ].shape[0]
        _flag("INFO", f"  {county}: {n} rows ({aug} KAMIS-augmented)")

    # Save
    avail_meta = [c for c in META_COLS if c in train_fe.columns]
    avail_feat = [c for c in FEATURE_COLS if c in train_fe.columns]

    train_out = train_fe[avail_meta + avail_feat + [TARGET_COL]].reset_index(drop=True)
    test_meta = [c for c in META_COLS if c in test.columns]
    test_feat = [c for c in FEATURE_COLS if c in test.columns]
    test_out = test[test_meta + test_feat].reset_index(drop=True)

    train_out.to_parquet(OUTPUT_DIR / "train_fe.parquet", index=False)
    test_out.to_parquet(OUTPUT_DIR / "test_fe.parquet", index=False)

    _flag("INFO", f"\nSaved: {OUTPUT_DIR}/train_fe.parquet ({train_out.shape})")
    _flag("INFO", f"Saved: {OUTPUT_DIR}/test_fe.parquet ({test_out.shape})")

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
