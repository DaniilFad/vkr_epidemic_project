from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
BASE_DATASET_PATH = ROOT / "data" / "interim" / "base_weekly_dataset.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "modeling_dataset_advanced.csv"


WORDSTAT_GROUPS = {
    "disease": [
        "ws_covid",
        "ws_ковид",
        "ws_коронавирус",
        "ws_симптомы_ковида",
    ],
    "testing": [
        "ws_пцр_тест",
        "ws_тест_на_ковид",
        "ws_экспресс_тест_ковид",
        "ws_положительный_тест_ковид",
    ],
    "specific_symptoms": [
        "ws_нет_запахов",
        "ws_потеря_вкуса",
        "ws_пропало_обоняние",
        "ws_сатурация",
    ],
    "general_symptoms": [
        "ws_боль_в_горле",
        "ws_ломота",
        "ws_одышка",
        "ws_сухой_кашель",
        "ws_температура_38",
        "ws_температура_39",
    ],
    "complications_treatment": [
        "ws_кт_легких",
        "ws_пневмония",
        "ws_чем_лечить_ковид",
    ],
}


def safe_log1p(series: pd.Series) -> pd.Series:
    return np.log1p(series.clip(lower=0))


def make_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["year"] = out["week_start"].dt.year
    out["month"] = out["week_start"].dt.month
    out["quarter"] = out["week_start"].dt.quarter
    out["weekofyear"] = out["week_start"].dt.isocalendar().week.astype(int)

    out["time_idx"] = np.arange(len(out))

    out["weekofyear_sin"] = np.sin(2 * np.pi * out["weekofyear"] / 52.0)
    out["weekofyear_cos"] = np.cos(2 * np.pi * out["weekofyear"] / 52.0)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)

    return out


def make_case_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["cases_log1p"] = safe_log1p(out["cases"])

    for lag in [1, 2, 3, 4, 8, 12]:
        out[f"cases_lag_{lag}"] = out["cases"].shift(lag)
        out[f"cases_log1p_lag_{lag}"] = out["cases_log1p"].shift(lag)

    for win in [2, 4, 8]:
        out[f"cases_roll_mean_{win}"] = out["cases"].rolling(win, min_periods=1).mean()
        out[f"cases_roll_std_{win}"] = out["cases"].rolling(win, min_periods=2).std()
        out[f"cases_roll_min_{win}"] = out["cases"].rolling(win, min_periods=1).min()
        out[f"cases_roll_max_{win}"] = out["cases"].rolling(win, min_periods=1).max()

    out["cases_growth_1w"] = out["cases"] / out["cases"].shift(1) - 1.0
    out["cases_growth_2w"] = out["cases"] / out["cases"].shift(2) - 1.0
    out["cases_growth_4w"] = out["cases"] / out["cases"].shift(4) - 1.0

    out["cases_ratio_1w"] = out["cases"] / out["cases"].shift(1)
    out["cases_ratio_2w"] = out["cases"] / out["cases"].shift(2)
    out["cases_ratio_4w"] = out["cases"] / out["cases"].shift(4)

    out["cases_acceleration_1w"] = out["cases_growth_1w"] - out["cases_growth_1w"].shift(1)
    out["cases_peak_distance_4w"] = out["cases"] - out["cases_roll_max_4"]

    for col in ["deaths", "recovered", "active_now", "active_delta"]:
        if col in out.columns:
            out[f"{col}_lag_1"] = out[col].shift(1)

    return out


def get_wordstat_cols(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if c.startswith("ws_") and not c.startswith("ws_share_")])


def get_share_cols(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if c.startswith("ws_share_")])


def make_wordstat_group_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for group_name, cols in WORDSTAT_GROUPS.items():
        existing = [c for c in cols if c in out.columns]
        if existing:
            out[f"ws_group_{group_name}"] = out[existing].sum(axis=1)
        else:
            out[f"ws_group_{group_name}"] = 0.0

    group_cols = [c for c in out.columns if c.startswith("ws_group_")]
    out["ws_group_total"] = out[group_cols].sum(axis=1)

    raw_cols = get_wordstat_cols(out)
    if raw_cols:
        out["ws_total_raw"] = out[raw_cols].sum(axis=1)
    else:
        out["ws_total_raw"] = 0.0

    out["ws_total_denoised"] = out["ws_total_raw"]

    roll4 = out["ws_total_raw"].rolling(4, min_periods=2).mean()
    roll8 = out["ws_total_raw"].rolling(8, min_periods=3).mean()
    std8 = out["ws_total_raw"].rolling(8, min_periods=3).std()

    out["ws_total_denoised_roll_mean_4"] = roll4
    out["ws_total_denoised_roll_mean_8"] = roll8

    out["ws_total_spike_ratio"] = out["ws_total_raw"] / roll4
    out["ws_total_spike_flags"] = (
        (out["ws_total_raw"] > roll4 + 1.5 * std8.fillna(0))
    ).astype(int)

    return out


def make_wordstat_growth_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    candidate_cols = []

    candidate_cols.extend([c for c in out.columns if c.startswith("ws_group_")])
    candidate_cols.extend(["ws_total_raw", "ws_total_denoised"])

    raw_cols = get_wordstat_cols(out)
    candidate_cols.extend(raw_cols)

    candidate_cols = sorted(set([c for c in candidate_cols if c in out.columns]))

    for col in candidate_cols:
        out[f"{col}_lag_1"] = out[col].shift(1)
        out[f"{col}_lag_2"] = out[col].shift(2)

        out[f"{col}_growth_1w"] = out[col] / out[col].shift(1) - 1.0
        out[f"{col}_growth_2w"] = out[col] / out[col].shift(2) - 1.0

        out[f"{col}_diff_1w"] = out[col] - out[col].shift(1)
        out[f"{col}_diff_2w"] = out[col] - out[col].shift(2)

        roll4 = out[col].rolling(4, min_periods=2).mean()
        roll8 = out[col].rolling(8, min_periods=3).mean()
        std8 = out[col].rolling(8, min_periods=3).std()

        out[f"{col}_roll_mean_4"] = roll4
        out[f"{col}_roll_mean_8"] = roll8
        out[f"{col}_zscore_8"] = (out[col] - roll8) / std8
        out[f"{col}_shock_ratio"] = out[col] / roll4

    return out


def best_horizon_specific_lag(
    series: pd.Series,
    target: pd.Series,
    lag_candidates: list[int],
) -> tuple[int, float]:
    best_lag = None
    best_abs_corr = -np.inf

    for lag in lag_candidates:
        shifted = series.shift(lag)
        valid = shifted.notna() & target.notna()

        if valid.sum() < 20:
            continue

        corr = shifted[valid].corr(target[valid])
        if pd.isna(corr):
            continue

        if abs(corr) > best_abs_corr:
            best_abs_corr = abs(corr)
            best_lag = lag

    if best_lag is None:
        return lag_candidates[0], np.nan

    return best_lag, best_abs_corr


def make_horizon_specific_lag_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()

    lag_rows = []
    lag_candidates = [1, 2, 3, 4, 5, 6, 7, 8]

    source_cols = []
    source_cols.extend([c for c in out.columns if c.startswith("ws_group_")])
    source_cols.extend(["ws_total_raw", "ws_total_denoised"])

    raw_cols = get_wordstat_cols(out)
    share_cols = get_share_cols(out)

    source_cols.extend(raw_cols)
    source_cols.extend(share_cols)

    source_cols = sorted(set([c for c in source_cols if c in out.columns]))

    for horizon in [1, 2, 3]:
        target_col = f"target_t_plus_{horizon}"
        if target_col not in out.columns:
            continue

        for col in source_cols:
            best_lag, best_abs_corr = best_horizon_specific_lag(
                series=out[col],
                target=out[target_col],
                lag_candidates=lag_candidates,
            )

            new_col = f"{col}_lagbest_h{horizon}"
            out[new_col] = out[col].shift(best_lag)

            lag_rows.append({
                "horizon_weeks": horizon,
                "feature": col,
                "best_lag_weeks": best_lag,
                "best_corr_abs": best_abs_corr,
            })

    lag_df = pd.DataFrame(lag_rows).sort_values(
        ["horizon_weeks", "best_corr_abs"],
        ascending=[True, False]
    ).reset_index(drop=True)

    return out, lag_df


def make_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if {"temp_max", "temp_min"}.issubset(out.columns):
        out["temp_range"] = out["temp_max"] - out["temp_min"]

    for col in ["temp_mean", "humidity_mean", "precip_sum"]:
        if col in out.columns:
            out[f"{col}_lag_1"] = out[col].shift(1)
            out[f"{col}_roll_mean_4"] = out[col].rolling(4, min_periods=2).mean()
            out[f"{col}_roll_mean_8"] = out[col].rolling(8, min_periods=3).mean()

    return out


def make_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for horizon in [1, 2, 3]:
        future = out["cases"].shift(-horizon)

        out[f"target_t_plus_{horizon}"] = future

        out[f"target_logdelta_h{horizon}"] = safe_log1p(future) - safe_log1p(out["cases"])

        ratio = future / out["cases"]
        out[f"target_ratio_h{horizon}"] = ratio

        out[f"target_growth_pct_h{horizon}"] = ratio - 1.0

    return out


def main():
    if not BASE_DATASET_PATH.exists():
        raise FileNotFoundError(f"Не найден файл: {BASE_DATASET_PATH}")

    df = pd.read_csv(BASE_DATASET_PATH)
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values("week_start").reset_index(drop=True)

    df = make_time_features(df)
    df = make_case_features(df)
    df = make_wordstat_group_features(df)
    df = make_wordstat_growth_features(df)
    df = make_weather_features(df)
    df = make_targets(df)
    df, lag_df = make_horizon_specific_lag_features(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    lag_output = ROOT / "reports" / "tables" / "wordstat_horizon_specific_lags_step10.csv"
    lag_output.parent.mkdir(parents=True, exist_ok=True)
    lag_df.to_csv(lag_output, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("STEP 10 RESULT")
    print("=" * 80)
    print("saved advanced dataset:", OUTPUT_PATH)
    print("saved horizon lags    :", lag_output)
    print()
    print("shape:", df.shape)
    print("date range:", df["week_start"].min(), "->", df["week_start"].max())
    print()

    preview_cols = [
        "week_start",
        "cases",
        "target_t_plus_1",
        "target_t_plus_2",
        "target_t_plus_3",
        "target_logdelta_h1",
        "target_logdelta_h2",
        "target_logdelta_h3",
        "target_ratio_h1",
        "target_ratio_h2",
        "target_ratio_h3",
    ]
    preview_cols = [c for c in preview_cols if c in df.columns]

    print("TARGET PREVIEW")
    print(df[preview_cols].head(12).to_string(index=False))
    print()

    lag_preview = lag_df.head(30)
    print("TOP HORIZON-SPECIFIC LAGS")
    print(lag_preview.to_string(index=False))
    print()

    key_growth_cols = [
        "ws_total_raw_growth_1w",
        "ws_total_raw_growth_2w",
        "ws_total_raw_shock_ratio",
        "ws_group_general_symptoms_growth_1w",
        "ws_group_general_symptoms_growth_2w",
        "ws_group_general_symptoms_shock_ratio",
    ]
    key_growth_cols = [c for c in key_growth_cols if c in df.columns]

    if key_growth_cols:
        print("KEY WORDSTAT GROWTH FEATURE PREVIEW")
        print(df[["week_start"] + key_growth_cols].head(12).to_string(index=False))
        print()

    missing = df.isna().sum().sort_values(ascending=False)
    print("MISSING VALUES (TOP 40)")
    print(missing.head(40).to_string())


if __name__ == "__main__":
    main()
