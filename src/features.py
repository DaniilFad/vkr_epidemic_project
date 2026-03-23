from pathlib import Path
import numpy as np
import pandas as pd


def safe_divide(a, b):
    a = pd.Series(a)
    b = pd.Series(b)
    out = a / b.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def get_wordstat_count_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("ws_") and not c.startswith("ws_share_")]


def get_wordstat_share_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("ws_share_")]


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target_t_plus_1"] = out["cases"].shift(-1)
    out["target_t_plus_2"] = out["cases"].shift(-2)
    out["target_t_plus_3"] = out["cases"].shift(-3)
    return out


def add_case_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["cases_log1p"] = np.log1p(out["cases"])

    lag_list = [1, 2, 3, 4, 8, 12]
    for lag in lag_list:
        out[f"cases_lag_{lag}"] = out["cases"].shift(lag)
        out[f"cases_log1p_lag_{lag}"] = np.log1p(out[f"cases_lag_{lag}"])

    optional_case_cols = ["deaths", "recovered", "active_now", "active_delta"]
    for col in optional_case_cols:
        if col in out.columns:
            out[f"{col}_lag_1"] = out[col].shift(1)
            out[f"{col}_lag_2"] = out[col].shift(2)

    for window in [2, 4, 8, 12]:
        out[f"cases_roll_mean_{window}"] = out["cases"].rolling(window=window, min_periods=1).mean()
        out[f"cases_roll_std_{window}"] = out["cases"].rolling(window=window, min_periods=2).std()
        out[f"cases_roll_min_{window}"] = out["cases"].rolling(window=window, min_periods=1).min()
        out[f"cases_roll_max_{window}"] = out["cases"].rolling(window=window, min_periods=1).max()

    out["cases_growth_1w"] = safe_divide(out["cases"] - out["cases_lag_1"], out["cases_lag_1"])
    out["cases_growth_2w"] = safe_divide(out["cases"] - out["cases_lag_2"], out["cases_lag_2"])
    out["cases_growth_4w"] = safe_divide(out["cases"] - out["cases_lag_4"], out["cases_lag_4"])

    out["cases_ratio_1w"] = safe_divide(out["cases"], out["cases_lag_1"])
    out["cases_ratio_2w"] = safe_divide(out["cases"], out["cases_lag_2"])
    out["cases_ratio_4w"] = safe_divide(out["cases"], out["cases_lag_4"])

    out["cases_acceleration_1w"] = out["cases_growth_1w"] - out["cases_growth_1w"].shift(1)
    out["cases_peak_distance_4w"] = safe_divide(
        out["cases"] - out["cases_roll_max_4"], out["cases_roll_max_4"]
    )

    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    week = out["weekofyear"].astype(float)
    out["weekofyear_sin"] = np.sin(2 * np.pi * week / 52.0)
    out["weekofyear_cos"] = np.cos(2 * np.pi * week / 52.0)

    month = out["month"].astype(float)
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    out["time_idx"] = np.arange(len(out))
    return out


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    weather_cols = [
        "temp_mean",
        "temp_min",
        "temp_max",
        "humidity_mean",
        "pressure_mean",
        "wind_speed_mean",
        "precip_sum",
    ]

    for col in weather_cols:
        if col in out.columns:
            out[f"{col}_lag_1"] = out[col].shift(1)
            out[f"{col}_roll_mean_4"] = out[col].rolling(4, min_periods=1).mean()
            out[f"{col}_roll_mean_8"] = out[col].rolling(8, min_periods=1).mean()

    if {"temp_max", "temp_min"}.issubset(out.columns):
        out["temp_range"] = out["temp_max"] - out["temp_min"]

    return out


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    binary_cols = [
        "is_holiday_week",
        "is_school_break_week",
        "is_school_start_week",
        "is_new_year_period",
        "is_may_holiday_period",
        "mask_mandate",
        "remote_work",
    ]
    for col in binary_cols:
        if col in out.columns:
            out[col] = out[col].fillna(0).astype(int)

    count_cols = [
        "weekend_days",
        "holiday_days",
        "short_work_days",
        "school_break_days",
        "restriction_level",
    ]
    for col in count_cols:
        if col in out.columns:
            out[col] = out[col].fillna(0)

    if {"holiday_days", "weekend_days"}.issubset(out.columns):
        out["nonworking_days_total"] = out["holiday_days"] + out["weekend_days"]

    return out


def add_wordstat_anti_media_noise(df: pd.DataFrame, rolling_window: int = 8, cap_multiplier: float = 2.0) -> pd.DataFrame:
    out = df.copy()
    count_cols = get_wordstat_count_cols(out)

    for col in count_cols:
        baseline = out[col].shift(1).rolling(rolling_window, min_periods=3).median()
        shock_ratio = safe_divide(out[col], baseline)

        denoised = out[col].copy()
        mask = baseline.notna() & (baseline > 0)
        denoised.loc[mask] = np.minimum(out.loc[mask, col], baseline.loc[mask] * cap_multiplier)

        out[f"{col}_baseline_med{rolling_window}"] = baseline
        out[f"{col}_shock_ratio"] = shock_ratio
        out[f"{col}_denoised"] = denoised
        out[f"{col}_spike_flag"] = (shock_ratio > cap_multiplier).fillna(False).astype(int)

    count_cols_denoised = [f"{c}_denoised" for c in count_cols]
    spike_flag_cols = [f"{c}_spike_flag" for c in count_cols]

    if count_cols:
        out["ws_total_raw"] = out[count_cols].sum(axis=1)
    if count_cols_denoised:
        out["ws_total_denoised"] = out[count_cols_denoised].sum(axis=1)
    if spike_flag_cols:
        out["ws_total_spike_flags"] = out[spike_flag_cols].sum(axis=1)

    out["ws_total_denoised_roll_mean_4"] = out["ws_total_denoised"].rolling(4, min_periods=1).mean()
    out["ws_total_denoised_roll_mean_8"] = out["ws_total_denoised"].rolling(8, min_periods=1).mean()
    out["ws_total_spike_ratio"] = safe_divide(out["ws_total_raw"], out["ws_total_denoised_roll_mean_8"])

    return out


def find_best_wordstat_lags(
    df: pd.DataFrame,
    target_col: str = "cases",
    max_lag: int = 6,
    min_obs: int = 30,
) -> pd.DataFrame:
    records = []

    count_cols = get_wordstat_count_cols(df)

    for raw_col in count_cols:
        denoised_col = f"{raw_col}_denoised"
        if denoised_col not in df.columns:
            continue

        best_lag = None
        best_corr = None
        best_n = 0

        for lag in range(0, max_lag + 1):
            x = df[denoised_col].shift(lag)
            y = df[target_col]

            valid = pd.DataFrame({"x": x, "y": y}).dropna()
            n_obs = len(valid)
            if n_obs < min_obs:
                continue

            if valid["x"].nunique() <= 1 or valid["y"].nunique() <= 1:
                continue

            corr = valid["x"].corr(valid["y"])
            if pd.isna(corr):
                continue

            if best_corr is None or abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag
                best_n = n_obs

        share_col = raw_col.replace("ws_", "ws_share_", 1)

        records.append({
            "query_feature": raw_col,
            "query_feature_denoised": denoised_col,
            "share_feature": share_col if share_col in df.columns else None,
            "best_lag_weeks": best_lag,
            "best_corr_abs": abs(best_corr) if best_corr is not None else np.nan,
            "best_corr_signed": best_corr,
            "n_obs": best_n,
        })

    lag_df = pd.DataFrame(records)
    lag_df = lag_df.sort_values(["best_corr_abs", "query_feature"], ascending=[False, True]).reset_index(drop=True)
    return lag_df


def apply_best_wordstat_lags(df: pd.DataFrame, lag_table: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for _, row in lag_table.iterrows():
        raw_col = row["query_feature"]
        denoised_col = row["query_feature_denoised"]
        share_col = row["share_feature"]
        lag = row["best_lag_weeks"]

        if pd.isna(lag):
            continue

        lag = int(lag)

        if denoised_col in out.columns:
            out[f"{raw_col}_denoised_lagbest"] = out[denoised_col].shift(lag)

        if isinstance(share_col, str) and share_col in out.columns:
            out[f"{share_col}_lagbest"] = out[share_col].shift(lag)

    return out


def add_wordstat_group_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    groups = {
        "ws_group_disease": [
            "ws_covid_denoised_lagbest",
            "ws_ковид_denoised_lagbest",
            "ws_коронавирус_denoised_lagbest",
            "ws_симптомы_ковида_denoised_lagbest",
        ],
        "ws_group_testing": [
            "ws_тест_на_ковид_denoised_lagbest",
            "ws_экспресс_тест_ковид_denoised_lagbest",
            "ws_пцр_тест_denoised_lagbest",
            "ws_положительный_тест_ковид_denoised_lagbest",
        ],
        "ws_group_specific_symptoms": [
            "ws_пропало_обоняние_denoised_lagbest",
            "ws_потеря_вкуса_denoised_lagbest",
            "ws_нет_запахов_denoised_lagbest",
            "ws_сатурация_denoised_lagbest",
            "ws_одышка_denoised_lagbest",
        ],
        "ws_group_general_symptoms": [
            "ws_температура_38_denoised_lagbest",
            "ws_температура_39_denoised_lagbest",
            "ws_сухой_кашель_denoised_lagbest",
            "ws_ломота_denoised_lagbest",
            "ws_боль_в_горле_denoised_lagbest",
        ],
        "ws_group_complications_treatment": [
            "ws_пневмония_denoised_lagbest",
            "ws_кт_легких_denoised_lagbest",
            "ws_чем_лечить_ковид_denoised_lagbest",
        ],
    }

    for group_name, cols in groups.items():
        existing = [c for c in cols if c in out.columns]
        if existing:
            out[group_name] = out[existing].sum(axis=1)
        else:
            out[group_name] = 0.0

    out["ws_group_total"] = out[
        [
            "ws_group_disease",
            "ws_group_testing",
            "ws_group_specific_symptoms",
            "ws_group_general_symptoms",
            "ws_group_complications_treatment",
        ]
    ].sum(axis=1)

    return out


def build_feature_dataset(base_df: pd.DataFrame, max_wordstat_lag: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = base_df.copy()
    df = df.sort_values("week_start").reset_index(drop=True)

    df = add_targets(df)
    df = add_case_features(df)
    df = add_time_features(df)
    df = add_weather_features(df)
    df = add_calendar_features(df)
    df = add_wordstat_anti_media_noise(df, rolling_window=8, cap_multiplier=2.0)

    lag_table = find_best_wordstat_lags(
        df=df,
        target_col="cases",
        max_lag=max_wordstat_lag,
        min_obs=30,
    )

    df = apply_best_wordstat_lags(df, lag_table)
    df = add_wordstat_group_features(df)

    return df, lag_table


def save_feature_outputs(
    feature_df: pd.DataFrame,
    lag_table: pd.DataFrame,
    root: Path,
) -> tuple[Path, Path]:
    processed_dir = root / "data" / "processed"
    tables_dir = root / "reports" / "tables"

    processed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    feature_path = processed_dir / "modeling_dataset.csv"
    lag_path = tables_dir / "wordstat_lag_diagnostics.csv"

    feature_df.to_csv(feature_path, index=False, encoding="utf-8-sig")
    lag_table.to_csv(lag_path, index=False, encoding="utf-8-sig")

    return feature_path, lag_path