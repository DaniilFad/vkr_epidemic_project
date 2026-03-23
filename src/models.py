from pathlib import Path
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class ModelSpec:
    model_name: str
    model_kind: str
    feature_cols: list[str]


def parse_horizon_from_target(target_col: str) -> int:
    m = re.search(r"target_t_plus_(\d+)", target_col)
    if not m:
        raise ValueError(f"Cannot parse horizon from target column: {target_col}")
    return int(m.group(1))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-8, np.nan, np.abs(y_true))
    val = np.abs((y_true - y_pred) / denom)
    return float(np.nanmean(val) * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom < 1e-8, np.nan, denom)
    val = 2.0 * np.abs(y_pred - y_true) / denom
    return float(np.nanmean(val) * 100.0)


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    if denom < 1e-8:
        return np.nan
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, current_cases: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    current_cases = np.asarray(current_cases, dtype=float)

    actual_direction = np.sign(y_true - current_cases)
    pred_direction = np.sign(y_pred - current_cases)

    valid = ~np.isnan(actual_direction) & ~np.isnan(pred_direction)
    if valid.sum() == 0:
        return np.nan

    return float((actual_direction[valid] == pred_direction[valid]).mean() * 100.0)


def make_catboost_regressor(random_state: int = 42) -> CatBoostRegressor:
    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=400,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=8.0,
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
    )
    return model


def get_available(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [c for c in candidates if c in df.columns]


def get_top_wordstat_features(df: pd.DataFrame, top_k: int = 8) -> list[str]:
    lag_path = Path("reports/tables/wordstat_lag_diagnostics.csv")
    if lag_path.exists():
        lag_df = pd.read_csv(lag_path)
        lag_df = lag_df.sort_values("best_corr_abs", ascending=False).head(top_k)

        top_cols = []
        for _, row in lag_df.iterrows():
            den = f"{row['query_feature']}_denoised_lagbest"
            share = row["share_feature"]
            if den in df.columns:
                top_cols.append(den)
            if isinstance(share, str):
                share_lag = f"{share}_lagbest"
                if share_lag in df.columns:
                    top_cols.append(share_lag)
        return sorted(set(top_cols))

    fallback = [c for c in df.columns if c.endswith("_denoised_lagbest")]
    return sorted(fallback[:top_k])


def get_time_cols(df: pd.DataFrame) -> list[str]:
    candidates = [
        "year",
        "month",
        "quarter",
        "weekofyear",
        "weekofyear_sin",
        "weekofyear_cos",
        "month_sin",
        "month_cos",
        "time_idx",
    ]
    return get_available(df, candidates)


def get_calendar_cols(df: pd.DataFrame) -> list[str]:
    candidates = [
        "weekend_days",
        "holiday_days",
        "short_work_days",
        "school_break_days",
        "is_holiday_week",
        "is_school_break_week",
        "is_school_start_week",
        "is_new_year_period",
        "is_may_holiday_period",
        "restriction_level",
        "mask_mandate",
        "remote_work",
        "nonworking_days_total",
    ]
    return get_available(df, candidates)


def get_weather_cols(df: pd.DataFrame) -> list[str]:
    candidates = [
        "temp_mean",
        "temp_min",
        "temp_max",
        "humidity_mean",
        "pressure_mean",
        "wind_speed_mean",
        "precip_sum",
        "temp_range",
        "temp_mean_lag_1",
        "temp_mean_roll_mean_4",
        "temp_mean_roll_mean_8",
        "humidity_mean_lag_1",
        "precip_sum_lag_1",
        "precip_sum_roll_mean_4",
    ]
    return get_available(df, candidates)


def get_epi_cols_robust(df: pd.DataFrame) -> list[str]:
    candidates = [
        "cases",
        "cases_log1p",
        "cases_lag_1",
        "cases_lag_2",
        "cases_lag_3",
        "cases_lag_4",
        "cases_lag_8",
        "cases_lag_12",
        "cases_log1p_lag_1",
        "cases_log1p_lag_2",
        "cases_log1p_lag_3",
        "cases_roll_mean_2",
        "cases_roll_mean_4",
        "cases_roll_mean_8",
        "cases_roll_std_4",
        "cases_roll_std_8",
        "cases_growth_1w",
        "cases_growth_2w",
        "cases_growth_4w",
        "cases_ratio_1w",
        "cases_ratio_2w",
        "cases_ratio_4w",
        "cases_acceleration_1w",
        "cases_peak_distance_4w",
        "deaths",
        "recovered",
        "active_now",
        "active_delta",
        "deaths_lag_1",
        "recovered_lag_1",
        "active_now_lag_1",
        "active_delta_lag_1",
    ]
    cols = get_available(df, candidates)
    cols += get_time_cols(df)
    return sorted(set(cols))


def get_digital_cols_robust(df: pd.DataFrame) -> list[str]:
    candidates = [
        "ws_total_raw",
        "ws_total_denoised",
        "ws_total_spike_flags",
        "ws_total_denoised_roll_mean_4",
        "ws_total_denoised_roll_mean_8",
        "ws_total_spike_ratio",
        "ws_group_disease",
        "ws_group_testing",
        "ws_group_specific_symptoms",
        "ws_group_general_symptoms",
        "ws_group_complications_treatment",
        "ws_group_total",
    ]
    cols = get_available(df, candidates)
    cols += get_top_wordstat_features(df, top_k=8)
    cols += get_weather_cols(df)
    cols += get_calendar_cols(df)
    cols += get_time_cols(df)
    return sorted(set(cols))


def get_hybrid_cols_robust(df: pd.DataFrame) -> list[str]:
    cols = get_epi_cols_robust(df) + get_digital_cols_robust(df)
    return sorted(set(cols))


def build_model_specs(df: pd.DataFrame) -> list[ModelSpec]:
    return [
        ModelSpec(
            model_name="naive_last_week",
            model_kind="naive_last_week",
            feature_cols=[],
        ),
        ModelSpec(
            model_name="seasonal_naive_52",
            model_kind="seasonal_naive_52",
            feature_cols=[],
        ),
        ModelSpec(
            model_name="baseline_epi_catboost_log",
            model_kind="catboost_log",
            feature_cols=get_epi_cols_robust(df),
        ),
        ModelSpec(
            model_name="baseline_digital_catboost_log",
            model_kind="catboost_log",
            feature_cols=get_digital_cols_robust(df),
        ),
        ModelSpec(
            model_name="hybrid_catboost_log",
            model_kind="catboost_log",
            feature_cols=get_hybrid_cols_robust(df),
        ),
    ]


def prepare_train_data(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    train = df.copy()
    needed = feature_cols + [target_col]
    train = train.dropna(subset=[target_col]).copy()

    for col in feature_cols:
        if col not in train.columns:
            raise ValueError(f"Missing feature column: {col}")

    X = train[feature_cols].copy()
    y = train[target_col].astype(float).copy()

    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()

    return X, y


def predict_single_row_catboost_log(
    train_df: pd.DataFrame,
    test_row: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    random_state: int,
) -> float:
    X_train, y_train = prepare_train_data(train_df, feature_cols, target_col)

    X_test = test_row[feature_cols].copy()
    if X_test.isna().any(axis=1).iloc[0]:
        return np.nan

    y_train_log = np.log1p(y_train)

    model = make_catboost_regressor(random_state=random_state)
    model.fit(X_train, y_train_log)

    pred_log = float(model.predict(X_test)[0])
    pred = np.expm1(pred_log)
    pred = max(0.0, pred)
    return pred


def run_single_walkforward(
    df: pd.DataFrame,
    spec: ModelSpec,
    target_col: str,
    min_train_weeks: int = 80,
    evaluation_window: int = 52,
    train_window_weeks: int = 104,
    random_state: int = 42,
) -> pd.DataFrame:
    horizon = parse_horizon_from_target(target_col)
    n = len(df)

    test_start_idx = max(min_train_weeks, n - evaluation_window - horizon + 1)
    rows = []

    for i in range(test_start_idx, n):
        if pd.isna(df.loc[i, target_col]):
            continue

        train_start = max(0, i - train_window_weeks)
        train = df.iloc[train_start:i].copy()
        train = train[train[target_col].notna()].copy()

        if len(train) < min_train_weeks:
            continue

        current_row = df.iloc[[i]].copy()

        y_true = float(current_row.iloc[0][target_col])
        current_cases = float(current_row.iloc[0]["cases"])

        if spec.model_kind == "naive_last_week":
            y_pred = current_cases

        elif spec.model_kind == "seasonal_naive_52":
            if i - 52 >= 0 and not pd.isna(df.iloc[i - 52]["cases"]):
                y_pred = float(df.iloc[i - 52]["cases"])
            else:
                y_pred = current_cases

        elif spec.model_kind == "catboost_log":
            y_pred = predict_single_row_catboost_log(
                train_df=train,
                test_row=current_row,
                feature_cols=spec.feature_cols,
                target_col=target_col,
                random_state=random_state,
            )

        else:
            raise ValueError(f"Unknown model kind: {spec.model_kind}")

        rows.append({
            "week_start": current_row.iloc[0]["week_start"],
            "model_name": spec.model_name,
            "target_col": target_col,
            "horizon_weeks": horizon,
            "train_size": len(train),
            "current_cases": current_cases,
            "y_true": y_true,
            "y_pred": y_pred,
            "abs_error": abs(y_true - y_pred) if pd.notna(y_pred) else np.nan,
            "signed_error": y_pred - y_true if pd.notna(y_pred) else np.nan,
            "ape_pct": abs(y_true - y_pred) / y_true * 100.0 if (pd.notna(y_pred) and y_true != 0) else np.nan,
        })

    pred_df = pd.DataFrame(rows)
    if not pred_df.empty:
        pred_df["week_start"] = pd.to_datetime(pred_df["week_start"])
    return pred_df


def compute_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame()

    pred_df = pred_df.dropna(subset=["y_true", "y_pred"]).copy()

    metrics_rows = []
    group_cols = ["model_name", "target_col", "horizon_weeks"]

    for keys, group in pred_df.groupby(group_cols):
        model_name, target_col, horizon_weeks = keys

        y_true = group["y_true"].to_numpy(dtype=float)
        y_pred = group["y_pred"].to_numpy(dtype=float)
        current_cases = group["current_cases"].to_numpy(dtype=float)

        metrics_rows.append({
            "model_name": model_name,
            "target_col": target_col,
            "horizon_weeks": int(horizon_weeks),
            "n_predictions": len(group),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mape_pct": mape(y_true, y_pred),
            "smape_pct": smape(y_true, y_pred),
            "wape_pct": wape(y_true, y_pred),
            "directional_accuracy_pct": directional_accuracy(y_true, y_pred, current_cases),
            "mean_true": float(np.mean(y_true)),
            "mean_pred": float(np.mean(y_pred)),
        })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df.sort_values(["horizon_weeks", "rmse", "mae"]).reset_index(drop=True)
    return metrics_df


def run_all_models_walkforward(
    feature_df: pd.DataFrame,
    target_cols: list[str],
    min_train_weeks: int = 80,
    evaluation_window: int = 52,
    train_window_weeks: int = 104,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    specs = build_model_specs(feature_df)

    feature_usage_rows = []
    all_pred_parts = []

    for spec in specs:
        feature_usage_rows.append({
            "model_name": spec.model_name,
            "model_kind": spec.model_kind,
            "n_features": len(spec.feature_cols),
            "feature_cols": ", ".join(spec.feature_cols),
        })

        for target_col in target_cols:
            pred_df = run_single_walkforward(
                df=feature_df,
                spec=spec,
                target_col=target_col,
                min_train_weeks=min_train_weeks,
                evaluation_window=evaluation_window,
                train_window_weeks=train_window_weeks,
                random_state=random_state,
            )
            all_pred_parts.append(pred_df)

    predictions_df = pd.concat(all_pred_parts, ignore_index=True)
    predictions_df = predictions_df.sort_values(["horizon_weeks", "model_name", "week_start"]).reset_index(drop=True)

    metrics_df = compute_metrics(predictions_df)
    feature_usage_df = pd.DataFrame(feature_usage_rows).sort_values("model_name").reset_index(drop=True)

    return predictions_df, metrics_df, feature_usage_df


def add_metric_improvements(metrics_df: pd.DataFrame, baseline_name: str = "naive_last_week") -> pd.DataFrame:
    out = metrics_df.copy()
    if out.empty:
        return out

    out["rmse_improvement_vs_naive_pct"] = np.nan
    out["mae_improvement_vs_naive_pct"] = np.nan
    out["mape_improvement_vs_naive_pct"] = np.nan

    for horizon in sorted(out["horizon_weeks"].unique()):
        mask_h = out["horizon_weeks"] == horizon
        base = out.loc[mask_h & (out["model_name"] == baseline_name)]

        if base.empty:
            continue

        base_rmse = float(base["rmse"].iloc[0])
        base_mae = float(base["mae"].iloc[0])
        base_mape = float(base["mape_pct"].iloc[0])

        idx = out.index[mask_h]
        out.loc[idx, "rmse_improvement_vs_naive_pct"] = (base_rmse - out.loc[idx, "rmse"]) / base_rmse * 100.0
        out.loc[idx, "mae_improvement_vs_naive_pct"] = (base_mae - out.loc[idx, "mae"]) / base_mae * 100.0
        out.loc[idx, "mape_improvement_vs_naive_pct"] = (base_mape - out.loc[idx, "mape_pct"]) / base_mape * 100.0

    return out


def save_step5_outputs(
    root: Path,
    predictions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    feature_usage_df: pd.DataFrame,
    suffix: str = "step5_robust",
) -> tuple[Path, Path, Path]:
    predictions_dir = root / "reports" / "predictions"
    tables_dir = root / "reports" / "tables"

    predictions_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = predictions_dir / f"walkforward_predictions_{suffix}.csv"
    metrics_path = tables_dir / f"walkforward_metrics_{suffix}.csv"
    feature_usage_path = tables_dir / f"model_feature_sets_{suffix}.csv"

    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    feature_usage_df.to_csv(feature_usage_path, index=False, encoding="utf-8-sig")

    return predictions_path, metrics_path, feature_usage_path
