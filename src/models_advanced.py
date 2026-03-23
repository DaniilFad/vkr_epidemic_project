from pathlib import Path
from dataclasses import dataclass
import re

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class AdvancedModelSpec:
    model_name: str
    model_kind: str  # naive | level | logdelta | ratio


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


def make_catboost_regressor(random_state: int = 42) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=350,
        learning_rate=0.04,
        depth=4,
        l2_leaf_reg=8.0,
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
    )


def get_available(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [c for c in candidates if c in df.columns]


def build_model_specs() -> list[AdvancedModelSpec]:
    return [
        AdvancedModelSpec("naive_last_week", "naive"),
        AdvancedModelSpec("advanced_level_catboost", "level"),
        AdvancedModelSpec("advanced_logdelta_catboost", "logdelta"),
        AdvancedModelSpec("advanced_ratio_catboost", "ratio"),
    ]


def get_time_cols(df: pd.DataFrame) -> list[str]:
    return get_available(df, [
        "year", "month", "quarter", "weekofyear",
        "weekofyear_sin", "weekofyear_cos",
        "month_sin", "month_cos",
        "time_idx",
    ])


def get_weather_cols(df: pd.DataFrame) -> list[str]:
    return get_available(df, [
        "temp_mean", "temp_min", "temp_max",
        "humidity_mean", "pressure_mean", "wind_speed_mean", "precip_sum",
        "temp_range",
        "temp_mean_lag_1", "temp_mean_roll_mean_4", "temp_mean_roll_mean_8",
        "humidity_mean_lag_1", "humidity_mean_roll_mean_4",
        "precip_sum_lag_1", "precip_sum_roll_mean_4",
    ])


def get_calendar_cols(df: pd.DataFrame) -> list[str]:
    return get_available(df, [
        "weekend_days", "holiday_days", "short_work_days", "school_break_days",
        "is_holiday_week", "is_school_break_week", "is_school_start_week",
        "is_new_year_period", "is_may_holiday_period",
        "restriction_level", "mask_mandate", "remote_work",
    ])


def get_epi_core_cols(df: pd.DataFrame) -> list[str]:
    return get_available(df, [
        "cases",
        "cases_log1p",
        "cases_lag_1", "cases_lag_2", "cases_lag_3", "cases_lag_4", "cases_lag_8", "cases_lag_12",
        "cases_log1p_lag_1", "cases_log1p_lag_2", "cases_log1p_lag_3",
        "cases_roll_mean_2", "cases_roll_mean_4", "cases_roll_mean_8",
        "cases_roll_std_2", "cases_roll_std_4", "cases_roll_std_8",
        "cases_growth_1w", "cases_growth_2w", "cases_growth_4w",
        "cases_ratio_1w", "cases_ratio_2w", "cases_ratio_4w",
        "cases_acceleration_1w", "cases_peak_distance_4w",
        "deaths", "recovered", "active_now", "active_delta",
        "deaths_lag_1", "recovered_lag_1", "active_now_lag_1", "active_delta_lag_1",
    ])


def get_horizon_specific_digital_candidates(df: pd.DataFrame, horizon: int) -> list[str]:
    cols = []

    # агрегаты
    for c in df.columns:
        if c.startswith("ws_group_") or c.startswith("ws_total_"):
            cols.append(c)

    # динамика и шоки
    for c in df.columns:
        if c.startswith("ws_") or c.startswith("ws_group_") or c.startswith("ws_total_"):
            if any(key in c for key in [
                "_growth_1w", "_growth_2w",
                "_diff_1w", "_diff_2w",
                "_zscore_8",
                "_shock_ratio",
                "_roll_mean_4", "_roll_mean_8",
                "_lag_1", "_lag_2",
            ]):
                cols.append(c)

    # именно горизонт-специфические лаги
    suffix = f"_lagbest_h{horizon}"
    for c in df.columns:
        if c.endswith(suffix):
            cols.append(c)

    cols = sorted(set(cols))
    return cols


def clean_feature_series(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s


def select_top_features(
    train_df: pd.DataFrame,
    target_col: str,
    candidate_cols: list[str],
    mandatory_cols: list[str],
    top_k_total: int = 40,
    min_obs: int = 25,
) -> list[str]:
    target = clean_feature_series(train_df[target_col])

    scores = []
    for col in candidate_cols:
        if col not in train_df.columns:
            continue

        x = clean_feature_series(train_df[col])

        valid = x.notna() & target.notna()
        if valid.sum() < min_obs:
            continue

        xv = x.loc[valid]
        yv = target.loc[valid]

        if xv.nunique(dropna=True) <= 1:
            continue

        corr = xv.corr(yv)
        if pd.isna(corr):
            continue

        scores.append((col, abs(float(corr))))

    scores = sorted(scores, key=lambda z: z[1], reverse=True)

    selected = []
    for col in mandatory_cols:
        if col in train_df.columns and col not in selected:
            selected.append(col)

    for col, _ in scores:
        if col not in selected:
            selected.append(col)
        if len(selected) >= top_k_total:
            break

    return selected


def prepare_xy(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
):
    X = train_df[feature_cols].copy()
    y = pd.to_numeric(train_df[target_col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)

    mask = y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].astype(float).copy()

    return X, y, medians


def fit_predict_level(
    train_df: pd.DataFrame,
    test_row: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    random_state: int,
) -> float:
    X_train, y_train, medians = prepare_xy(train_df, feature_cols, target_col)
    X_test = test_row[feature_cols].copy().replace([np.inf, -np.inf], np.nan).fillna(medians)

    model = make_catboost_regressor(random_state=random_state)
    model.fit(X_train, np.log1p(y_train))

    pred_log = float(model.predict(X_test)[0])
    pred = float(np.expm1(pred_log))
    return max(0.0, pred)


def fit_predict_logdelta(
    train_df: pd.DataFrame,
    test_row: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    random_state: int,
) -> float:
    X_train, y_train, medians = prepare_xy(train_df, feature_cols, target_col)
    X_test = test_row[feature_cols].copy().replace([np.inf, -np.inf], np.nan).fillna(medians)

    model = make_catboost_regressor(random_state=random_state)
    model.fit(X_train, y_train)

    pred_delta = float(model.predict(X_test)[0])
    current_cases = float(test_row.iloc[0]["cases"])
    pred = float(np.expm1(np.log1p(current_cases) + pred_delta))
    return max(0.0, pred)


def fit_predict_ratio(
    train_df: pd.DataFrame,
    test_row: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    random_state: int,
) -> float:
    X_train, y_train, medians = prepare_xy(train_df, feature_cols, target_col)
    X_test = test_row[feature_cols].copy().replace([np.inf, -np.inf], np.nan).fillna(medians)

    y_train = y_train.clip(lower=0.01, upper=20.0)

    model = make_catboost_regressor(random_state=random_state)
    model.fit(X_train, np.log(y_train))

    pred_log_ratio = float(model.predict(X_test)[0])
    pred_ratio = float(np.exp(pred_log_ratio))
    pred_ratio = min(max(pred_ratio, 0.01), 20.0)

    current_cases = float(test_row.iloc[0]["cases"])
    pred = current_cases * pred_ratio
    return max(0.0, pred)


def build_feature_pool_for_horizon(df: pd.DataFrame, horizon: int) -> tuple[list[str], list[str], list[str]]:
    epi_core = get_epi_core_cols(df)
    time_cols = get_time_cols(df)
    weather_cols = get_weather_cols(df)
    calendar_cols = get_calendar_cols(df)
    digital = get_horizon_specific_digital_candidates(df, horizon)

    mandatory = sorted(set([
        c for c in [
            "cases", "cases_log1p",
            "cases_lag_1", "cases_lag_2", "cases_lag_3",
            "cases_roll_mean_2", "cases_roll_mean_4",
            "cases_growth_1w", "cases_growth_2w",
            "weekofyear_sin", "weekofyear_cos",
        ]
        if c in df.columns
    ]))

    pool = sorted(set(epi_core + time_cols + weather_cols + calendar_cols + digital))
    return pool, mandatory, digital


def run_single_walkforward_advanced(
    df: pd.DataFrame,
    spec: AdvancedModelSpec,
    horizon: int,
    min_train_weeks: int = 52,
    evaluation_window: int = 52,
    train_window_weeks: int = 104,
    random_state: int = 42,
) -> pd.DataFrame:
    level_target = f"target_t_plus_{horizon}"
    logdelta_target = f"target_logdelta_h{horizon}"
    ratio_target = f"target_ratio_h{horizon}"

    n = len(df)
    test_start_idx = max(min_train_weeks, n - evaluation_window - horizon + 1)

    pool, mandatory, digital_pool = build_feature_pool_for_horizon(df, horizon)

    rows = []

    for i in range(test_start_idx, n):
        if pd.isna(df.loc[i, level_target]):
            continue

        train_start = max(0, i - train_window_weeks)
        train_df = df.iloc[train_start:i].copy()
        test_row = df.iloc[[i]].copy()

        current_cases = float(test_row.iloc[0]["cases"])
        y_true = float(test_row.iloc[0][level_target])

        if spec.model_kind == "naive":
            y_pred = current_cases
            selected_features = []

        else:
            if spec.model_kind == "level":
                target_col = level_target
                selected_features = select_top_features(
                    train_df=train_df,
                    target_col=target_col,
                    candidate_cols=pool,
                    mandatory_cols=mandatory,
                    top_k_total=38,
                    min_obs=25,
                )
                y_pred = fit_predict_level(
                    train_df=train_df,
                    test_row=test_row,
                    target_col=target_col,
                    feature_cols=selected_features,
                    random_state=random_state,
                )

            elif spec.model_kind == "logdelta":
                target_col = logdelta_target
                selected_features = select_top_features(
                    train_df=train_df,
                    target_col=target_col,
                    candidate_cols=pool,
                    mandatory_cols=mandatory,
                    top_k_total=42,
                    min_obs=25,
                )
                y_pred = fit_predict_logdelta(
                    train_df=train_df,
                    test_row=test_row,
                    target_col=target_col,
                    feature_cols=selected_features,
                    random_state=random_state,
                )

            elif spec.model_kind == "ratio":
                target_col = ratio_target
                selected_features = select_top_features(
                    train_df=train_df,
                    target_col=target_col,
                    candidate_cols=pool,
                    mandatory_cols=mandatory,
                    top_k_total=42,
                    min_obs=25,
                )
                y_pred = fit_predict_ratio(
                    train_df=train_df,
                    test_row=test_row,
                    target_col=target_col,
                    feature_cols=selected_features,
                    random_state=random_state,
                )

            else:
                raise ValueError(f"Unknown model kind: {spec.model_kind}")

        rows.append({
            "week_start": test_row.iloc[0]["week_start"],
            "model_name": spec.model_name,
            "model_kind": spec.model_kind,
            "horizon_weeks": horizon,
            "current_cases": current_cases,
            "y_true": y_true,
            "y_pred": float(y_pred),
            "abs_error": abs(float(y_true) - float(y_pred)),
            "signed_error": float(y_pred) - float(y_true),
            "n_selected_features": len(selected_features),
            "selected_features": ", ".join(selected_features),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["week_start"] = pd.to_datetime(out["week_start"])
    return out


def compute_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame()

    rows = []
    for (model_name, horizon), grp in pred_df.groupby(["model_name", "horizon_weeks"]):
        y_true = grp["y_true"].to_numpy(dtype=float)
        y_pred = grp["y_pred"].to_numpy(dtype=float)

        rows.append({
            "model_name": model_name,
            "horizon_weeks": int(horizon),
            "n_predictions": len(grp),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mape_pct": mape(y_true, y_pred),
            "smape_pct": smape(y_true, y_pred),
            "wape_pct": wape(y_true, y_pred),
            "mean_true": float(np.mean(y_true)),
            "mean_pred": float(np.mean(y_pred)),
            "mean_n_selected_features": float(grp["n_selected_features"].mean()),
        })

    metrics = pd.DataFrame(rows).sort_values(["horizon_weeks", "rmse", "mae"]).reset_index(drop=True)
    return metrics


def add_improvement_vs_naive(metrics_df: pd.DataFrame) -> pd.DataFrame:
    out = metrics_df.copy()
    out["rmse_improvement_vs_naive_pct"] = np.nan
    out["mae_improvement_vs_naive_pct"] = np.nan

    for h in sorted(out["horizon_weeks"].unique()):
        base = out[(out["horizon_weeks"] == h) & (out["model_name"] == "naive_last_week")]
        if base.empty:
            continue

        base_rmse = float(base["rmse"].iloc[0])
        base_mae = float(base["mae"].iloc[0])

        idx = out["horizon_weeks"] == h
        out.loc[idx, "rmse_improvement_vs_naive_pct"] = (base_rmse - out.loc[idx, "rmse"]) / base_rmse * 100.0
        out.loc[idx, "mae_improvement_vs_naive_pct"] = (base_mae - out.loc[idx, "mae"]) / base_mae * 100.0

    return out


def run_all_advanced_models(
    df: pd.DataFrame,
    min_train_weeks: int = 52,
    evaluation_window: int = 52,
    train_window_weeks: int = 104,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = build_model_specs()
    parts = []

    for spec in specs:
        for horizon in [1, 2, 3]:
            part = run_single_walkforward_advanced(
                df=df,
                spec=spec,
                horizon=horizon,
                min_train_weeks=min_train_weeks,
                evaluation_window=evaluation_window,
                train_window_weeks=train_window_weeks,
                random_state=random_state,
            )
            parts.append(part)

    pred_df = pd.concat(parts, ignore_index=True)
    pred_df = pred_df.sort_values(["horizon_weeks", "model_name", "week_start"]).reset_index(drop=True)

    metrics_df = compute_metrics(pred_df)
    metrics_df = add_improvement_vs_naive(metrics_df)

    return pred_df, metrics_df


def save_outputs(
    root: Path,
    pred_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> tuple[Path, Path]:
    pred_dir = root / "reports" / "predictions"
    tbl_dir = root / "reports" / "tables"
    pred_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    pred_path = pred_dir / "walkforward_predictions_step11_advanced.csv"
    metrics_path = tbl_dir / "walkforward_metrics_step11_advanced.csv"

    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    return pred_path, metrics_path
