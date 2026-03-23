from pathlib import Path
import numpy as np
import pandas as pd

from src.models import build_model_specs, make_catboost_regressor


def prepare_train_xy(train_df: pd.DataFrame, feature_cols: list[str], target_col: str):
    train = train_df.copy()
    train = train.dropna(subset=[target_col]).copy()

    X = train[feature_cols].copy()
    y = train[target_col].astype(float).copy()

    medians = X.median(numeric_only=True)
    X = X.fillna(medians)

    mask = y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()

    return X, y, medians


def train_and_predict_log_model(
    train_df: pd.DataFrame,
    predict_row: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    random_state: int = 42,
) -> float:
    X_train, y_train, medians = prepare_train_xy(train_df, feature_cols, target_col)

    X_pred = predict_row[feature_cols].copy()
    X_pred = X_pred.fillna(medians)

    model = make_catboost_regressor(random_state=random_state)
    model.fit(X_train, np.log1p(y_train))

    pred_log = float(model.predict(X_pred)[0])
    pred = float(np.expm1(pred_log))
    return max(0.0, pred)


def get_best_models_by_horizon(metrics_df: pd.DataFrame) -> pd.DataFrame:
    best = (
        metrics_df.sort_values(["horizon_weeks", "rmse", "mae"])
        .groupby("horizon_weeks", as_index=False)
        .first()
    )
    return best[["horizon_weeks", "model_name", "rmse", "mae", "mape_pct", "rmse_improvement_vs_naive_pct"]].copy()


def empirical_interval_from_residuals(
    residuals: pd.Series,
    point_forecast: float,
    alpha: float = 0.20,
) -> tuple[float, float]:
    """
    residual = y_true - y_pred
    interval: point + quantiles(residual)
    """
    residuals = residuals.dropna().astype(float)
    if len(residuals) < 10:
        return np.nan, np.nan

    q_low = residuals.quantile(alpha / 2.0)
    q_high = residuals.quantile(1.0 - alpha / 2.0)

    lower = max(0.0, point_forecast + q_low)
    upper = max(0.0, point_forecast + q_high)
    return float(lower), float(upper)


def get_latest_observed_row(feature_df: pd.DataFrame) -> pd.DataFrame:
    observed = feature_df[feature_df["cases"].notna()].copy()
    if observed.empty:
        raise ValueError("В feature_df нет строк с наблюдаемыми cases.")
    observed = observed.sort_values("week_start").reset_index(drop=True)
    return observed.iloc[[-1]].copy()


def forecast_with_best_models(
    feature_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    walkforward_predictions_df: pd.DataFrame,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_df = feature_df.sort_values("week_start").reset_index(drop=True).copy()

    # Берём последнюю реально наблюдаемую неделю по cases
    latest_row = get_latest_observed_row(feature_df)
    last_week = pd.Timestamp(latest_row.iloc[0]["week_start"])
    current_cases = float(latest_row.iloc[0]["cases"])

    best_models = get_best_models_by_horizon(metrics_df)
    specs = {spec.model_name: spec for spec in build_model_specs(feature_df)}

    forecast_rows = []

    for _, best_row in best_models.iterrows():
        horizon = int(best_row["horizon_weeks"])
        model_name = best_row["model_name"]
        target_col = f"target_t_plus_{horizon}"

        spec = specs[model_name]

        # Обучаем на всей доступной истории с наблюдаемым target
        train_df = feature_df.copy()

        if spec.model_kind == "naive_last_week":
            point_forecast = current_cases

        elif spec.model_kind == "seasonal_naive_52":
            observed = feature_df[feature_df["cases"].notna()].copy().sort_values("week_start").reset_index(drop=True)
            if len(observed) >= 53 and pd.notna(observed.iloc[-53]["cases"]):
                point_forecast = float(observed.iloc[-53]["cases"])
            else:
                point_forecast = current_cases

        elif spec.model_kind == "catboost_log":
            point_forecast = train_and_predict_log_model(
                train_df=train_df,
                predict_row=latest_row,
                feature_cols=spec.feature_cols,
                target_col=target_col,
                random_state=random_state,
            )

        else:
            raise ValueError(f"Unknown model kind: {spec.model_kind}")

        pred_hist = walkforward_predictions_df[
            (walkforward_predictions_df["model_name"] == model_name) &
            (walkforward_predictions_df["horizon_weeks"] == horizon)
        ].copy()

        pred_hist["residual"] = pred_hist["y_true"] - pred_hist["y_pred"]

        low80, high80 = empirical_interval_from_residuals(pred_hist["residual"], point_forecast, alpha=0.20)
        low95, high95 = empirical_interval_from_residuals(pred_hist["residual"], point_forecast, alpha=0.05)

        forecast_rows.append({
            "forecast_origin_week": last_week,
            "forecast_week_start": last_week + pd.Timedelta(days=7 * horizon),
            "horizon_weeks": horizon,
            "best_model_name": model_name,
            "current_cases": round(current_cases, 3),
            "point_forecast": round(point_forecast, 3),
            "lower_80": round(low80, 3) if pd.notna(low80) else np.nan,
            "upper_80": round(high80, 3) if pd.notna(high80) else np.nan,
            "lower_95": round(low95, 3) if pd.notna(low95) else np.nan,
            "upper_95": round(high95, 3) if pd.notna(high95) else np.nan,
            "rmse_cv": round(float(best_row["rmse"]), 3),
            "mae_cv": round(float(best_row["mae"]), 3),
            "mape_cv": round(float(best_row["mape_pct"]), 3),
            "rmse_improvement_vs_naive_pct": round(float(best_row["rmse_improvement_vs_naive_pct"]), 3),
        })

    forecast_df = pd.DataFrame(forecast_rows).sort_values("horizon_weeks").reset_index(drop=True)
    best_models = best_models.sort_values("horizon_weeks").reset_index(drop=True)

    return forecast_df, best_models


def assign_alert_level(
    point_forecast: float,
    lower_80: float,
    current_cases: float,
    recent_mean_4: float,
    recent_mean_8: float,
    recent_std_8: float,
    recent_max_8: float,
) -> tuple[str, str]:
    growth_pct = 100.0 * (point_forecast - current_cases) / current_cases if current_cases > 0 else np.nan

    high_condition = (
        ((point_forecast >= current_cases * 1.25) and (point_forecast >= recent_mean_8 + recent_std_8))
        or (pd.notna(lower_80) and lower_80 > recent_mean_8)
        or (point_forecast > recent_max_8 * 1.05)
    )

    medium_condition = (
        (point_forecast >= current_cases * 1.10)
        or (point_forecast > recent_mean_8)
        or ((pd.notna(growth_pct)) and (growth_pct >= 10.0))
    )

    if high_condition:
        return "high", "прогноз существенно выше текущего уровня и/или устойчиво выше недавнего фона"
    if medium_condition:
        return "medium", "прогноз показывает заметный рост относительно текущего уровня или недавнего среднего"
    return "low", "прогноз не показывает выраженного роста относительно текущего уровня и недавнего фона"


def build_alerts(forecast_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    feature_df = feature_df.sort_values("week_start").reset_index(drop=True).copy()
    observed = feature_df[feature_df["cases"].notna()].copy().sort_values("week_start").reset_index(drop=True)

    recent_4 = observed["cases"].tail(4)
    recent_8 = observed["cases"].tail(8)

    current_cases = float(observed["cases"].iloc[-1])
    recent_mean_4 = float(recent_4.mean())
    recent_mean_8 = float(recent_8.mean())
    recent_std_8 = float(recent_8.std(ddof=0)) if len(recent_8) > 1 else 0.0
    recent_max_8 = float(recent_8.max())

    rows = []
    for _, row in forecast_df.iterrows():
        level, reason = assign_alert_level(
            point_forecast=float(row["point_forecast"]),
            lower_80=float(row["lower_80"]) if pd.notna(row["lower_80"]) else np.nan,
            current_cases=current_cases,
            recent_mean_4=recent_mean_4,
            recent_mean_8=recent_mean_8,
            recent_std_8=recent_std_8,
            recent_max_8=recent_max_8,
        )

        growth_pct = 100.0 * (float(row["point_forecast"]) - current_cases) / current_cases if current_cases > 0 else np.nan

        rows.append({
            "forecast_origin_week": row["forecast_origin_week"],
            "forecast_week_start": row["forecast_week_start"],
            "horizon_weeks": int(row["horizon_weeks"]),
            "risk_level": level,
            "current_cases": round(current_cases, 3),
            "forecast_cases": round(float(row["point_forecast"]), 3),
            "forecast_growth_pct_vs_current": round(growth_pct, 3) if pd.notna(growth_pct) else np.nan,
            "lower_80": row["lower_80"],
            "upper_80": row["upper_80"],
            "reason": reason,
            "best_model_name": row["best_model_name"],
        })

    return pd.DataFrame(rows).sort_values("horizon_weeks").reset_index(drop=True)


def save_step7_outputs(
    root: Path,
    forecast_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    best_models_df: pd.DataFrame,
) -> tuple[Path, Path, Path]:
    predictions_dir = root / "reports" / "predictions"
    tables_dir = root / "reports" / "tables"

    predictions_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    forecast_path = predictions_dir / "final_forecast_step7.csv"
    alerts_path = predictions_dir / "alerts_step7.csv"
    best_models_path = tables_dir / "best_models_step7.csv"

    forecast_df.to_csv(forecast_path, index=False, encoding="utf-8-sig")
    alerts_df.to_csv(alerts_path, index=False, encoding="utf-8-sig")
    best_models_df.to_csv(best_models_path, index=False, encoding="utf-8-sig")

    return forecast_path, alerts_path, best_models_path
