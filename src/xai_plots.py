from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import Pool

from src.models import build_model_specs, make_catboost_regressor


def get_latest_observed_row(feature_df: pd.DataFrame) -> pd.DataFrame:
    observed = feature_df[feature_df["cases"].notna()].copy()
    if observed.empty:
        raise ValueError("В feature_df нет строк с наблюдаемыми cases.")
    observed = observed.sort_values("week_start").reset_index(drop=True)
    return observed.iloc[[-1]].copy()


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


def fit_catboost_log_model(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    random_state: int = 42,
):
    X_train, y_train, medians = prepare_train_xy(feature_df, feature_cols, target_col)

    model = make_catboost_regressor(random_state=random_state)
    model.fit(X_train, np.log1p(y_train))

    return model, X_train, y_train, medians


def predict_catboost_log(model, row_df: pd.DataFrame, feature_cols: list[str], medians: pd.Series) -> tuple[float, float]:
    X_row = row_df[feature_cols].copy().fillna(medians)
    pred_log = float(model.predict(X_row)[0])
    pred = float(np.expm1(pred_log))
    pred = max(0.0, pred)
    return pred_log, pred


def compute_global_importance(model, feature_cols: list[str]) -> pd.DataFrame:
    importances = model.get_feature_importance(type="PredictionValuesChange")
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    })
    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return imp_df


def compute_local_shap(model, row_df: pd.DataFrame, feature_cols: list[str], medians: pd.Series) -> pd.DataFrame:
    X_row = row_df[feature_cols].copy().fillna(medians)
    pool = Pool(X_row)

    shap_values = model.get_feature_importance(pool, type="ShapValues")
    # shape: (1, n_features + 1), last column = expected value
    shap_row = shap_values[0]
    feature_shap = shap_row[:-1]
    expected_value = float(shap_row[-1])

    local_df = pd.DataFrame({
        "feature": feature_cols,
        "shap_value_log": feature_shap,
        "abs_shap_value_log": np.abs(feature_shap),
        "feature_value": X_row.iloc[0].values,
    }).sort_values("abs_shap_value_log", ascending=False).reset_index(drop=True)

    raw_pred_log = expected_value + feature_shap.sum()
    pred_cases = float(np.expm1(raw_pred_log))
    pred_cases = max(0.0, pred_cases)

    local_df["expected_value_log"] = expected_value
    local_df["prediction_log"] = raw_pred_log
    local_df["prediction_cases"] = pred_cases

    return local_df


def save_xai_tables(
    root: Path,
    horizon: int,
    global_df: pd.DataFrame,
    local_df: pd.DataFrame,
):
    tables_dir = root / "reports" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    global_path = tables_dir / f"global_importance_h{horizon}_step8.csv"
    local_path = tables_dir / f"local_shap_h{horizon}_step8.csv"

    global_df.to_csv(global_path, index=False, encoding="utf-8-sig")
    local_df.to_csv(local_path, index=False, encoding="utf-8-sig")

    return global_path, local_path


def plot_forecast(feature_df: pd.DataFrame, forecast_df: pd.DataFrame, out_path: Path, lookback_weeks: int = 52):
    observed = feature_df[feature_df["cases"].notna()].copy().sort_values("week_start")
    tail = observed.tail(lookback_weeks).copy()

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(tail["week_start"], tail["cases"], linewidth=2, label="Фактические случаи")

    ax.plot(
        forecast_df["forecast_week_start"],
        forecast_df["point_forecast"],
        marker="o",
        linewidth=2,
        label="Прогноз",
    )

    if {"lower_80", "upper_80"}.issubset(forecast_df.columns):
        ax.fill_between(
            forecast_df["forecast_week_start"],
            forecast_df["lower_80"],
            forecast_df["upper_80"],
            alpha=0.2,
            label="Интервал 80%",
        )

    if {"lower_95", "upper_95"}.issubset(forecast_df.columns):
        ax.fill_between(
            forecast_df["forecast_week_start"],
            forecast_df["lower_95"],
            forecast_df["upper_95"],
            alpha=0.1,
            label="Интервал 95%",
        )

    ax.set_title("Краткосрочный прогноз заболеваемости COVID-19 по Москве")
    ax.set_xlabel("Неделя")
    ax.set_ylabel("Новые случаи за неделю")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_rmse_comparison(metrics_df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(12, 7))

    for model_name in metrics_df["model_name"].unique():
        part = metrics_df[metrics_df["model_name"] == model_name].sort_values("horizon_weeks")
        ax.plot(
            part["horizon_weeks"],
            part["rmse"],
            marker="o",
            linewidth=2,
            label=model_name,
        )

    ax.set_title("Сравнение моделей по RMSE на разных горизонтах")
    ax.set_xlabel("Горизонт прогноза, недели")
    ax.set_ylabel("RMSE")
    ax.set_xticks(sorted(metrics_df["horizon_weeks"].unique()))
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_best_predictions(
    predictions_df: pd.DataFrame,
    best_models_df: pd.DataFrame,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in best_models_df.iterrows():
        horizon = int(row["horizon_weeks"])
        model_name = row["model_name"]

        part = predictions_df[
            (predictions_df["horizon_weeks"] == horizon) &
            (predictions_df["model_name"] == model_name)
        ].copy().sort_values("week_start")

        if part.empty:
            continue

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(part["week_start"], part["y_true"], linewidth=2, label="Факт")
        ax.plot(part["week_start"], part["y_pred"], linewidth=2, label="Прогноз")

        ax.set_title(f"Лучший walk-forward прогноз: горизонт +{horizon} неделя(и), модель {model_name}")
        ax.set_xlabel("Неделя")
        ax.set_ylabel("Новые случаи за неделю")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_dir / f"best_prediction_h{horizon}_step8.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_global_importance(global_df: pd.DataFrame, horizon: int, out_path: Path, top_n: int = 15):
    top = global_df.head(top_n).copy()
    top = top.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"], top["importance"])
    ax.set_title(f"Global importance признаков, горизонт +{horizon}")
    ax.set_xlabel("Важность")
    ax.set_ylabel("Признак")
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_local_shap(local_df: pd.DataFrame, horizon: int, out_path: Path, top_n: int = 15):
    top = local_df.head(top_n).copy()
    top = top.sort_values("shap_value_log", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"], top["shap_value_log"])
    ax.set_title(f"Local explanation прогноза, горизонт +{horizon}")
    ax.set_xlabel("SHAP вклад в log-прогноз")
    ax.set_ylabel("Признак")
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fit_best_models_and_generate_xai(
    root: Path,
    feature_df: pd.DataFrame,
    best_models_df: pd.DataFrame,
    random_state: int = 42,
):
    figures_dir = root / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    specs = {spec.model_name: spec for spec in build_model_specs(feature_df)}
    latest_row = get_latest_observed_row(feature_df)

    xai_summary_rows = []

    for _, row in best_models_df.iterrows():
        horizon = int(row["horizon_weeks"])
        model_name = row["model_name"]
        target_col = f"target_t_plus_{horizon}"

        spec = specs[model_name]

        if spec.model_kind != "catboost_log":
            continue

        model, X_train, y_train, medians = fit_catboost_log_model(
            feature_df=feature_df,
            feature_cols=spec.feature_cols,
            target_col=target_col,
            random_state=random_state,
        )

        pred_log, pred_cases = predict_catboost_log(
            model=model,
            row_df=latest_row,
            feature_cols=spec.feature_cols,
            medians=medians,
        )

        global_df = compute_global_importance(model, spec.feature_cols)
        local_df = compute_local_shap(model, latest_row, spec.feature_cols, medians)

        global_path, local_path = save_xai_tables(
            root=root,
            horizon=horizon,
            global_df=global_df,
            local_df=local_df,
        )

        plot_global_importance(
            global_df=global_df,
            horizon=horizon,
            out_path=figures_dir / f"global_importance_h{horizon}_step8.png",
            top_n=15,
        )

        plot_local_shap(
            local_df=local_df,
            horizon=horizon,
            out_path=figures_dir / f"local_shap_h{horizon}_step8.png",
            top_n=15,
        )

        top_positive = local_df.sort_values("shap_value_log", ascending=False).head(5)["feature"].tolist()
        top_negative = local_df.sort_values("shap_value_log", ascending=True).head(5)["feature"].tolist()

        xai_summary_rows.append({
            "horizon_weeks": horizon,
            "model_name": model_name,
            "predicted_cases": round(pred_cases, 3),
            "predicted_log": round(pred_log, 6),
            "global_importance_path": str(global_path),
            "local_shap_path": str(local_path),
            "top_positive_features": ", ".join(top_positive),
            "top_negative_features": ", ".join(top_negative),
        })

    xai_summary_df = pd.DataFrame(xai_summary_rows)
    summary_path = root / "reports" / "tables" / "xai_summary_step8.csv"
    xai_summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    return xai_summary_df, summary_path
