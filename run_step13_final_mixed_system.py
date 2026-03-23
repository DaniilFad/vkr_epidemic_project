from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.models_advanced import (
    build_feature_pool_for_horizon,
    select_top_features,
    fit_predict_level,
    fit_predict_logdelta,
    fit_predict_ratio,
)

ROOT = Path(__file__).resolve().parent

STEP6_METRICS_PATH = ROOT / "reports" / "tables" / "walkforward_metrics_step6_robust.csv"
STEP7_FORECAST_PATH = ROOT / "reports" / "predictions" / "final_forecast_step7.csv"
STEP11_METRICS_PATH = ROOT / "reports" / "tables" / "walkforward_metrics_step11_advanced.csv"
ADVANCED_DATA_PATH = ROOT / "data" / "processed" / "modeling_dataset_advanced.csv"

Z80 = 1.28155
Z95 = 1.95996


def load_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл: {path}")
    return pd.read_csv(path)


def get_old_best_models(step6_metrics: pd.DataFrame) -> pd.DataFrame:
    df = step6_metrics.copy()
    best = (
        df.sort_values(["horizon_weeks", "rmse", "mae"])
        .groupby("horizon_weeks", as_index=False)
        .first()
        .copy()
    )
    best = best.rename(columns={
        "model_name": "old_best_model_name",
        "rmse": "old_best_rmse",
        "mae": "old_best_mae",
        "mape_pct": "old_best_mape_pct",
    })
    cols = [
        "horizon_weeks",
        "old_best_model_name",
        "old_best_rmse",
        "old_best_mae",
        "old_best_mape_pct",
    ]
    return best[cols].copy()


def get_advanced_best_models(step11_metrics: pd.DataFrame) -> pd.DataFrame:
    df = step11_metrics.copy()
    df = df[df["model_name"].str.startswith("advanced_")].copy()

    if df.empty:
        raise ValueError("В step11 metrics не найдены advanced-модели.")

    best = (
        df.sort_values(["horizon_weeks", "rmse", "mae"])
        .groupby("horizon_weeks", as_index=False)
        .first()
        .copy()
    )
    best = best.rename(columns={
        "model_name": "advanced_best_model_name",
        "rmse": "advanced_best_rmse",
        "mae": "advanced_best_mae",
        "mape_pct": "advanced_best_mape_pct",
    })
    cols = [
        "horizon_weeks",
        "advanced_best_model_name",
        "advanced_best_rmse",
        "advanced_best_mae",
        "advanced_best_mape_pct",
    ]
    return best[cols].copy()


def choose_final_models(old_best: pd.DataFrame, adv_best: pd.DataFrame) -> pd.DataFrame:
    merged = old_best.merge(adv_best, on="horizon_weeks", how="inner")

    rows = []
    for _, row in merged.iterrows():
        horizon = int(row["horizon_weeks"])

        old_rmse = float(row["old_best_rmse"])
        adv_rmse = float(row["advanced_best_rmse"])

        if adv_rmse < old_rmse:
            chosen_source = "advanced"
            chosen_model_name = row["advanced_best_model_name"]
            chosen_rmse = float(row["advanced_best_rmse"])
            chosen_mae = float(row["advanced_best_mae"])
            chosen_mape = float(row["advanced_best_mape_pct"])
        else:
            chosen_source = "old"
            chosen_model_name = row["old_best_model_name"]
            chosen_rmse = float(row["old_best_rmse"])
            chosen_mae = float(row["old_best_mae"])
            chosen_mape = float(row["old_best_mape_pct"])

        rows.append({
            "horizon_weeks": horizon,
            "old_best_model_name": row["old_best_model_name"],
            "old_best_rmse": old_rmse,
            "advanced_best_model_name": row["advanced_best_model_name"],
            "advanced_best_rmse": adv_rmse,
            "chosen_source": chosen_source,
            "chosen_model_name": chosen_model_name,
            "chosen_rmse": chosen_rmse,
            "chosen_mae": chosen_mae,
            "chosen_mape_pct": chosen_mape,
        })

    out = pd.DataFrame(rows).sort_values("horizon_weeks").reset_index(drop=True)
    return out


def infer_advanced_model_kind(model_name: str) -> str:
    if model_name == "advanced_level_catboost":
        return "level"
    if model_name == "advanced_logdelta_catboost":
        return "logdelta"
    if model_name == "advanced_ratio_catboost":
        return "ratio"
    raise ValueError(f"Неизвестная advanced-модель: {model_name}")


def get_latest_origin_row(advanced_df: pd.DataFrame) -> pd.DataFrame:
    df = advanced_df.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values("week_start").reset_index(drop=True)

    df_non_null = df[df["cases"].notna()].copy()
    if df_non_null.empty:
        raise ValueError("В advanced dataset нет строк с непустыми cases.")

    latest_row = df_non_null.tail(1).copy()
    return latest_row


def build_advanced_forecast_for_horizon(
    advanced_df: pd.DataFrame,
    horizon: int,
    model_name: str,
) -> tuple[dict, pd.DataFrame]:
    model_kind = infer_advanced_model_kind(model_name)

    df = advanced_df.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values("week_start").reset_index(drop=True)

    df_train = df[df["cases"].notna()].copy().reset_index(drop=True)
    latest_row = df_train.tail(1).copy()

    origin_week = pd.to_datetime(latest_row.iloc[0]["week_start"])
    current_cases = float(latest_row.iloc[0]["cases"])
    target_week = origin_week + pd.Timedelta(days=7 * horizon)

    pool, mandatory, _ = build_feature_pool_for_horizon(df_train, horizon)

    if model_kind == "level":
        target_col = f"target_t_plus_{horizon}"
        top_k = 38
        selected_features = select_top_features(
            train_df=df_train,
            target_col=target_col,
            candidate_cols=pool,
            mandatory_cols=mandatory,
            top_k_total=top_k,
            min_obs=25,
        )
        point_forecast = fit_predict_level(
            train_df=df_train,
            test_row=latest_row,
            target_col=target_col,
            feature_cols=selected_features,
            random_state=42,
        )

    elif model_kind == "logdelta":
        target_col = f"target_logdelta_h{horizon}"
        top_k = 42
        selected_features = select_top_features(
            train_df=df_train,
            target_col=target_col,
            candidate_cols=pool,
            mandatory_cols=mandatory,
            top_k_total=top_k,
            min_obs=25,
        )
        point_forecast = fit_predict_logdelta(
            train_df=df_train,
            test_row=latest_row,
            target_col=target_col,
            feature_cols=selected_features,
            random_state=42,
        )

    elif model_kind == "ratio":
        target_col = f"target_ratio_h{horizon}"
        top_k = 42
        selected_features = select_top_features(
            train_df=df_train,
            target_col=target_col,
            candidate_cols=pool,
            mandatory_cols=mandatory,
            top_k_total=top_k,
            min_obs=25,
        )
        point_forecast = fit_predict_ratio(
            train_df=df_train,
            test_row=latest_row,
            target_col=target_col,
            feature_cols=selected_features,
            random_state=42,
        )

    else:
        raise ValueError(f"Неизвестный model_kind: {model_kind}")

    feat_table = pd.DataFrame({
        "horizon_weeks": [horizon] * len(selected_features),
        "advanced_model_name": [model_name] * len(selected_features),
        "feature_rank": np.arange(1, len(selected_features) + 1),
        "feature_name": selected_features,
    })

    result = {
        "forecast_origin_week": origin_week,
        "forecast_week_start": target_week,
        "horizon_weeks": horizon,
        "model_name": model_name,
        "model_source": "advanced",
        "current_cases": current_cases,
        "point_forecast": float(point_forecast),
    }
    return result, feat_table


def get_old_forecast_for_horizon(
    step7_forecast: pd.DataFrame,
    horizon: int,
    expected_model_name: str,
) -> dict:
    df = step7_forecast.copy()
    df["forecast_origin_week"] = pd.to_datetime(df["forecast_origin_week"])
    df["forecast_week_start"] = pd.to_datetime(df["forecast_week_start"])

    cur = df[df["horizon_weeks"] == horizon].copy()
    if cur.empty:
        raise ValueError(f"В step7 forecast нет горизонта {horizon}")

    cur = cur.sort_values("forecast_origin_week").tail(1).copy()
    row = cur.iloc[0]

    if row["best_model_name"] != expected_model_name:
        raise ValueError(
            f"Ожидалась old-модель {expected_model_name} для горизонта {horizon}, "
            f"но в step7 найдено {row['best_model_name']}"
        )

    return {
        "forecast_origin_week": pd.to_datetime(row["forecast_origin_week"]),
        "forecast_week_start": pd.to_datetime(row["forecast_week_start"]),
        "horizon_weeks": int(row["horizon_weeks"]),
        "model_name": row["best_model_name"],
        "model_source": "old",
        "current_cases": float(row["current_cases"]),
        "point_forecast": float(row["point_forecast"]),
    }


def add_intervals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["lower_80"] = (out["point_forecast"] - Z80 * out["rmse_cv"]).clip(lower=0)
    out["upper_80"] = out["point_forecast"] + Z80 * out["rmse_cv"]

    out["lower_95"] = (out["point_forecast"] - Z95 * out["rmse_cv"]).clip(lower=0)
    out["upper_95"] = out["point_forecast"] + Z95 * out["rmse_cv"]

    return out


def build_alerts(
    final_forecast_df: pd.DataFrame,
    advanced_df: pd.DataFrame,
) -> pd.DataFrame:
    df_hist = advanced_df.copy()
    df_hist["week_start"] = pd.to_datetime(df_hist["week_start"])
    df_hist = df_hist[df_hist["cases"].notna()].copy().sort_values("week_start")

    recent_mean_4 = float(df_hist["cases"].tail(4).mean())
    recent_mean_8 = float(df_hist["cases"].tail(8).mean())

    rows = []
    for _, row in final_forecast_df.iterrows():
        current_cases = float(row["current_cases"])
        forecast_cases = float(row["point_forecast"])

        if current_cases > 0:
            growth_pct = (forecast_cases / current_cases - 1.0) * 100.0
        else:
            growth_pct = np.nan

        high_threshold = max(current_cases * 1.50, recent_mean_4 * 1.40)
        medium_threshold = max(current_cases * 1.15, recent_mean_4 * 1.15)

        if forecast_cases >= high_threshold:
            risk_level = "high"
            reason = "прогноз существенно выше текущего уровня и/или недавнего фона"
        elif forecast_cases >= medium_threshold:
            risk_level = "medium"
            reason = "прогноз показывает заметный рост относительно текущего уровня или недавнего среднего"
        else:
            risk_level = "low"
            reason = "прогноз не показывает выраженного роста относительно текущего уровня и недавнего фона"

        rows.append({
            "forecast_origin_week": row["forecast_origin_week"],
            "forecast_week_start": row["forecast_week_start"],
            "horizon_weeks": int(row["horizon_weeks"]),
            "risk_level": risk_level,
            "current_cases": current_cases,
            "forecast_cases": forecast_cases,
            "forecast_growth_pct_vs_current": growth_pct,
            "recent_mean_4": recent_mean_4,
            "recent_mean_8": recent_mean_8,
            "lower_80": float(row["lower_80"]),
            "upper_80": float(row["upper_80"]),
            "reason": reason,
            "chosen_model_name": row["model_name"],
            "chosen_model_source": row["model_source"],
        })

    out = pd.DataFrame(rows).sort_values("horizon_weeks").reset_index(drop=True)
    return out


def make_final_forecast_plot(
    advanced_df: pd.DataFrame,
    final_forecast_df: pd.DataFrame,
    out_path: Path,
    lookback_weeks: int = 52,
):
    hist = advanced_df.copy()
    hist["week_start"] = pd.to_datetime(hist["week_start"])
    hist = hist[hist["cases"].notna()].copy().sort_values("week_start").reset_index(drop=True)

    origin_week = pd.to_datetime(final_forecast_df["forecast_origin_week"].iloc[0])
    start_week = origin_week - pd.Timedelta(days=7 * lookback_weeks)

    hist_plot = hist[hist["week_start"] >= start_week].copy()

    forecast_plot = final_forecast_df.sort_values("forecast_week_start").copy()

    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(
        hist_plot["week_start"],
        hist_plot["cases"],
        linewidth=2.7,
        label="Фактические случаи",
    )

    ax.axvline(origin_week, linestyle="--", linewidth=2, alpha=0.8, label="Момент финального прогноза")

    x = forecast_plot["forecast_week_start"]
    y = forecast_plot["point_forecast"]

    ax.plot(
        x,
        y,
        marker="o",
        linewidth=2.2,
        label="Финальный mixed-прогноз",
    )

    ax.fill_between(
        x,
        forecast_plot["lower_95"],
        forecast_plot["upper_95"],
        alpha=0.12,
        label="Интервал 95%",
    )

    ax.fill_between(
        x,
        forecast_plot["lower_80"],
        forecast_plot["upper_80"],
        alpha=0.22,
        label="Интервал 80%",
    )

    for _, row in forecast_plot.iterrows():
        ax.annotate(
            f"+{int(row['horizon_weeks'])}\n{row['model_name']}",
            (row["forecast_week_start"], row["point_forecast"]),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=9,
        )

    ax.set_title("Финальный mixed-прогноз заболеваемости COVID-19 по Москве")
    ax.set_xlabel("Неделя")
    ax.set_ylabel("Новые случаи за неделю")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.autofmt_xdate()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    step6_metrics = load_required_csv(STEP6_METRICS_PATH)
    step7_forecast = load_required_csv(STEP7_FORECAST_PATH)
    step11_metrics = load_required_csv(STEP11_METRICS_PATH)
    advanced_df = load_required_csv(ADVANCED_DATA_PATH)

    advanced_df["week_start"] = pd.to_datetime(advanced_df["week_start"])

    old_best = get_old_best_models(step6_metrics)
    adv_best = get_advanced_best_models(step11_metrics)
    final_model_selection = choose_final_models(old_best, adv_best)

    latest_row = get_latest_origin_row(advanced_df)
    latest_origin_week = pd.to_datetime(latest_row.iloc[0]["week_start"])

    old_origin_check = step7_forecast.copy()
    old_origin_check["forecast_origin_week"] = pd.to_datetime(old_origin_check["forecast_origin_week"])
    old_latest_origin_week = pd.to_datetime(old_origin_check["forecast_origin_week"].max())

    if latest_origin_week != old_latest_origin_week:
        raise ValueError(
            f"Не совпадает latest origin week между advanced data ({latest_origin_week.date()}) "
            f"и step7 forecast ({old_latest_origin_week.date()})"
        )

    forecast_rows = []
    feature_tables = []

    for _, row in final_model_selection.iterrows():
        horizon = int(row["horizon_weeks"])
        chosen_source = row["chosen_source"]
        chosen_model_name = row["chosen_model_name"]

        if chosen_source == "old":
            res = get_old_forecast_for_horizon(
                step7_forecast=step7_forecast,
                horizon=horizon,
                expected_model_name=chosen_model_name,
            )
            feature_tables.append(
                pd.DataFrame({
                    "horizon_weeks": [horizon],
                    "advanced_model_name": [np.nan],
                    "feature_rank": [np.nan],
                    "feature_name": [np.nan],
                })
            )

        elif chosen_source == "advanced":
            res, feat_table = build_advanced_forecast_for_horizon(
                advanced_df=advanced_df,
                horizon=horizon,
                model_name=chosen_model_name,
            )
            feature_tables.append(feat_table)

        else:
            raise ValueError(f"Неизвестный chosen_source: {chosen_source}")

        res["rmse_cv"] = float(row["chosen_rmse"])
        res["mae_cv"] = float(row["chosen_mae"])
        res["mape_cv"] = float(row["chosen_mape_pct"])
        res["rmse_improvement_vs_old_best_pct"] = (
            (float(row["old_best_rmse"]) - float(row["chosen_rmse"])) / float(row["old_best_rmse"]) * 100.0
        )
        forecast_rows.append(res)

    final_forecast_df = pd.DataFrame(forecast_rows).sort_values("horizon_weeks").reset_index(drop=True)
    final_forecast_df = add_intervals(final_forecast_df)

    alerts_df = build_alerts(
        final_forecast_df=final_forecast_df,
        advanced_df=advanced_df,
    )

    feature_df = pd.concat(feature_tables, ignore_index=True)
    feature_df = feature_df.dropna(subset=["feature_name"], how="all").reset_index(drop=True)

    tables_dir = ROOT / "reports" / "tables"
    pred_dir = ROOT / "reports" / "predictions"
    fig_dir = ROOT / "reports" / "figures"

    tables_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    selection_path = tables_dir / "final_mixed_model_selection_step13.csv"
    forecast_path = pred_dir / "final_forecast_step13_mixed.csv"
    alerts_path = pred_dir / "alerts_step13_mixed.csv"
    features_path = tables_dir / "advanced_selected_features_step13.csv"
    figure_path = fig_dir / "final_forecast_step13_mixed.png"

    final_model_selection.to_csv(selection_path, index=False, encoding="utf-8-sig")
    final_forecast_df.to_csv(forecast_path, index=False, encoding="utf-8-sig")
    alerts_df.to_csv(alerts_path, index=False, encoding="utf-8-sig")
    feature_df.to_csv(features_path, index=False, encoding="utf-8-sig")

    make_final_forecast_plot(
        advanced_df=advanced_df,
        final_forecast_df=final_forecast_df,
        out_path=figure_path,
        lookback_weeks=52,
    )

    print("=" * 80)
    print("STEP 13 FINAL MIXED SYSTEM RESULT")
    print("=" * 80)
    print("saved model selection :", selection_path)
    print("saved final forecast  :", forecast_path)
    print("saved alerts          :", alerts_path)
    print("saved advanced feats  :", features_path)
    print("saved final figure    :", figure_path)
    print()

    print("=" * 80)
    print("FINAL MODEL SELECTION")
    print("=" * 80)
    print(final_model_selection.to_string(index=False))
    print()

    print("=" * 80)
    print("FINAL FORECAST")
    print("=" * 80)
    print(final_forecast_df.to_string(index=False))
    print()

    print("=" * 80)
    print("ALERTS")
    print("=" * 80)
    print(alerts_df.to_string(index=False))
    print()

    if not feature_df.empty:
        print("=" * 80)
        print("ADVANCED SELECTED FEATURES PREVIEW")
        print("=" * 80)
        print(feature_df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()