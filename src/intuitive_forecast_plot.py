from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def normalize_best_models(best_models_df: pd.DataFrame) -> pd.DataFrame:
    df = best_models_df.copy()

    rename_map = {}
    if "horizon_weeks" not in df.columns and "horizon" in df.columns:
        rename_map["horizon"] = "horizon_weeks"
    if "best_model_name" not in df.columns and "model_name" in df.columns:
        rename_map["model_name"] = "best_model_name"

    if rename_map:
        df = df.rename(columns=rename_map)

    required = {"horizon_weeks", "best_model_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В best_models_df не хватает колонок: {missing}")

    df["horizon_weeks"] = df["horizon_weeks"].astype(int)
    return df


def find_latest_common_origin(
    predictions_df: pd.DataFrame,
    best_models_df: pd.DataFrame,
    horizons=(1, 2, 3),
) -> pd.Timestamp:
    pred = predictions_df.copy()
    pred["week_start"] = pd.to_datetime(pred["week_start"])

    best = normalize_best_models(best_models_df)

    valid_origin_sets = []

    for h in horizons:
        row = best[best["horizon_weeks"] == h]
        if row.empty:
            raise ValueError(f"В best_models_df нет лучшей модели для горизонта {h}")

        model_name = row.iloc[0]["best_model_name"]

        cur = pred[
            (pred["horizon_weeks"] == h) &
            (pred["model_name"] == model_name)
        ].copy()

        if cur.empty:
            raise ValueError(f"Нет предсказаний для горизонта {h} и модели {model_name}")

        valid_origin_sets.append(set(cur["week_start"]))

    common = set.intersection(*valid_origin_sets)
    if not common:
        raise ValueError("Не найдено общей forecast origin week для всех горизонтов 1/2/3.")

    return max(common)


def build_single_origin_comparison(
    feature_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    best_models_df: pd.DataFrame,
    origin_week: pd.Timestamp,
) -> pd.DataFrame:
    feature_df = feature_df.copy()
    feature_df["week_start"] = pd.to_datetime(feature_df["week_start"])

    pred = predictions_df.copy()
    pred["week_start"] = pd.to_datetime(pred["week_start"])

    best = normalize_best_models(best_models_df)

    rows = []
    for h in [1, 2, 3]:
        model_name = best.loc[best["horizon_weeks"] == h, "best_model_name"].iloc[0]

        cur = pred[
            (pred["week_start"] == origin_week) &
            (pred["horizon_weeks"] == h) &
            (pred["model_name"] == model_name)
        ].copy()

        if cur.empty:
            raise ValueError(
                f"Не найдено предсказание для origin={origin_week.date()}, horizon={h}, model={model_name}"
            )

        r = cur.iloc[0]
        target_week = origin_week + pd.Timedelta(days=7 * h)

        rows.append({
            "forecast_origin_week": origin_week,
            "target_week_start": target_week,
            "horizon_weeks": h,
            "model_name": model_name,
            "current_cases_at_origin": float(r["current_cases"]),
            "actual_cases": float(r["y_true"]),
            "predicted_cases": float(r["y_pred"]),
            "abs_error": float(r["abs_error"]),
            "signed_error": float(r["signed_error"]),
        })

    out = pd.DataFrame(rows).sort_values("horizon_weeks").reset_index(drop=True)
    return out


def plot_single_origin_forecast(
    feature_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    out_path: Path,
    lookback_weeks: int = 16,
):
    feature_df = feature_df.copy()
    feature_df["week_start"] = pd.to_datetime(feature_df["week_start"])

    origin_week = pd.to_datetime(comparison_df["forecast_origin_week"].iloc[0])
    end_week = pd.to_datetime(comparison_df["target_week_start"].max())
    start_week = origin_week - pd.Timedelta(days=7 * lookback_weeks)

    plot_df = feature_df[
        (feature_df["week_start"] >= start_week) &
        (feature_df["week_start"] <= end_week)
    ].copy().sort_values("week_start")

    hist_df = plot_df[plot_df["week_start"] <= origin_week].copy()
    fut_df = plot_df[plot_df["week_start"] > origin_week].copy()

    fig, ax = plt.subplots(figsize=(15, 7))

    # обучающая история
    ax.plot(
        hist_df["week_start"],
        hist_df["cases"],
        linewidth=2.5,
        label="История (доступна модели на момент прогноза)",
    )

    # реальные будущие значения
    if not fut_df.empty:
        ax.plot(
            fut_df["week_start"],
            fut_df["cases"],
            linewidth=2.5,
            linestyle="--",
            label="Реальные будущие значения",
        )

    # вертикальная линия прогноза
    ax.axvline(origin_week, linestyle="--", linewidth=2, alpha=0.8, label="Момент прогноза")

    # мягкая заливка обучающей области
    ax.axvspan(start_week, origin_week, alpha=0.08)

    # прогнозные точки
    ax.plot(
        comparison_df["target_week_start"],
        comparison_df["predicted_cases"],
        marker="o",
        linewidth=2,
        label="Прогноз +1/+2/+3 недели",
    )

    # реальные точки в тех же целевых датах
    ax.scatter(
        comparison_df["target_week_start"],
        comparison_df["actual_cases"],
        s=90,
        zorder=5,
        label="Факт в целевых точках",
    )

    # подписи горизонтов
    for _, row in comparison_df.iterrows():
        ax.annotate(
            f"+{int(row['horizon_weeks'])}",
            (row["target_week_start"], row["predicted_cases"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    model_text = "\n".join(
        [
            f"+{int(r.horizon_weeks)}: {r.model_name}"
            for r in comparison_df.itertuples()
        ]
    )

    text_x = comparison_df["target_week_start"].min()
    text_y = max(
        comparison_df["actual_cases"].max(),
        comparison_df["predicted_cases"].max(),
        hist_df["cases"].max() if not hist_df.empty else 0,
    )

    ax.text(
        text_x,
        text_y * 1.02,
        f"Лучшие модели по горизонтам:\n{model_text}",
        fontsize=10,
        va="bottom",
    )

    ax.set_title("Интуитивный график single-origin прогноза: история → момент прогноза → факт и прогноз вперёд")
    ax.set_xlabel("Неделя")
    ax.set_ylabel("Новые случаи за неделю")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_single_origin_table(comparison_df: pd.DataFrame, out_path: Path):
    comparison_df.to_csv(out_path, index=False, encoding="utf-8-sig")
