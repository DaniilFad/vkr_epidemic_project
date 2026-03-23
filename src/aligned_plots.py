from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def build_aligned_prediction_frame(predictions_df: pd.DataFrame) -> pd.DataFrame:
    df = predictions_df.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["target_week_start"] = df["week_start"] + pd.to_timedelta(df["horizon_weeks"] * 7, unit="D")
    return df


def plot_aligned_best_prediction(
    predictions_df: pd.DataFrame,
    best_models_df: pd.DataFrame,
    horizon: int,
    out_path: Path,
):
    best_row = best_models_df[best_models_df["horizon_weeks"] == horizon].copy()
    if best_row.empty:
        raise ValueError(f"Для горизонта {horizon} нет лучшей модели.")

    model_name = best_row.iloc[0]["model_name"]

    df = build_aligned_prediction_frame(predictions_df)
    part = df[
        (df["horizon_weeks"] == horizon) &
        (df["model_name"] == model_name)
    ].copy()

    if part.empty:
        raise ValueError(f"Нет предсказаний для horizon={horizon}, model={model_name}")

    part = part.sort_values("target_week_start").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(part["target_week_start"], part["y_true"], linewidth=2, label="Факт")
    ax.plot(part["target_week_start"], part["y_pred"], linewidth=2, label="Прогноз")

    ax.set_title(
        f"Сдвинутый walk-forward прогноз: горизонт +{horizon} неделя(и), модель {model_name}"
    )
    ax.set_xlabel("Целевая неделя")
    ax.set_ylabel("Новые случаи за неделю")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_aligned_all_best_predictions(
    predictions_df: pd.DataFrame,
    best_models_df: pd.DataFrame,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    for horizon in sorted(best_models_df["horizon_weeks"].unique()):
        out_path = out_dir / f"aligned_best_prediction_h{int(horizon)}.png"
        plot_aligned_best_prediction(
            predictions_df=predictions_df,
            best_models_df=best_models_df,
            horizon=int(horizon),
            out_path=out_path,
        )


def plot_alignment_explanation_example(
    predictions_df: pd.DataFrame,
    best_models_df: pd.DataFrame,
    horizon: int,
    out_path: Path,
):
    """
    Поясняющий график:
    - серые точки: неделя, в которую прогноз был сделан
    - оранжевые точки: целевая неделя, к которой прогноз относится
    """
    best_row = best_models_df[best_models_df["horizon_weeks"] == horizon].copy()
    if best_row.empty:
        raise ValueError(f"Для горизонта {horizon} нет лучшей модели.")

    model_name = best_row.iloc[0]["model_name"]

    df = build_aligned_prediction_frame(predictions_df)
    part = df[
        (df["horizon_weeks"] == horizon) &
        (df["model_name"] == model_name)
    ].copy()

    if part.empty:
        raise ValueError(f"Нет предсказаний для horizon={horizon}, model={model_name}")

    part = part.sort_values("week_start").reset_index(drop=True).head(12)

    fig, ax = plt.subplots(figsize=(14, 5))

    for _, row in part.iterrows():
        ax.plot(
            [row["week_start"], row["target_week_start"]],
            [row["current_cases"], row["y_pred"]],
            linewidth=1.5,
            alpha=0.7,
        )

    ax.scatter(part["week_start"], part["current_cases"], s=50, label="Момент прогноза / текущий уровень")
    ax.scatter(part["target_week_start"], part["y_pred"], s=50, label="Прогноз для целевой недели")

    ax.set_title(
        f"Как читать walk-forward график для горизонта +{horizon}: от недели прогноза к целевой неделе"
    )
    ax.set_xlabel("Неделя")
    ax.set_ylabel("Новые случаи за неделю")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
