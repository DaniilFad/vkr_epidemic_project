from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def add_target_week_start(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["week_start"] = pd.to_datetime(out["week_start"])
    out["target_week_start"] = out["week_start"] + pd.to_timedelta(out["horizon_weeks"] * 7, unit="D")
    return out


def get_best_advanced_models(metrics_df: pd.DataFrame) -> pd.DataFrame:
    df = metrics_df.copy()

    df = df[df["model_name"].str.startswith("advanced_")].copy()
    if df.empty:
        raise ValueError("В metrics_df не найдено advanced-моделей.")

    best = (
        df.sort_values(["horizon_weeks", "rmse", "mae"])
        .groupby("horizon_weeks", as_index=False)
        .first()
        .copy()
    )

    best = best[["horizon_weeks", "model_name", "rmse", "mae", "mape_pct"]].copy()
    best = best.rename(columns={"model_name": "advanced_model_name"})
    best["horizon_weeks"] = best["horizon_weeks"].astype(int)
    return best


def build_comparison_frame(
    step6_predictions: pd.DataFrame,
    step11_predictions: pd.DataFrame,
    step11_metrics: pd.DataFrame,
    horizons=(1, 2, 3),
    window_weeks: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    p6 = add_target_week_start(step6_predictions)
    p11 = add_target_week_start(step11_predictions)

    best_adv = get_best_advanced_models(step11_metrics)

    all_parts = []

    for horizon in horizons:
        adv_row = best_adv[best_adv["horizon_weeks"] == horizon]
        if adv_row.empty:
            raise ValueError(f"Не найдена лучшая advanced-модель для горизонта {horizon}")

        advanced_model_name = adv_row.iloc[0]["advanced_model_name"]

        fact_df = p6[
            (p6["horizon_weeks"] == horizon) &
            (p6["model_name"] == "naive_last_week")
        ][["week_start", "target_week_start", "horizon_weeks", "y_true"]].copy()

        naive_df = p6[
            (p6["horizon_weeks"] == horizon) &
            (p6["model_name"] == "naive_last_week")
        ][["target_week_start", "y_pred"]].copy().rename(columns={"y_pred": "naive_pred"})

        hybrid_df = p6[
            (p6["horizon_weeks"] == horizon) &
            (p6["model_name"] == "hybrid_catboost_log")
        ][["target_week_start", "y_pred"]].copy().rename(columns={"y_pred": "hybrid_pred"})

        advanced_df = p11[
            (p11["horizon_weeks"] == horizon) &
            (p11["model_name"] == advanced_model_name)
        ][["target_week_start", "y_pred"]].copy().rename(columns={"y_pred": "advanced_pred"})

        cur = fact_df.merge(naive_df, on="target_week_start", how="inner")
        cur = cur.merge(hybrid_df, on="target_week_start", how="inner")
        cur = cur.merge(advanced_df, on="target_week_start", how="inner")

        cur["advanced_model_name"] = advanced_model_name
        cur = cur.sort_values("target_week_start").reset_index(drop=True)

        if window_weeks is not None and len(cur) > window_weeks:
            cur = cur.tail(window_weeks).reset_index(drop=True)

        all_parts.append(cur)

    out = pd.concat(all_parts, ignore_index=True)
    out = out.sort_values(["horizon_weeks", "target_week_start"]).reset_index(drop=True)

    return out, best_adv


def save_comparison_table(comparison_df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(out_path, index=False, encoding="utf-8-sig")


def plot_model_comparison_panels(
    comparison_df: pd.DataFrame,
    best_adv_df: pd.DataFrame,
    out_path: Path,
):
    horizons = sorted(comparison_df["horizon_weeks"].unique())

    fig, axes = plt.subplots(
        nrows=len(horizons),
        ncols=1,
        figsize=(16, 13),
        sharex=False,
    )

    if len(horizons) == 1:
        axes = [axes]

    for ax, horizon in zip(axes, horizons):
        part = comparison_df[comparison_df["horizon_weeks"] == horizon].copy()
        part = part.sort_values("target_week_start")

        adv_name = part["advanced_model_name"].iloc[0]

        ax.plot(part["target_week_start"], part["y_true"], linewidth=2.8, label="Факт")
        ax.plot(part["target_week_start"], part["naive_pred"], linewidth=2.0, label="Naive")
        ax.plot(part["target_week_start"], part["hybrid_pred"], linewidth=2.0, label="Hybrid")
        ax.plot(part["target_week_start"], part["advanced_pred"], linewidth=2.0, label=f"Advanced ({adv_name})")

        ax.set_title(f"Горизонт +{horizon} неделя(и): сравнение факт / naive / hybrid / advanced")
        ax.set_xlabel("Целевая неделя")
        ax.set_ylabel("Случаи")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(
        "Сравнение моделей на большом окне walk-forward backtest\n"
        "(все прогнозы выровнены по целевой неделе, а не по моменту прогноза)",
        fontsize=14,
        y=0.995,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_error_comparison_panels(
    comparison_df: pd.DataFrame,
    out_path: Path,
):
    horizons = sorted(comparison_df["horizon_weeks"].unique())

    fig, axes = plt.subplots(
        nrows=len(horizons),
        ncols=1,
        figsize=(16, 13),
        sharex=False,
    )

    if len(horizons) == 1:
        axes = [axes]

    for ax, horizon in zip(axes, horizons):
        part = comparison_df[comparison_df["horizon_weeks"] == horizon].copy()
        part = part.sort_values("target_week_start")

        part["naive_abs_error"] = (part["y_true"] - part["naive_pred"]).abs()
        part["hybrid_abs_error"] = (part["y_true"] - part["hybrid_pred"]).abs()
        part["advanced_abs_error"] = (part["y_true"] - part["advanced_pred"]).abs()

        ax.plot(part["target_week_start"], part["naive_abs_error"], linewidth=2.0, label="|Ошибка| Naive")
        ax.plot(part["target_week_start"], part["hybrid_abs_error"], linewidth=2.0, label="|Ошибка| Hybrid")
        ax.plot(part["target_week_start"], part["advanced_abs_error"], linewidth=2.0, label="|Ошибка| Advanced")

        ax.set_title(f"Горизонт +{horizon}: абсолютные ошибки моделей во времени")
        ax.set_xlabel("Целевая неделя")
        ax.set_ylabel("|Ошибка|")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(
        "Сравнение абсолютных ошибок моделей на большом окне backtest",
        fontsize=14,
        y=0.995,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
