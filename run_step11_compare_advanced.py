from pathlib import Path
import pandas as pd

from src.models_advanced import run_all_advanced_models, save_outputs


ROOT = Path(__file__).resolve().parent
ADVANCED_PATH = ROOT / "data" / "processed" / "modeling_dataset_advanced.csv"
OLD_METRICS_PATH = ROOT / "reports" / "tables" / "walkforward_metrics_step6_robust.csv"

if not ADVANCED_PATH.exists():
    raise FileNotFoundError(f"Не найден файл: {ADVANCED_PATH}")

df = pd.read_csv(ADVANCED_PATH)
df["week_start"] = pd.to_datetime(df["week_start"])

# тот же рабочий период, что и раньше
df = df[df["week_start"] >= pd.Timestamp("2022-01-03")].copy()
df = df.sort_values("week_start").reset_index(drop=True)

pred_df, metrics_df = run_all_advanced_models(
    df=df,
    min_train_weeks=52,
    evaluation_window=52,
    train_window_weeks=104,
    random_state=42,
)

pred_path, metrics_path = save_outputs(ROOT, pred_df, metrics_df)

print("=" * 80)
print("STEP 11 RESULT")
print("=" * 80)
print("saved predictions:", pred_path)
print("saved metrics    :", metrics_path)
print()

print("=" * 80)
print("ADVANCED LEADERBOARD BY HORIZON")
print("=" * 80)
for horizon in sorted(metrics_df["horizon_weeks"].unique()):
    print(f"\nHORIZON +{horizon}")
    part = metrics_df[metrics_df["horizon_weeks"] == horizon].copy().sort_values(["rmse", "mae"])
    cols = [
        "model_name",
        "n_predictions",
        "rmse",
        "mae",
        "mape_pct",
        "smape_pct",
        "wape_pct",
        "mean_n_selected_features",
        "rmse_improvement_vs_naive_pct",
    ]
    print(part[cols].to_string(index=False))

print()

if OLD_METRICS_PATH.exists():
    old_metrics = pd.read_csv(OLD_METRICS_PATH)

    old_best = (
        old_metrics.sort_values(["horizon_weeks", "rmse", "mae"])
        .groupby("horizon_weeks", as_index=False)
        .first()
        .copy()
    )
    old_best = old_best.rename(columns={
        "model_name": "old_best_model",
        "rmse": "old_best_rmse",
        "mae": "old_best_mae",
        "mape_pct": "old_best_mape_pct",
    })

    new_best = (
        metrics_df.sort_values(["horizon_weeks", "rmse", "mae"])
        .groupby("horizon_weeks", as_index=False)
        .first()
        .copy()
    )
    new_best = new_best.rename(columns={
        "model_name": "new_best_model",
        "rmse": "new_best_rmse",
        "mae": "new_best_mae",
        "mape_pct": "new_best_mape_pct",
    })

    compare = old_best.merge(new_best, on="horizon_weeks", how="outer")
    compare["rmse_gain_vs_old_best_pct"] = (
        (compare["old_best_rmse"] - compare["new_best_rmse"]) / compare["old_best_rmse"] * 100.0
    )

    print("=" * 80)
    print("COMPARISON VS STEP 6 ROBUST BEST")
    print("=" * 80)
    cols = [
        "horizon_weeks",
        "old_best_model", "old_best_rmse",
        "new_best_model", "new_best_rmse",
        "rmse_gain_vs_old_best_pct",
    ]
    print(compare[cols].to_string(index=False))
    print()

print("=" * 80)
print("PREDICTION PREVIEW")
print("=" * 80)
print(
    pred_df[
        ["week_start", "model_name", "horizon_weeks", "current_cases", "y_true", "y_pred", "abs_error"]
    ]
    .head(24)
    .to_string(index=False)
)
