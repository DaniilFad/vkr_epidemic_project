from pathlib import Path
import pandas as pd

from src.models import (
    run_all_models_walkforward,
    add_metric_improvements,
    save_step5_outputs,
)


ROOT = Path(__file__).resolve().parent
FEATURE_PATH = ROOT / "data" / "processed" / "modeling_dataset.csv"

if not FEATURE_PATH.exists():
    raise FileNotFoundError(f"Не найден файл: {FEATURE_PATH}")

feature_df = pd.read_csv(FEATURE_PATH)
feature_df["week_start"] = pd.to_datetime(feature_df["week_start"])

target_cols = [
    "target_t_plus_1",
    "target_t_plus_2",
    "target_t_plus_3",
]

predictions_df, metrics_df, feature_usage_df = run_all_models_walkforward(
    feature_df=feature_df,
    target_cols=target_cols,
    min_train_weeks=80,
    evaluation_window=52,
    random_state=42,
)

metrics_df = add_metric_improvements(metrics_df, baseline_name="naive_last_week")

predictions_path, metrics_path, feature_usage_path = save_step5_outputs(
    ROOT,
    predictions_df,
    metrics_df,
    feature_usage_df,
)

print("=" * 80)
print("STEP 5 RESULT")
print("=" * 80)
print("saved predictions:", predictions_path)
print("saved metrics    :", metrics_path)
print("saved features   :", feature_usage_path)
print()

print("=" * 80)
print("FEATURE SET SIZES")
print("=" * 80)
print(feature_usage_df[["model_name", "model_kind", "n_features"]].to_string(index=False))
print()

print("=" * 80)
print("LEADERBOARD BY HORIZON (sorted by RMSE)")
print("=" * 80)
for horizon in sorted(metrics_df["horizon_weeks"].unique()):
    print(f"\nHORIZON +{horizon} WEEK(S)")
    subset = metrics_df[metrics_df["horizon_weeks"] == horizon].copy()
    subset = subset.sort_values(["rmse", "mae"])
    cols = [
        "model_name",
        "n_predictions",
        "rmse",
        "mae",
        "mape_pct",
        "smape_pct",
        "wape_pct",
        "directional_accuracy_pct",
        "rmse_improvement_vs_naive_pct",
    ]
    print(subset[cols].to_string(index=False))
print()

print("=" * 80)
print("PREDICTION PREVIEW")
print("=" * 80)
print(
    predictions_df[
        ["week_start", "model_name", "horizon_weeks", "current_cases", "y_true", "y_pred", "abs_error"]
    ]
    .head(20)
    .to_string(index=False)
)
print()

print("=" * 80)
print("BEST MODELS SUMMARY")
print("=" * 80)
best_rows = (
    metrics_df.sort_values(["horizon_weeks", "rmse", "mae"])
    .groupby("horizon_weeks", as_index=False)
    .first()
)
print(
    best_rows[
        [
            "horizon_weeks",
            "model_name",
            "rmse",
            "mae",
            "mape_pct",
            "directional_accuracy_pct",
            "rmse_improvement_vs_naive_pct",
        ]
    ].to_string(index=False)
)