from pathlib import Path
import pandas as pd

from src.final_forecast import (
    forecast_with_best_models,
    build_alerts,
    save_step7_outputs,
)

ROOT = Path(__file__).resolve().parent

FEATURE_PATH = ROOT / "data" / "processed" / "modeling_dataset.csv"
METRICS_PATH = ROOT / "reports" / "tables" / "walkforward_metrics_step6_robust.csv"
PREDICTIONS_PATH = ROOT / "reports" / "predictions" / "walkforward_predictions_step6_robust.csv"

for path in [FEATURE_PATH, METRICS_PATH, PREDICTIONS_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл: {path}")

feature_df = pd.read_csv(FEATURE_PATH)
feature_df["week_start"] = pd.to_datetime(feature_df["week_start"])

metrics_df = pd.read_csv(METRICS_PATH)
walkforward_predictions_df = pd.read_csv(PREDICTIONS_PATH)
walkforward_predictions_df["week_start"] = pd.to_datetime(walkforward_predictions_df["week_start"])

# Та же логика, что и в robust-валидации
feature_df = feature_df[feature_df["week_start"] >= pd.Timestamp("2022-01-03")].copy()
feature_df = feature_df.sort_values("week_start").reset_index(drop=True)

forecast_df, best_models_df = forecast_with_best_models(
    feature_df=feature_df,
    metrics_df=metrics_df,
    walkforward_predictions_df=walkforward_predictions_df,
    random_state=42,
)

alerts_df = build_alerts(forecast_df, feature_df)

forecast_path, alerts_path, best_models_path = save_step7_outputs(
    ROOT,
    forecast_df,
    alerts_df,
    best_models_df,
)

print("=" * 80)
print("STEP 7 RESULT")
print("=" * 80)
print("saved forecast   :", forecast_path)
print("saved alerts     :", alerts_path)
print("saved best models:", best_models_path)
print()

print("=" * 80)
print("BEST MODELS USED FOR FINAL FORECAST")
print("=" * 80)
print(best_models_df.to_string(index=False))
print()

print("=" * 80)
print("FINAL FORECAST")
print("=" * 80)
print(forecast_df.to_string(index=False))
print()

print("=" * 80)
print("ALERTS")
print("=" * 80)
print(alerts_df.to_string(index=False))
