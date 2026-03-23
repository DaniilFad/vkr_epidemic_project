from pathlib import Path
import pandas as pd

from src.xai_plots import (
    plot_forecast,
    plot_rmse_comparison,
    plot_best_predictions,
    fit_best_models_and_generate_xai,
)

ROOT = Path(__file__).resolve().parent

FEATURE_PATH = ROOT / "data" / "processed" / "modeling_dataset.csv"
METRICS_PATH = ROOT / "reports" / "tables" / "walkforward_metrics_step6_robust.csv"
PREDICTIONS_PATH = ROOT / "reports" / "predictions" / "walkforward_predictions_step6_robust.csv"
FORECAST_PATH = ROOT / "reports" / "predictions" / "final_forecast_step7.csv"
BEST_MODELS_PATH = ROOT / "reports" / "tables" / "best_models_step7.csv"

for path in [FEATURE_PATH, METRICS_PATH, PREDICTIONS_PATH, FORECAST_PATH, BEST_MODELS_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл: {path}")

feature_df = pd.read_csv(FEATURE_PATH)
feature_df["week_start"] = pd.to_datetime(feature_df["week_start"])

metrics_df = pd.read_csv(METRICS_PATH)
predictions_df = pd.read_csv(PREDICTIONS_PATH)
predictions_df["week_start"] = pd.to_datetime(predictions_df["week_start"])

forecast_df = pd.read_csv(FORECAST_PATH)
forecast_df["forecast_origin_week"] = pd.to_datetime(forecast_df["forecast_origin_week"])
forecast_df["forecast_week_start"] = pd.to_datetime(forecast_df["forecast_week_start"])

best_models_df = pd.read_csv(BEST_MODELS_PATH)

# тот же рабочий период, что и в robust-валидации
feature_df = feature_df[feature_df["week_start"] >= pd.Timestamp("2022-01-03")].copy()
feature_df = feature_df.sort_values("week_start").reset_index(drop=True)

figures_dir = ROOT / "reports" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

forecast_fig_path = figures_dir / "forecast_overview_step8.png"
rmse_fig_path = figures_dir / "rmse_comparison_step8.png"

plot_forecast(
    feature_df=feature_df,
    forecast_df=forecast_df,
    out_path=forecast_fig_path,
    lookback_weeks=52,
)

plot_rmse_comparison(
    metrics_df=metrics_df,
    out_path=rmse_fig_path,
)

plot_best_predictions(
    predictions_df=predictions_df,
    best_models_df=best_models_df,
    out_dir=figures_dir,
)

xai_summary_df, xai_summary_path = fit_best_models_and_generate_xai(
    root=ROOT,
    feature_df=feature_df,
    best_models_df=best_models_df,
    random_state=42,
)

print("=" * 80)
print("STEP 8 RESULT")
print("=" * 80)
print("saved figure:", forecast_fig_path)
print("saved figure:", rmse_fig_path)
print("saved xai summary:", xai_summary_path)
print()

print("=" * 80)
print("XAI SUMMARY")
print("=" * 80)
if xai_summary_df.empty:
    print("XAI summary is empty")
else:
    print(xai_summary_df.to_string(index=False))
print()

print("=" * 80)
print("GENERATED FIGURES")
print("=" * 80)
for path in sorted(figures_dir.glob("*step8*.png")):
    print(path)
for path in sorted(figures_dir.glob("best_prediction_h*_step8.png")):
    print(path)
