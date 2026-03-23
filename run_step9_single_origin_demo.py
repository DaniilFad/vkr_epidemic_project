from pathlib import Path
import pandas as pd

from src.intuitive_forecast_plot import (
    find_latest_common_origin,
    build_single_origin_comparison,
    plot_single_origin_forecast,
    save_single_origin_table,
)

ROOT = Path(__file__).resolve().parent

FEATURE_PATH = ROOT / "data" / "processed" / "modeling_dataset.csv"
PREDICTIONS_PATH = ROOT / "reports" / "predictions" / "walkforward_predictions_step6_robust.csv"
BEST_MODELS_PATH = ROOT / "reports" / "tables" / "best_models_step7.csv"

for path in [FEATURE_PATH, PREDICTIONS_PATH, BEST_MODELS_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл: {path}")

feature_df = pd.read_csv(FEATURE_PATH)
feature_df["week_start"] = pd.to_datetime(feature_df["week_start"])

predictions_df = pd.read_csv(PREDICTIONS_PATH)
predictions_df["week_start"] = pd.to_datetime(predictions_df["week_start"])

best_models_df = pd.read_csv(BEST_MODELS_PATH)

# Оставляем тот же рабочий период, что и для robust-валидации
feature_df = feature_df[feature_df["week_start"] >= pd.Timestamp("2022-01-03")].copy()
feature_df = feature_df.sort_values("week_start").reset_index(drop=True)

# Автоматически выбираем последнюю общую неделю, где есть прогнозы +1/+2/+3
origin_week = find_latest_common_origin(
    predictions_df=predictions_df,
    best_models_df=best_models_df,
    horizons=(1, 2, 3),
)

comparison_df = build_single_origin_comparison(
    feature_df=feature_df,
    predictions_df=predictions_df,
    best_models_df=best_models_df,
    origin_week=origin_week,
)

figures_dir = ROOT / "reports" / "figures"
tables_dir = ROOT / "reports" / "tables"
figures_dir.mkdir(parents=True, exist_ok=True)
tables_dir.mkdir(parents=True, exist_ok=True)

fig_path = figures_dir / "single_origin_demo_step9.png"
table_path = tables_dir / "single_origin_demo_step9.csv"

plot_single_origin_forecast(
    feature_df=feature_df,
    comparison_df=comparison_df,
    out_path=fig_path,
    lookback_weeks=16,
)

save_single_origin_table(
    comparison_df=comparison_df,
    out_path=table_path,
)

print("=" * 80)
print("STEP 9 SINGLE-ORIGIN DEMO")
print("=" * 80)
print("origin week:", origin_week)
print("saved figure:", fig_path)
print("saved table :", table_path)
print()
print(comparison_df.to_string(index=False))
