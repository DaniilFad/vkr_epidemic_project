from pathlib import Path
import pandas as pd

from src.model_comparison_plots import (
    build_comparison_frame,
    save_comparison_table,
    plot_model_comparison_panels,
    plot_error_comparison_panels,
)

ROOT = Path(__file__).resolve().parent

STEP6_PRED_PATH = ROOT / "reports" / "predictions" / "walkforward_predictions_step6_robust.csv"
STEP11_PRED_PATH = ROOT / "reports" / "predictions" / "walkforward_predictions_step11_advanced.csv"
STEP11_METRICS_PATH = ROOT / "reports" / "tables" / "walkforward_metrics_step11_advanced.csv"

for path in [STEP6_PRED_PATH, STEP11_PRED_PATH, STEP11_METRICS_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл: {path}")

step6_pred = pd.read_csv(STEP6_PRED_PATH)
step11_pred = pd.read_csv(STEP11_PRED_PATH)
step11_metrics = pd.read_csv(STEP11_METRICS_PATH)

comparison_df, best_adv_df = build_comparison_frame(
    step6_predictions=step6_pred,
    step11_predictions=step11_pred,
    step11_metrics=step11_metrics,
    horizons=(1, 2, 3),
    window_weeks=40,
)

figures_dir = ROOT / "reports" / "figures"
tables_dir = ROOT / "reports" / "tables"
figures_dir.mkdir(parents=True, exist_ok=True)
tables_dir.mkdir(parents=True, exist_ok=True)

comparison_table_path = tables_dir / "model_comparison_step12_window.csv"
comparison_plot_path = figures_dir / "model_comparison_step12_window.png"
error_plot_path = figures_dir / "model_error_comparison_step12_window.png"

save_comparison_table(comparison_df, comparison_table_path)

plot_model_comparison_panels(
    comparison_df=comparison_df,
    best_adv_df=best_adv_df,
    out_path=comparison_plot_path,
)

plot_error_comparison_panels(
    comparison_df=comparison_df,
    out_path=error_plot_path,
)

print("=" * 80)
print("STEP 12 VISUAL COMPARISON RESULT")
print("=" * 80)
print("saved comparison table :", comparison_table_path)
print("saved comparison plot  :", comparison_plot_path)
print("saved error plot       :", error_plot_path)
print()

print("=" * 80)
print("BEST ADVANCED MODELS BY HORIZON")
print("=" * 80)
print(best_adv_df.to_string(index=False))
print()

print("=" * 80)
print("COMPARISON TABLE PREVIEW")
print("=" * 80)
print(comparison_df.head(18).to_string(index=False))
