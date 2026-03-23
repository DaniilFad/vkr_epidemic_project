from pathlib import Path
import pandas as pd

from src.aligned_plots import (
    plot_aligned_all_best_predictions,
    plot_alignment_explanation_example,
)

ROOT = Path(__file__).resolve().parent

PREDICTIONS_PATH = ROOT / "reports" / "predictions" / "walkforward_predictions_step6_robust.csv"
BEST_MODELS_PATH = ROOT / "reports" / "tables" / "best_models_step7.csv"

for path in [PREDICTIONS_PATH, BEST_MODELS_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл: {path}")

predictions_df = pd.read_csv(PREDICTIONS_PATH)
predictions_df["week_start"] = pd.to_datetime(predictions_df["week_start"])

best_models_df = pd.read_csv(BEST_MODELS_PATH)

figures_dir = ROOT / "reports" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

plot_aligned_all_best_predictions(
    predictions_df=predictions_df,
    best_models_df=best_models_df,
    out_dir=figures_dir,
)

# Поясняющий график только для горизонта +3
plot_alignment_explanation_example(
    predictions_df=predictions_df,
    best_models_df=best_models_df,
    horizon=3,
    out_path=figures_dir / "alignment_explanation_h3.png",
)

print("=" * 80)
print("ALIGNED WALK-FORWARD PLOTS CREATED")
print("=" * 80)

for name in [
    "aligned_best_prediction_h1.png",
    "aligned_best_prediction_h2.png",
    "aligned_best_prediction_h3.png",
    "alignment_explanation_h3.png",
]:
    path = figures_dir / name
    print(path)
