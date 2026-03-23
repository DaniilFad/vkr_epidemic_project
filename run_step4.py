from pathlib import Path
import pandas as pd

from src.features import build_feature_dataset, save_feature_outputs


ROOT = Path(__file__).resolve().parent
BASE_PATH = ROOT / "data" / "interim" / "base_weekly_dataset.csv"

if not BASE_PATH.exists():
    raise FileNotFoundError(f"Не найден файл: {BASE_PATH}")

base_df = pd.read_csv(BASE_PATH)
base_df["week_start"] = pd.to_datetime(base_df["week_start"])

feature_df, lag_table = build_feature_dataset(base_df, max_wordstat_lag=6)
feature_path, lag_path = save_feature_outputs(feature_df, lag_table, ROOT)

print("=" * 80)
print("STEP 4 RESULT")
print("=" * 80)
print("feature dataset shape:", feature_df.shape)
print("date range:", feature_df["week_start"].min(), "->", feature_df["week_start"].max())
print()
print("saved feature dataset:", feature_path)
print("saved lag diagnostics:", lag_path)
print()

print("=" * 80)
print("TARGET PREVIEW")
print("=" * 80)
target_cols = ["week_start", "cases", "target_t_plus_1", "target_t_plus_2", "target_t_plus_3"]
print(feature_df[target_cols].head(12).to_string(index=False))
print()

print("=" * 80)
print("SELECTED WORDSTAT LAGS")
print("=" * 80)
if lag_table.empty:
    print("lag_table is empty")
else:
    print(lag_table.head(20).to_string(index=False))
print()

print("=" * 80)
print("KEY FEATURE PREVIEW")
print("=" * 80)
preview_cols = [
    "week_start",
    "cases",
    "target_t_plus_1",
    "cases_lag_1",
    "cases_lag_2",
    "cases_roll_mean_4",
    "cases_roll_std_4",
    "cases_growth_1w",
    "weekofyear_sin",
    "temp_mean",
    "restriction_level",
    "ws_total_raw",
    "ws_total_denoised",
    "ws_total_spike_flags",
    "ws_group_disease",
    "ws_group_testing",
    "ws_group_specific_symptoms",
    "ws_group_general_symptoms",
    "ws_group_complications_treatment",
]
preview_cols = [c for c in preview_cols if c in feature_df.columns]
print(feature_df[preview_cols].head(12).to_string(index=False))
print()

print("=" * 80)
print("MISSING VALUES (TOP 40)")
print("=" * 80)
missing = feature_df.isna().sum().sort_values(ascending=False)
print(missing.head(40).to_string())