from pathlib import Path
from src.build_dataset import build_base_weekly_dataset

ROOT = Path(__file__).resolve().parent
df = build_base_weekly_dataset(ROOT)

print("=" * 80)
print("STEP 3 RESULT")
print("=" * 80)
print("shape:", df.shape)
print("date range:", df["week_start"].min(), "->", df["week_start"].max())
print()

important_cols = [c for c in [
    "week_start", "cases",
    "temp_mean", "temp_min", "temp_max", "humidity_mean", "precip_sum",
    "weekend_days", "holiday_days", "short_work_days",
    "school_break_days", "restriction_level",
    "is_holiday_week", "is_school_break_week", "mask_mandate", "remote_work"
] if c in df.columns]

print(df[important_cols].head(12).to_string(index=False))
print()
print("missing values:")
print(df.isna().sum().sort_values(ascending=False).to_string())

out_path = ROOT / "data" / "interim" / "base_weekly_dataset.csv"
df.to_csv(out_path, index=False, encoding="utf-8-sig")
print()
print("saved:", out_path)