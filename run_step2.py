from pathlib import Path
from src.build_dataset import build_base_weekly_dataset

ROOT = Path(__file__).resolve().parent

df = build_base_weekly_dataset(ROOT)

print("=" * 80)
print("STEP 2 RESULT")
print("=" * 80)
print("shape:", df.shape)
print("date range:", df["week_start"].min(), "->", df["week_start"].max())
print()
print(df.head(10).to_string(index=False))
print()
print("columns:")
for c in df.columns:
    print("-", c)
print()
print("missing values:")
print(df.isna().sum().sort_values(ascending=False).to_string())

out_path = ROOT / "data" / "interim" / "base_weekly_dataset.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False, encoding="utf-8-sig")
print()
print("saved:", out_path)