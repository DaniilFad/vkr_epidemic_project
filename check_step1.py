from pathlib import Path
from src.data_io import load_covid_weekly, load_wordstat_dir

ROOT = Path(__file__).resolve().parent

covid_path = ROOT / "data" / "raw" / "covid" / "Статистика КОВИД.csv"
wordstat_dir = ROOT / "data" / "raw" / "wordstat"

covid = load_covid_weekly(covid_path)
wordstat = load_wordstat_dir(wordstat_dir)

print("=" * 80)
print("COVID DATA")
print("=" * 80)
print(covid.head())
print()
print("shape:", covid.shape)
print("date range:", covid["week_start"].min(), "->", covid["week_start"].max())
print("columns:", list(covid.columns))
print()

print("=" * 80)
print("WORDSTAT DATA")
print("=" * 80)
print(wordstat.head())
print()
print("shape:", wordstat.shape)
print("date range:", wordstat["week_start"].min(), "->", wordstat["week_start"].max())
print("columns:", list(wordstat.columns))
print()

merged = covid.merge(wordstat, on="week_start", how="left")
print("=" * 80)
print("MERGED DATA")
print("=" * 80)
print(merged.head())
print()
print("shape:", merged.shape)
print("missing values:")
print(merged.isna().sum().sort_values(ascending=False))