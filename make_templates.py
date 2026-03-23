from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
base_path = ROOT / "data" / "interim" / "base_weekly_dataset.csv"

if not base_path.exists():
    raise FileNotFoundError(f"Не найден файл: {base_path}")

base = pd.read_csv(base_path)
base["week_start"] = pd.to_datetime(base["week_start"])

weeks = pd.DataFrame({
    "week_start": base["week_start"].drop_duplicates().sort_values()
})

weather = weeks.copy()
weather["temp_mean"] = ""
weather["temp_min"] = ""
weather["temp_max"] = ""
weather["humidity_mean"] = ""
weather["pressure_mean"] = ""
weather["wind_speed_mean"] = ""
weather["precip_sum"] = ""

calendar = weeks.copy()
calendar["holiday_days"] = 0
calendar["school_break_days"] = 0
calendar["is_holiday_week"] = 0
calendar["is_school_break_week"] = 0
calendar["is_school_start_week"] = 0
calendar["is_new_year_period"] = 0
calendar["is_may_holiday_period"] = 0
calendar["restriction_level"] = 0
calendar["mask_mandate"] = 0
calendar["remote_work"] = 0

weather_dir = ROOT / "data" / "raw" / "weather"
calendar_dir = ROOT / "data" / "raw" / "calendar"
weather_dir.mkdir(parents=True, exist_ok=True)
calendar_dir.mkdir(parents=True, exist_ok=True)

weather_path = weather_dir / "weather_moscow_weekly.csv"
calendar_path = calendar_dir / "calendar_moscow_weekly.csv"

weather.to_csv(weather_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
calendar.to_csv(calendar_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")

print("Создано:")
print(weather_path)
print(calendar_path)
print()
print("Строк в шаблонах:", len(weeks))
print("Диапазон:", weeks["week_start"].min(), "->", weeks["week_start"].max())