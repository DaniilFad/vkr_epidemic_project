from pathlib import Path
import pandas as pd

from src.data_io import (
    load_covid_weekly,
    load_wordstat_dir,
    load_weather_weekly,
    load_calendar_weekly,
)


def build_base_weekly_dataset(root: Path) -> pd.DataFrame:
    covid_path = root / "data" / "raw" / "covid" / "Статистика КОВИД.csv"
    wordstat_dir = root / "data" / "raw" / "wordstat"
    weather_path = root / "data" / "raw" / "weather" / "weather_moscow_weekly.csv"
    calendar_path = root / "data" / "raw" / "calendar" / "calendar_moscow_weekly.csv"

    covid = load_covid_weekly(covid_path)
    wordstat = load_wordstat_dir(wordstat_dir)

    min_date = min(covid["week_start"].min(), wordstat["week_start"].min())
    max_date = max(covid["week_start"].max(), wordstat["week_start"].max())

    calendar_axis = pd.DataFrame({
        "week_start": pd.date_range(min_date, max_date, freq="W-MON")
    })

    df = calendar_axis.merge(covid, on="week_start", how="left")
    df = df.merge(wordstat, on="week_start", how="left")

    if weather_path.exists():
        weather = load_weather_weekly(weather_path)
        df = df.merge(weather, on="week_start", how="left")

    if calendar_path.exists():
        cal = load_calendar_weekly(calendar_path)
        df = df.merge(cal, on="week_start", how="left")

    iso = df["week_start"].dt.isocalendar()
    df["year"] = df["week_start"].dt.year
    df["month"] = df["week_start"].dt.month
    df["quarter"] = df["week_start"].dt.quarter
    df["weekofyear"] = iso.week.astype(int)

    wordstat_cols = [c for c in df.columns if c.startswith("ws_")]
    for col in wordstat_cols:
        df[col] = df[col].fillna(0.0)

    calendar_zero_cols = [
        "weekend_days",
        "holiday_days",
        "short_work_days",
        "school_break_days",
        "is_holiday_week",
        "is_school_break_week",
        "is_school_start_week",
        "is_new_year_period",
        "is_may_holiday_period",
        "restriction_level",
        "mask_mandate",
        "remote_work",
    ]
    for col in calendar_zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # Для недельной шкалы каждая неделя — это понедельник-воскресенье,
    # поэтому обычных выходных всегда 2.
    df["weekend_days"] = 2.0

    weather_fill_cols = [
        "temp_mean",
        "temp_min",
        "temp_max",
        "humidity_mean",
        "pressure_mean",
        "wind_speed_mean",
        "precip_sum",
    ]
    for col in weather_fill_cols:
        if col in df.columns:
            df[col] = df[col].interpolate(limit_direction="both")

    df = df.sort_values("week_start").reset_index(drop=True)
    return df


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "data" / "interim"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_base_weekly_dataset(root)
    out_path = out_dir / "base_weekly_dataset.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("BASE WEEKLY DATASET CREATED")
    print("=" * 80)
    print("Saved to:", out_path)
    print()
    print("Shape:", df.shape)
    print("Date range:", df["week_start"].min(), "->", df["week_start"].max())
    print()
    print("Columns:")
    print(df.columns.tolist())
    print()
    print("Head:")
    print(df.head(10).to_string(index=False))
    print()
    print("Missing values:")
    print(df.isna().sum().sort_values(ascending=False).to_string())