from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW_CAL_DIR = ROOT / "data" / "raw" / "calendar" / "raw"
OUT_DIR = ROOT / "data" / "raw" / "calendar"
OUT_PATH = OUT_DIR / "calendar_moscow_weekly.csv"
BASE_PATH = ROOT / "data" / "interim" / "base_weekly_dataset.csv"


def week_monday(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return (ts - pd.Timedelta(days=ts.weekday())).normalize()


def parse_month_cell(year: int, month: int, cell_value: str) -> pd.DataFrame:
    if pd.isna(cell_value):
        return pd.DataFrame(columns=["date", "is_nonworking_from_calendar", "is_short_workday", "is_shifted_holiday"])

    tokens = [t.strip() for t in str(cell_value).split(",") if t.strip()]
    rows = []

    for token in tokens:
        is_short = token.endswith("*")
        is_shifted = token.endswith("+")

        day_str = token.replace("*", "").replace("+", "").strip()
        if not day_str.isdigit():
            continue

        day = int(day_str)

        try:
            dt = pd.Timestamp(year=year, month=month, day=day)
        except ValueError:
            continue

        rows.append({
            "date": dt,
            "is_nonworking_from_calendar": 1,
            "is_short_workday": int(is_short),
            "is_shifted_holiday": int(is_shifted),
        })

    return pd.DataFrame(rows)


def load_year_calendar(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty:
        raise ValueError(f"Пустой CSV: {csv_path}")

    row = df.iloc[0]
    year = int(row["Год/Месяц"])

    months_ru = {
        "Январь": 1,
        "Февраль": 2,
        "Март": 3,
        "Апрель": 4,
        "Май": 5,
        "Июнь": 6,
        "Июль": 7,
        "Август": 8,
        "Сентябрь": 9,
        "Октябрь": 10,
        "Ноябрь": 11,
        "Декабрь": 12,
    }

    all_parts = []
    for month_name, month_num in months_ru.items():
        if month_name not in df.columns:
            continue
        part = parse_month_cell(year, month_num, row[month_name])
        all_parts.append(part)

    if not all_parts:
        raise ValueError(f"Не удалось распарсить месяцы из {csv_path}")

    out = pd.concat(all_parts, ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date")
    out["weekday"] = out["date"].dt.weekday
    out["is_weekend"] = out["weekday"].isin([5, 6]).astype(int)

    out["is_extra_holiday_day"] = (
        ((out["is_nonworking_from_calendar"] == 1) & (out["is_weekend"] == 0))
        | (out["is_shifted_holiday"] == 1)
    ).astype(int)

    return out[[
        "date",
        "is_weekend",
        "is_nonworking_from_calendar",
        "is_short_workday",
        "is_shifted_holiday",
        "is_extra_holiday_day",
    ]].copy()


def add_days_to_weeks(calendar_df: pd.DataFrame, start_date: str, end_date: str, column_days: str, binary_cols=None):
    if binary_cols is None:
        binary_cols = []

    days = pd.date_range(start=start_date, end=end_date, freq="D")
    if len(days) == 0:
        return calendar_df

    tmp = pd.DataFrame({"date": days})
    tmp["week_start"] = tmp["date"].apply(week_monday)
    counts = tmp.groupby("week_start").size().rename(column_days).reset_index()

    calendar_df = calendar_df.merge(counts, on="week_start", how="left", suffixes=("", "_tmp"))
    calendar_df[column_days] = calendar_df[column_days].fillna(0) + calendar_df[f"{column_days}_tmp"].fillna(0)
    calendar_df = calendar_df.drop(columns=[f"{column_days}_tmp"])

    for col in binary_cols:
        calendar_df.loc[calendar_df[column_days] > 0, col] = 1

    return calendar_df


def build_school_breaks(calendar_weekly: pd.DataFrame) -> pd.DataFrame:
    school_break_ranges = [
        ("2022-10-10", "2022-10-16"),
        ("2022-10-29", "2022-11-06"),
        ("2022-11-21", "2022-11-27"),
        ("2022-12-31", "2023-01-08"),
        ("2023-02-20", "2023-02-26"),
        ("2023-04-03", "2023-04-16"),
        ("2023-10-09", "2023-10-15"),
        ("2023-10-30", "2023-11-05"),
        ("2023-11-20", "2023-11-26"),
        ("2024-01-01", "2024-01-08"),
        ("2024-02-19", "2024-02-25"),
        ("2024-03-25", "2024-03-31"),
        ("2024-04-08", "2024-04-14"),
        ("2024-10-07", "2024-10-13"),
        ("2024-10-28", "2024-11-04"),
        ("2024-11-18", "2024-11-24"),
        ("2024-12-30", "2025-01-12"),
        ("2025-02-17", "2025-02-24"),
        ("2025-03-24", "2025-03-31"),
        ("2025-04-07", "2025-04-13"),
    ]

    for start, end in school_break_ranges:
        calendar_weekly = add_days_to_weeks(
            calendar_weekly,
            start,
            end,
            column_days="school_break_days",
            binary_cols=["is_school_break_week"]
        )

    return calendar_weekly


def main():
    if not BASE_PATH.exists():
        raise FileNotFoundError(f"Не найден файл: {BASE_PATH}")

    base = pd.read_csv(BASE_PATH)
    base["week_start"] = pd.to_datetime(base["week_start"])

    files = [
        RAW_CAL_DIR / "calendar_2022.csv",
        RAW_CAL_DIR / "calendar_2023.csv",
        RAW_CAL_DIR / "calendar_2024.csv",
        RAW_CAL_DIR / "calendar_2025.csv",
    ]

    missing_files = [str(f) for f in files if not f.exists()]
    if missing_files:
        raise FileNotFoundError(f"Не найдены файлы календаря: {missing_files}")

    day_tables = [load_year_calendar(f) for f in files]
    days = pd.concat(day_tables, ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date")
    days["week_start"] = days["date"].apply(week_monday)

    weekly = (
        days.groupby("week_start", as_index=False)
        .agg(
            holiday_days=("is_extra_holiday_day", "sum"),
            short_work_days=("is_short_workday", "sum"),
        )
        .sort_values("week_start")
        .reset_index(drop=True)
    )

    calendar_weekly = pd.DataFrame({
        "week_start": base["week_start"].drop_duplicates().sort_values()
    }).reset_index(drop=True)

    calendar_weekly = calendar_weekly.merge(weekly, on="week_start", how="left")

    base_min = calendar_weekly["week_start"].min()
    base_max = calendar_weekly["week_start"].max() + pd.Timedelta(days=6)

    all_days = pd.DataFrame({"date": pd.date_range(base_min, base_max, freq="D")})
    all_days["week_start"] = all_days["date"].apply(week_monday)
    all_days["is_weekend_auto"] = all_days["date"].dt.weekday.isin([5, 6]).astype(int)

    weekend_auto = (
        all_days.groupby("week_start", as_index=False)["is_weekend_auto"]
        .sum()
        .rename(columns={"is_weekend_auto": "weekend_days"})
    )

    calendar_weekly = calendar_weekly.merge(weekend_auto, on="week_start", how="left")

    calendar_weekly["holiday_days"] = calendar_weekly["holiday_days"].fillna(0).astype(int)
    calendar_weekly["short_work_days"] = calendar_weekly["short_work_days"].fillna(0).astype(int)
    calendar_weekly["weekend_days"] = calendar_weekly["weekend_days"].fillna(2).astype(int)

    calendar_weekly["is_holiday_week"] = (calendar_weekly["holiday_days"] > 0).astype(int)
    calendar_weekly["is_school_break_week"] = 0
    calendar_weekly["school_break_days"] = 0

    calendar_weekly["is_new_year_period"] = 0
    calendar_weekly["is_may_holiday_period"] = 0
    calendar_weekly["is_school_start_week"] = 0

    years = sorted(calendar_weekly["week_start"].dt.year.unique())

    for year in years:
        ny_days = pd.date_range(f"{year}-01-01", f"{year}-01-08", freq="D")
        ny_weeks = pd.Series(ny_days).apply(week_monday).drop_duplicates()
        calendar_weekly.loc[calendar_weekly["week_start"].isin(ny_weeks), "is_new_year_period"] = 1

        may_days = pd.date_range(f"{year}-05-01", f"{year}-05-10", freq="D")
        may_weeks = pd.Series(may_days).apply(week_monday).drop_duplicates()
        calendar_weekly.loc[calendar_weekly["week_start"].isin(may_weeks), "is_may_holiday_period"] = 1

        sep_1 = pd.Timestamp(f"{year}-09-01")
        ws = week_monday(sep_1)
        calendar_weekly.loc[calendar_weekly["week_start"] == ws, "is_school_start_week"] = 1

    calendar_weekly = build_school_breaks(calendar_weekly)

    calendar_weekly["restriction_level"] = 0
    calendar_weekly["mask_mandate"] = 0
    calendar_weekly["remote_work"] = 0

    restriction_periods = [
        ("2020-03-30", "2020-06-08", 3, 0, 1),
        ("2020-10-05", "2021-01-17", 1, 1, 0),
        ("2021-06-28", "2022-03-14", 1, 1, 0),
    ]

    for start, end, level, mask, remote in restriction_periods:
        days_range = pd.date_range(start=start, end=end, freq="D")
        weeks = pd.Series(days_range).apply(week_monday).drop_duplicates()
        calendar_weekly.loc[calendar_weekly["week_start"].isin(weeks), "restriction_level"] = level
        calendar_weekly.loc[calendar_weekly["week_start"].isin(weeks), "mask_mandate"] = mask
        calendar_weekly.loc[calendar_weekly["week_start"].isin(weeks), "remote_work"] = remote

    int_cols = [
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
    for col in int_cols:
        calendar_weekly[col] = calendar_weekly[col].fillna(0).astype(int)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    calendar_weekly.to_csv(OUT_PATH, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")

    print("=" * 80)
    print("CALENDAR FEATURES CREATED")
    print("=" * 80)
    print("output:", OUT_PATH)
    print("shape :", calendar_weekly.shape)
    print("date range:", calendar_weekly["week_start"].min(), "->", calendar_weekly["week_start"].max())
    print()
    print(calendar_weekly.head(20).to_string(index=False))
    print()
    print("Non-zero counts:")
    for col in int_cols:
        print(f"{col}: {(calendar_weekly[col] > 0).sum()}")


if __name__ == "__main__":
    main()