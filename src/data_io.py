from pathlib import Path
import pandas as pd


def _clean_num(series: pd.Series) -> pd.Series:
    s = (
        series.astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace("\u00a0", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    s = s.replace({"nan": None, "": None, "-": None})
    return pd.to_numeric(s, errors="coerce")


def _read_csv_robust(csv_path: str | Path, sep=",") -> pd.DataFrame:
    csv_path = Path(csv_path)

    last_error = None
    for encoding in ["utf-8", "utf-8-sig", "cp1251"]:
        try:
            return pd.read_csv(csv_path, sep=sep, encoding=encoding, engine="python")
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Could not read CSV: {csv_path}\nLast error: {last_error}")


def _to_week_monday(series: pd.Series) -> pd.Series:
    series = pd.to_datetime(series, errors="coerce")
    return (series - pd.to_timedelta(series.dt.weekday, unit="D")).dt.normalize()


def _parse_wordstat_dates(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()

    # Сначала пробуем самый ожидаемый формат dd.mm.yyyy
    dt = pd.to_datetime(s, format="%d.%m.%Y", errors="coerce")

    # Если не вышло — fallback
    mask = dt.isna()
    if mask.any():
        dt.loc[mask] = pd.to_datetime(s.loc[mask], dayfirst=True, errors="coerce")

    return dt


def load_covid_weekly(csv_path: str | Path) -> pd.DataFrame:
    df = _read_csv_robust(csv_path, sep=",")

    if "date" not in df.columns:
        raise ValueError(f"'date' column not found in {csv_path}. Columns: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    numeric_candidates = [
        "cases_day",
        "deaths_day",
        "recovered_day",
        "active_now",
        "active_delta",
        "total_cases",
        "total_deaths",
        "recovered_total",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = _clean_num(df[col])

    df["week_start"] = _to_week_monday(df["date"])

    agg_dict = {}
    if "cases_day" in df.columns:
        agg_dict["cases_day"] = "sum"
    if "deaths_day" in df.columns:
        agg_dict["deaths_day"] = "sum"
    if "recovered_day" in df.columns:
        agg_dict["recovered_day"] = "sum"
    if "active_now" in df.columns:
        agg_dict["active_now"] = "last"
    if "active_delta" in df.columns:
        agg_dict["active_delta"] = "sum"
    if "total_cases" in df.columns:
        agg_dict["total_cases"] = "last"
    if "total_deaths" in df.columns:
        agg_dict["total_deaths"] = "last"
    if "recovered_total" in df.columns:
        agg_dict["recovered_total"] = "last"

    out = (
        df.groupby("week_start", as_index=False)
        .agg(agg_dict)
        .sort_values("week_start")
        .reset_index(drop=True)
    )

    rename_map = {
        "cases_day": "cases",
        "deaths_day": "deaths",
        "recovered_day": "recovered",
    }
    out = out.rename(columns=rename_map)
    return out


def load_single_wordstat(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    query_name = csv_path.stem.strip().lower().replace(" ", "_")

    raw = pd.read_csv(
        csv_path,
        sep=";",
        header=None,
        names=["date_raw", "query_count", "share", "sparkline"],
        encoding="utf-8-sig",
        engine="python",
    )

    raw["date"] = _parse_wordstat_dates(raw["date_raw"])
    raw = raw.dropna(subset=["date"]).copy()

    raw["week_start"] = _to_week_monday(raw["date"])
    raw["query_count"] = _clean_num(raw["query_count"])
    raw["share"] = _clean_num(raw["share"])

    raw = raw[["week_start", "query_count", "share"]].copy()
    raw = raw.rename(
        columns={
            "query_count": f"ws_{query_name}",
            "share": f"ws_share_{query_name}",
        }
    )

    raw = (
        raw.groupby("week_start", as_index=False)
        .agg({
            f"ws_{query_name}": "sum",
            f"ws_share_{query_name}": "mean",
        })
        .sort_values("week_start")
        .reset_index(drop=True)
    )

    return raw


def load_wordstat_dir(wordstat_dir: str | Path) -> pd.DataFrame:
    wordstat_dir = Path(wordstat_dir)
    files = sorted(wordstat_dir.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found in {wordstat_dir}")

    merged = None
    for fp in files:
        cur = load_single_wordstat(fp)
        if merged is None:
            merged = cur
        else:
            merged = merged.merge(cur, on="week_start", how="outer")

    merged = merged.sort_values("week_start").reset_index(drop=True)
    return merged


def load_weather_weekly(csv_path: str | Path) -> pd.DataFrame:
    df = _read_csv_robust(csv_path, sep=",")

    if "week_start" not in df.columns:
        raise ValueError(f"'week_start' column not found in {csv_path}. Columns: {list(df.columns)}")

    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    df["week_start"] = _to_week_monday(df["week_start"])
    df = df.dropna(subset=["week_start"]).copy()

    weather_cols = [
        "temp_mean",
        "temp_min",
        "temp_max",
        "humidity_mean",
        "pressure_mean",
        "wind_speed_mean",
        "precip_sum",
    ]

    keep_cols = ["week_start"]
    for col in weather_cols:
        if col in df.columns:
            df[col] = _clean_num(df[col])
            keep_cols.append(col)

    df = df[keep_cols].copy()
    df = df.groupby("week_start", as_index=False).mean(numeric_only=True)
    return df


def load_calendar_weekly(csv_path: str | Path) -> pd.DataFrame:
    df = _read_csv_robust(csv_path, sep=",")

    if "week_start" not in df.columns:
        raise ValueError(f"'week_start' column not found in {csv_path}. Columns: {list(df.columns)}")

    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    df["week_start"] = _to_week_monday(df["week_start"])
    df = df.dropna(subset=["week_start"]).copy()

    cal_cols = [
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

    keep_cols = ["week_start"]
    for col in cal_cols:
        if col in df.columns:
            df[col] = _clean_num(df[col])
            keep_cols.append(col)

    df = df[keep_cols].copy()
    df = df.groupby("week_start", as_index=False).max(numeric_only=True)
    return df