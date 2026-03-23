from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parent

INPUT_PATH = ROOT / "27612.16.03.2020.31.03.2025.1.0.0.en.ansi.00000000.csv"
OUTPUT_DIR = ROOT / "data" / "raw" / "weather"
OUTPUT_PATH = OUTPUT_DIR / "weather_moscow_weekly.csv"


def clean_numeric(series: pd.Series) -> pd.Series:
    s = (
        series.astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace("\u00a0", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    s = s.replace({
        "nan": None,
        "": None,
        "-": None,
        "No precipitation": "0",
        "Осадков нет": "0",
    })
    return pd.to_numeric(s, errors="coerce")


def to_week_monday(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, dayfirst=True, errors="coerce")
    return (dt - pd.to_timedelta(dt.dt.weekday, unit="D")).dt.normalize()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        INPUT_PATH,
        sep=";",
        comment="#",
        encoding="cp1251",   # ANSI / Windows encoding
        engine="python",
        index_col=False,
    )

    # Переименование в короткие имена
    rename_map = {
        "Local time in Moscow": "datetime_local",
        "T": "temp_c",
        "Po": "pressure_hpa_station",
        "P": "pressure_hpa_sea",
        "U": "humidity_pct",
        "Ff": "wind_speed_ms",
        "RRR": "precip_mm",
    }
    df = df.rename(columns=rename_map)

    required = ["datetime_local", "temp_c", "humidity_pct", "wind_speed_ms"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"В погодном файле не найдены обязательные колонки: {missing}")

    # Дата/время
    df["datetime_local"] = pd.to_datetime(df["datetime_local"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["datetime_local"]).copy()

    # Числа
    for col in ["temp_c", "pressure_hpa_station", "pressure_hpa_sea", "humidity_pct", "wind_speed_ms", "precip_mm"]:
        if col in df.columns:
            df[col] = clean_numeric(df[col])

    # Берется давление Po как более стабильное для станции
    if "pressure_hpa_station" in df.columns:
        df["pressure_mean_source"] = df["pressure_hpa_station"]
    elif "pressure_hpa_sea" in df.columns:
        df["pressure_mean_source"] = df["pressure_hpa_sea"]
    else:
        df["pressure_mean_source"] = pd.NA

    # Недельная ось
    df["week_start"] = to_week_monday(df["datetime_local"])

    weekly = (
        df.groupby("week_start", as_index=False)
        .agg(
            temp_mean=("temp_c", "mean"),
            temp_min=("temp_c", "min"),
            temp_max=("temp_c", "max"),
            humidity_mean=("humidity_pct", "mean"),
            pressure_mean=("pressure_mean_source", "mean"),
            wind_speed_mean=("wind_speed_ms", "mean"),
            precip_sum=("precip_mm", "sum"),
        )
        .sort_values("week_start")
        .reset_index(drop=True)
    )

    # Округление
    float_cols = [
        "temp_mean", "temp_min", "temp_max",
        "humidity_mean", "pressure_mean",
        "wind_speed_mean", "precip_sum",
    ]
    for col in float_cols:
        weekly[col] = weekly[col].round(3)

    weekly.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")

    print("=" * 80)
    print("WEEKLY WEATHER CREATED")
    print("=" * 80)
    print("input :", INPUT_PATH)
    print("output:", OUTPUT_PATH)
    print("shape :", weekly.shape)
    print("date range:", weekly["week_start"].min(), "->", weekly["week_start"].max())
    print()
    print(weekly.head(12).to_string(index=False))
    print()
    print("missing values:")
    print(weekly.isna().sum().to_string())


if __name__ == "__main__":
    main()