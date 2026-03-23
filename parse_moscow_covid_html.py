from pathlib import Path
import shutil
import pandas as pd
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parent

INPUT_HTML = ROOT / "Статистика коронавируса в Москве_ динамика заболевших и умерших по дням.html"

OUTPUT_DIR = ROOT / "data" / "raw" / "covid"
OUTPUT_CSV = OUTPUT_DIR / "Статистика КОВИД.csv"
BACKUP_CSV = OUTPUT_DIR / "Статистика КОВИД_BACKUP_OLD.csv"


def clean_text(x: str) -> str:
    return " ".join(str(x).replace("\xa0", " ").split()).strip()


def normalize_header(x: str) -> str:
    x = clean_text(x)
    x = x.replace("Выздоров лений", "Выздоровлений")
    return x


def find_moscow_covid_table(html_path: Path) -> pd.DataFrame:
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    tables = soup.find_all("table")
    if not tables:
        raise ValueError(f"В HTML не найдено ни одной таблицы: {html_path}")

    required_cols = {"Дата", "Заражений", "Смертей", "Выздоровлений", "Заражено на дату"}

    for table in tables:
        rows = table.find_all("tr")
        if not rows:
            continue

        header_cells = rows[0].find_all(["th", "td"])
        headers = [normalize_header(cell.get_text(" ", strip=True)) for cell in header_cells]

        if not required_cols.issubset(set(headers)):
            continue

        data = []
        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            if len(cells) != len(headers):
                continue
            values = [clean_text(cell.get_text(" ", strip=True)) for cell in cells]
            data.append(values)

        df = pd.DataFrame(data, columns=headers)
        return df

    raise ValueError(
        f"Не найдена таблица с колонками {required_cols}. Проверь структуру HTML: {html_path}"
    )


def to_num(series: pd.Series) -> pd.Series:
    s = (
        series.astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace("\xa0", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    return pd.to_numeric(s, errors="coerce")


def build_moscow_covid_csv(table_df: pd.DataFrame) -> pd.DataFrame:
    df = table_df.copy()

    df = df.rename(
        columns={
            "Дата": "date",
            "Заражений": "total_cases",
            "Смертей": "total_deaths",
            "Выздоровлений": "recovered_total",
            "Заражено на дату": "active_now",
        }
    )

    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y", errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    for col in ["total_cases", "total_deaths", "recovered_total", "active_now"]:
        df[col] = to_num(df[col])

    df = df.dropna(subset=["total_cases", "total_deaths", "recovered_total", "active_now"]).copy()

    df = df.sort_values("date").reset_index(drop=True)

    # Разница между соседними наблюдениями: если участок дневной — получим день,
    # если участок недельный — получим прирост за неделю
    df["period_days"] = df["date"].diff().dt.days
    df["cases_day"] = df["total_cases"].diff()
    df["deaths_day"] = df["total_deaths"].diff()
    df["recovered_day"] = df["recovered_total"].diff()
    df["active_delta"] = df["active_now"].diff()

    out = df[
        [
            "date",
            "cases_day",
            "deaths_day",
            "recovered_day",
            "active_now",
            "active_delta",
            "total_cases",
            "total_deaths",
            "recovered_total",
            "period_days",
        ]
    ].copy()

    return out


def main():
    if not INPUT_HTML.exists():
        raise FileNotFoundError(
            f"Не найден HTML-файл: {INPUT_HTML}\n"
            f"Положи его в корень проекта или измени путь в INPUT_HTML."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_CSV.exists() and not BACKUP_CSV.exists():
        shutil.copy2(OUTPUT_CSV, BACKUP_CSV)

    raw_table = find_moscow_covid_table(INPUT_HTML)
    out_df = build_moscow_covid_csv(raw_table)

    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")

    print("=" * 80)
    print("MOSCOW COVID HTML PARSED")
    print("=" * 80)
    print("input :", INPUT_HTML)
    print("output:", OUTPUT_CSV)
    print("backup:", BACKUP_CSV if BACKUP_CSV.exists() else "not created")
    print()
    print("shape:", out_df.shape)
    print("date range:", out_df["date"].min(), "->", out_df["date"].max())
    print()
    print("period_days value counts:")
    print(out_df["period_days"].value_counts(dropna=False).sort_index().to_string())
    print()
    print("head:")
    print(out_df.head(10).to_string(index=False))
    print()
    print("tail:")
    print(out_df.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
