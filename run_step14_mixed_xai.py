from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import Pool
from src.models_advanced import (
    build_feature_pool_for_horizon,
    select_top_features,
    prepare_xy,
    make_catboost_regressor,
)

ROOT = Path(__file__).resolve().parent

SELECTION_PATH = ROOT / "reports" / "tables" / "final_mixed_model_selection_step13.csv"
FINAL_FORECAST_PATH = ROOT / "reports" / "predictions" / "final_forecast_step13_mixed.csv"
ADVANCED_DATA_PATH = ROOT / "data" / "processed" / "modeling_dataset_advanced.csv"

OLD_XAI_SUMMARY_PATH = ROOT / "reports" / "tables" / "xai_summary_step8.csv"

OUT_TABLES_DIR = ROOT / "reports" / "tables"
OUT_FIGURES_DIR = ROOT / "reports" / "figures"


def infer_advanced_model_kind(model_name: str) -> str:
    if model_name == "advanced_level_catboost":
        return "level"
    if model_name == "advanced_logdelta_catboost":
        return "logdelta"
    if model_name == "advanced_ratio_catboost":
        return "ratio"
    raise ValueError(f"Неизвестная advanced-модель: {model_name}")


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл: {path}")
    return pd.read_csv(path)


def latest_train_df(advanced_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = advanced_df.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values("week_start").reset_index(drop=True)

    train_df = df[df["cases"].notna()].copy().reset_index(drop=True)
    if train_df.empty:
        raise ValueError("В advanced dataset нет строк с непустыми cases.")

    latest_row = train_df.tail(1).copy()
    return train_df, latest_row


def get_target_col(model_kind: str, horizon: int) -> str:
    if model_kind == "level":
        return f"target_t_plus_{horizon}"
    if model_kind == "logdelta":
        return f"target_logdelta_h{horizon}"
    if model_kind == "ratio":
        return f"target_ratio_h{horizon}"
    raise ValueError(f"Неизвестный model_kind: {model_kind}")


def get_top_k(model_kind: str) -> int:
    if model_kind == "level":
        return 38
    return 42


def fit_advanced_model_with_xai(
    advanced_df: pd.DataFrame,
    horizon: int,
    model_name: str,
):
    model_kind = infer_advanced_model_kind(model_name)

    train_df, latest_row = latest_train_df(advanced_df)

    pool_cols, mandatory_cols, _ = build_feature_pool_for_horizon(train_df, horizon)
    target_col = get_target_col(model_kind, horizon)
    top_k = get_top_k(model_kind)

    selected_features = select_top_features(
        train_df=train_df,
        target_col=target_col,
        candidate_cols=pool_cols,
        mandatory_cols=mandatory_cols,
        top_k_total=top_k,
        min_obs=25,
    )

    X_train, y_train_raw, medians = prepare_xy(
        train_df=train_df,
        feature_cols=selected_features,
        target_col=target_col,
    )

    X_test = (
        latest_row[selected_features]
        .copy()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(medians)
    )

    if model_kind == "level":
        y_fit = np.log1p(y_train_raw)
    elif model_kind == "logdelta":
        y_fit = y_train_raw.copy()
    elif model_kind == "ratio":
        y_fit = np.log(y_train_raw.clip(lower=0.01, upper=20.0))
    else:
        raise ValueError(f"Неизвестный model_kind: {model_kind}")

    model = make_catboost_regressor(random_state=42)
    model.fit(X_train, y_fit)

    train_pool = Pool(X_train, y_fit, feature_names=list(X_train.columns))
    test_pool = Pool(X_test, feature_names=list(X_test.columns))

    global_importance = model.get_feature_importance(train_pool, type="FeatureImportance")
    shap_values_full = model.get_feature_importance(test_pool, type="ShapValues")

    # shap_values_full shape: (1, n_features + 1), последний столбец - expected value
    shap_row = shap_values_full[0]
    local_shap_values = shap_row[:-1]
    expected_value = float(shap_row[-1])

    global_df = pd.DataFrame({
        "feature_name": selected_features,
        "importance": global_importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    local_df = pd.DataFrame({
        "feature_name": selected_features,
        "feature_value": X_test.iloc[0].values,
        "shap_value": local_shap_values,
    }).sort_values("shap_value", ascending=True).reset_index(drop=True)

    pred_transformed = float(model.predict(X_test)[0])

    if model_kind == "level":
        predicted_cases = float(np.expm1(pred_transformed))
        shap_scale = "log1p_cases"
    elif model_kind == "logdelta":
        current_cases = float(latest_row.iloc[0]["cases"])
        predicted_cases = float(np.expm1(np.log1p(current_cases) + pred_transformed))
        shap_scale = "logdelta"
    elif model_kind == "ratio":
        current_cases = float(latest_row.iloc[0]["cases"])
        pred_ratio = float(np.exp(pred_transformed))
        predicted_cases = float(current_cases * pred_ratio)
        shap_scale = "log_ratio"
    else:
        raise ValueError(f"Неизвестный model_kind: {model_kind}")

    return {
        "model_kind": model_kind,
        "selected_features": selected_features,
        "global_df": global_df,
        "local_df": local_df,
        "expected_value": expected_value,
        "predicted_cases": predicted_cases,
        "shap_scale": shap_scale,
    }


def save_barplot(
    df: pd.DataFrame,
    value_col: str,
    label_col: str,
    title: str,
    out_path: Path,
    top_n: int = 15,
    ascending: bool = True,
):
    part = df.copy()

    if value_col == "importance":
        part = part.sort_values(value_col, ascending=False).head(top_n).copy()
        part = part.sort_values(value_col, ascending=True)
    else:
        if ascending:
            part = part.sort_values(value_col, ascending=True).head(top_n).copy()
        else:
            part = part.sort_values(value_col, ascending=False).head(top_n).copy()
            part = part.sort_values(value_col, ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(part[label_col], part[value_col])
    ax.set_title(title)
    ax.set_xlabel(value_col)
    ax.set_ylabel("Признак")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def top_feature_strings(local_df: pd.DataFrame, n: int = 5) -> tuple[str, str]:
    pos = (
        local_df.sort_values("shap_value", ascending=False)
        .head(n)["feature_name"]
        .tolist()
    )
    neg = (
        local_df.sort_values("shap_value", ascending=True)
        .head(n)["feature_name"]
        .tolist()
    )
    return ", ".join(pos), ", ".join(neg)


def derive_old_png_from_csv_path(csv_path_str: str) -> Path:
    csv_path = Path(csv_path_str)
    png_name = csv_path.stem + ".png"
    return ROOT / "reports" / "figures" / png_name


def safe_copy(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def reuse_old_xai_for_horizon(
    horizon: int,
    final_forecast_df: pd.DataFrame,
    old_xai_summary_df: pd.DataFrame,
) -> dict:
    cur = old_xai_summary_df[old_xai_summary_df["horizon_weeks"] == horizon].copy()
    if cur.empty:
        raise ValueError(f"В старом xai_summary_step8.csv нет горизонта {horizon}")

    row = cur.iloc[0]

    global_csv_old = Path(str(row["global_importance_path"]))
    local_csv_old = Path(str(row["local_shap_path"]))

    global_png_old = derive_old_png_from_csv_path(str(global_csv_old))
    local_png_old = derive_old_png_from_csv_path(str(local_csv_old))

    global_csv_new = OUT_TABLES_DIR / f"mixed_global_importance_h{horizon}_step14.csv"
    local_csv_new = OUT_TABLES_DIR / f"mixed_local_shap_h{horizon}_step14.csv"
    global_png_new = OUT_FIGURES_DIR / f"mixed_global_importance_h{horizon}_step14.png"
    local_png_new = OUT_FIGURES_DIR / f"mixed_local_shap_h{horizon}_step14.png"

    safe_copy(global_csv_old, global_csv_new)
    safe_copy(local_csv_old, local_csv_new)
    safe_copy(global_png_old, global_png_new)
    safe_copy(local_png_old, local_png_new)

    forecast_row = final_forecast_df[final_forecast_df["horizon_weeks"] == horizon].iloc[0]

    return {
        "horizon_weeks": horizon,
        "model_name": forecast_row["model_name"],
        "model_source": "old",
        "predicted_cases": float(forecast_row["point_forecast"]),
        "shap_scale": "old_step8_reused",
        "global_importance_path": str(global_csv_new if global_csv_new.exists() else global_csv_old),
        "local_shap_path": str(local_csv_new if local_csv_new.exists() else local_csv_old),
        "global_plot_path": str(global_png_new if global_png_new.exists() else global_png_old),
        "local_plot_path": str(local_png_new if local_png_new.exists() else local_png_old),
        "top_positive_features": str(row["top_positive_features"]),
        "top_negative_features": str(row["top_negative_features"]),
        "note": "XAI переиспользован из step 8, так как mixed-система выбрала old-модель на этом горизонте",
    }


def build_advanced_xai_for_horizon(
    advanced_df: pd.DataFrame,
    horizon: int,
    model_name: str,
    final_forecast_df: pd.DataFrame,
) -> dict:
    xai = fit_advanced_model_with_xai(
        advanced_df=advanced_df,
        horizon=horizon,
        model_name=model_name,
    )

    global_df = xai["global_df"]
    local_df = xai["local_df"]

    global_csv_path = OUT_TABLES_DIR / f"mixed_global_importance_h{horizon}_step14.csv"
    local_csv_path = OUT_TABLES_DIR / f"mixed_local_shap_h{horizon}_step14.csv"

    global_png_path = OUT_FIGURES_DIR / f"mixed_global_importance_h{horizon}_step14.png"
    local_png_path = OUT_FIGURES_DIR / f"mixed_local_shap_h{horizon}_step14.png"

    global_df.to_csv(global_csv_path, index=False, encoding="utf-8-sig")
    local_df.to_csv(local_csv_path, index=False, encoding="utf-8-sig")

    save_barplot(
        df=global_df,
        value_col="importance",
        label_col="feature_name",
        title=f"Mixed XAI: global importance, горизонт +{horizon}",
        out_path=global_png_path,
        top_n=15,
        ascending=False,
    )

    local_top = local_df.copy()
    local_top["abs_shap"] = local_top["shap_value"].abs()
    local_top = local_top.sort_values("abs_shap", ascending=False).head(15).copy()
    local_top = local_top.sort_values("shap_value", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(local_top["feature_name"], local_top["shap_value"])
    ax.set_title(f"Mixed XAI: local SHAP, горизонт +{horizon} ({xai['shap_scale']})")
    ax.set_xlabel("SHAP contribution")
    ax.set_ylabel("Признак")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    global_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(local_png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    pos_str, neg_str = top_feature_strings(local_df, n=5)

    forecast_row = final_forecast_df[final_forecast_df["horizon_weeks"] == horizon].iloc[0]

    return {
        "horizon_weeks": horizon,
        "model_name": model_name,
        "model_source": "advanced",
        "predicted_cases": float(forecast_row["point_forecast"]),
        "shap_scale": xai["shap_scale"],
        "global_importance_path": str(global_csv_path),
        "local_shap_path": str(local_csv_path),
        "global_plot_path": str(global_png_path),
        "local_plot_path": str(local_png_path),
        "top_positive_features": pos_str,
        "top_negative_features": neg_str,
        "note": "XAI построен заново для финальной advanced-модели mixed-системы",
    }


def main():
    OUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    selection_df = load_csv(SELECTION_PATH)
    final_forecast_df = load_csv(FINAL_FORECAST_PATH)
    advanced_df = load_csv(ADVANCED_DATA_PATH)
    old_xai_summary_df = load_csv(OLD_XAI_SUMMARY_PATH)

    final_forecast_df["forecast_origin_week"] = pd.to_datetime(final_forecast_df["forecast_origin_week"])
    final_forecast_df["forecast_week_start"] = pd.to_datetime(final_forecast_df["forecast_week_start"])
    advanced_df["week_start"] = pd.to_datetime(advanced_df["week_start"])

    summary_rows = []

    for _, row in selection_df.sort_values("horizon_weeks").iterrows():
        horizon = int(row["horizon_weeks"])
        chosen_source = row["chosen_source"]
        chosen_model_name = row["chosen_model_name"]

        if chosen_source == "advanced":
            info = build_advanced_xai_for_horizon(
                advanced_df=advanced_df,
                horizon=horizon,
                model_name=chosen_model_name,
                final_forecast_df=final_forecast_df,
            )
        elif chosen_source == "old":
            info = reuse_old_xai_for_horizon(
                horizon=horizon,
                final_forecast_df=final_forecast_df,
                old_xai_summary_df=old_xai_summary_df,
            )
        else:
            raise ValueError(f"Неизвестный chosen_source: {chosen_source}")

        summary_rows.append(info)

    mixed_xai_summary = pd.DataFrame(summary_rows).sort_values("horizon_weeks").reset_index(drop=True)

    summary_path = OUT_TABLES_DIR / "mixed_xai_summary_step14.csv"
    mixed_xai_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("STEP 14 MIXED XAI RESULT")
    print("=" * 80)
    print("saved mixed xai summary:", summary_path)
    print()

    print("=" * 80)
    print("MIXED XAI SUMMARY")
    print("=" * 80)
    print(mixed_xai_summary.to_string(index=False))
    print()

    print("=" * 80)
    print("PLAIN-LANGUAGE INTERPRETATION")
    print("=" * 80)
    for _, row in mixed_xai_summary.iterrows():
        print(f"\nГоризонт +{int(row['horizon_weeks'])}")
        print(f"Модель: {row['model_name']} ({row['model_source']})")
        print(f"Прогноз: {row['predicted_cases']:.3f}")
        print(f"Факторы, толкающие прогноз вверх: {row['top_positive_features']}")
        print(f"Факторы, тянущие прогноз вниз: {row['top_negative_features']}")
        print(f"Комментарий: {row['note']}")


if __name__ == "__main__":
    main()