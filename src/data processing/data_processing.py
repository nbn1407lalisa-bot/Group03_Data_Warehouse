# + Xử lý dữ liệu đầu vào cho bài toán dự báo giá gạo từ fact_agriculture_final
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SchemaInfo:
    date_col: str
    target_col: str
    group_cols: List[str]
    id_cols: List[str]
    numeric_feature_cols: List[str]
    categorical_feature_cols: List[str]


# + Hàm lấy mode an toàn cho cột phân loại
def _safe_mode(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    mode = non_null.mode()
    if not mode.empty:
        return mode.iloc[0]
    return non_null.iloc[-1]


# + Đọc toàn bộ bảng từ BigQuery
def load_bq_table(
    table_id: str,
) -> pd.DataFrame:
    try:
        from google.cloud import bigquery
    except ImportError as exc:
        raise ImportError(
            "Chưa có google-cloud-bigquery. Cài package này trước khi chạy notebook."
        ) from exc

    client = bigquery.Client()
    query = f"SELECT * FROM `{table_id}`"
    return client.query(query).result().to_dataframe()


# + Đọc toàn bộ fact và các dimension cần dùng
def load_all_source_tables(
    project_id: str = "dwm-final-487216",
    dataset_id: str = "nong_san_dbscl",
) -> Dict[str, pd.DataFrame]:
    table_map = {
        "fact": f"{project_id}.{dataset_id}.fact_agriculture_final",
        "dim_date": f"{project_id}.{dataset_id}.dim_date",
        "dim_economy": f"{project_id}.{dataset_id}.dim_economy",
        "dim_fertilizer_type": f"{project_id}.{dataset_id}.dim_fertilizer_type",
        "dim_region": f"{project_id}.{dataset_id}.dim_region",
        "dim_rice_type": f"{project_id}.{dataset_id}.dim_rice_type",
        "dim_weather": f"{project_id}.{dataset_id}.dim_weather",
    }
    return {name: load_bq_table(table_id) for name, table_id in table_map.items()}


# + Chuẩn hóa kiểu dữ liệu cơ bản trước khi tách fact
def clean_source_tables(source: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    out = {k: v.copy() for k, v in source.items()}

    fact = out["fact"].copy()
    fact["date_id"] = pd.to_datetime(fact["date_id"], errors="coerce")
    fact["region_id"] = pd.to_numeric(fact["region_id"], errors="coerce").astype("Int64")
    fact["eco_id"] = pd.to_numeric(fact["eco_id"], errors="coerce").astype("Int64")
    fact["weather_id"] = pd.to_numeric(fact["weather_id"], errors="coerce").astype("Int64")

    for col in [
        "price_rice",
        "price_fert",
        "avg_temperature",
        "max_temp",
        "min_temp",
        "rainfall_mm",
        "avg_wind_speed",
        "production_ton",
        "area_harvest_ha",
        "yield_kg_per_ha",
        "cpi_change_value",
    ]:
        if col in fact.columns:
            fact[col] = pd.to_numeric(fact[col], errors="coerce")

    for col in ["rice_id", "fert_id"]:
        if col in fact.columns:
            fact[col] = fact[col].astype("string").str.strip()

    out["fact"] = fact

    for dim_name in [
        "dim_date",
        "dim_economy",
        "dim_fertilizer_type",
        "dim_region",
        "dim_rice_type",
        "dim_weather",
    ]:
        dim = out[dim_name].copy()
        if "date_id" in dim.columns:
            dim["date_id"] = pd.to_datetime(dim["date_id"], errors="coerce")
        for col in dim.columns:
            if dim[col].dtype == "object":
                dim[col] = dim[col].astype("string").str.strip()
        out[dim_name] = dim

    return out


# + Tạo bảng daily context ở mức date_id + region_id
def build_daily_context(fact_df: pd.DataFrame) -> pd.DataFrame:
    base = (
        fact_df.groupby(["date_id", "region_id"], as_index=False)
        .agg(
            eco_id=("eco_id", "last"),
            weather_id=("weather_id", "last"),
            avg_temperature=("avg_temperature", "mean"),
            max_temp=("max_temp", "mean"),
            min_temp=("min_temp", "mean"),
            rainfall_mm=("rainfall_mm", "mean"),
            avg_wind_speed=("avg_wind_speed", "mean"),
            production_ton=("production_ton", "mean"),
            area_harvest_ha=("area_harvest_ha", "mean"),
            yield_kg_per_ha=("yield_kg_per_ha", "mean"),
            cpi_change_value=("cpi_change_value", "mean"),
        )
    )
    return base


# + Tách giá gạo theo mức date_id + region_id + rice_id, chỉ giữ 3 loại gạo dùng cho mô hình
def build_daily_rice_target(
    fact_df: pd.DataFrame,
) -> pd.DataFrame:
    allowed_rice_ids = ["R01", "R02", "R03"]

    rice_rows = fact_df.loc[
        fact_df["rice_id"].notna() & fact_df["price_rice"].notna(),
        ["date_id", "region_id", "rice_id", "price_rice"],
    ].copy()

    rice_rows["rice_id"] = rice_rows["rice_id"].astype(str).str.strip()
    rice_rows = rice_rows.loc[rice_rows["rice_id"].isin(allowed_rice_ids)].copy()

    rice_daily = (
        rice_rows.groupby(["date_id", "region_id", "rice_id"], as_index=False)
        .agg(price_rice=("price_rice", "mean"))
    )
    return rice_daily


# + Tách giá phân theo mức date_id + region_id rồi pivot rộng theo fert_id
def build_daily_fertilizer_wide(fact_df: pd.DataFrame) -> pd.DataFrame:
    fert_rows = fact_df.loc[
        fact_df["fert_id"].notna() & fact_df["price_fert"].notna(),
        ["date_id", "region_id", "fert_id", "price_fert"],
    ].copy()

    fert_wide = (
        fert_rows.pivot_table(
            index=["date_id", "region_id"],
            columns="fert_id",
            values="price_fert",
            aggfunc="mean",
        )
        .reset_index()
    )

    fert_wide.columns = [
        col if col in ["date_id", "region_id"] else f"price_fert_{col}"
        for col in fert_wide.columns
    ]
    return fert_wide


# + Join fact đã tách với các dimension để tạo base daily hoàn chỉnh
def build_daily_model_base(source: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    source = clean_source_tables(source)

    fact = source["fact"]
    dim_date = source["dim_date"]
    dim_economy = source["dim_economy"]
    dim_region = source["dim_region"]
    dim_rice = source["dim_rice_type"]
    dim_weather = source["dim_weather"]

    daily_context = build_daily_context(fact)
    daily_rice = build_daily_rice_target(fact)
    daily_fert = build_daily_fertilizer_wide(fact)

    daily = (
        daily_rice.merge(daily_context, on=["date_id", "region_id"], how="left")
        .merge(daily_fert, on=["date_id", "region_id"], how="left")
    )

    daily = daily.merge(
        dim_date[["date_id", "year", "quarter", "month", "day", "crop_season"]],
        on="date_id",
        how="left",
    )

    daily = daily.merge(
        dim_region[["region_id", "region_name"]],
        on="region_id",
        how="left",
    )

    daily = daily.merge(
        dim_rice[["rice_id", "rice_name", "rice_group", "unit"]],
        on="rice_id",
        how="left",
    )

    daily = daily.merge(
        dim_economy[["eco_id", "cpi_status", "market_signal", "unit"]].rename(columns={"unit": "economy_unit"}),
        on="eco_id",
        how="left",
    )

    daily = daily.merge(
        dim_weather[["weather_id", "weather_type", "severity_level", "agriculture_impact"]],
        on="weather_id",
        how="left",
    )

    daily = daily.sort_values(["region_id", "rice_id", "date_id"]).reset_index(drop=True)
    return daily


# + Gom daily base về panel tháng để huấn luyện mô hình
def build_monthly_panel(
    daily_df: pd.DataFrame,
    horizons: Tuple[int, ...] = (1, 2, 3),
    max_target_lag: int = 6,
    max_feature_lag: int = 3,
    rolling_windows: Tuple[int, ...] = (2, 3, 6),
) -> pd.DataFrame:
    data = daily_df.copy()
    data["date_id"] = pd.to_datetime(data["date_id"], errors="coerce")
    data["month_date"] = data["date_id"].dt.to_period("M").dt.to_timestamp()

    fert_cols = [c for c in data.columns if str(c).startswith("price_fert_")]

    numeric_mean_cols = [
        "price_rice",
        "avg_temperature",
        "max_temp",
        "min_temp",
        "rainfall_mm",
        "avg_wind_speed",
        "production_ton",
        "area_harvest_ha",
        "yield_kg_per_ha",
        "cpi_change_value",
        *fert_cols,
    ]
    numeric_mean_cols = [c for c in numeric_mean_cols if c in data.columns]

    categorical_cols = [
        "region_name",
        "rice_name",
        "rice_group",
        "crop_season",
        "cpi_status",
        "market_signal",
        "weather_type",
        "severity_level",
        "agriculture_impact",
        "eco_id",
        "weather_id",
        "quarter",
        "month",
        "year",
    ]
    categorical_cols = [c for c in categorical_cols if c in data.columns]

    agg_map = {c: "mean" for c in numeric_mean_cols}
    agg_map.update({c: _safe_mode for c in categorical_cols})

    panel = (
        data.groupby(["month_date", "region_id", "rice_id"], as_index=False)
        .agg(agg_map)
        .rename(columns={"month_date": "date_id"})
        .sort_values(["region_id", "rice_id", "date_id"])
        .reset_index(drop=True)
    )

    panel["month_num"] = pd.to_datetime(panel["date_id"]).dt.month
    panel["year_num"] = pd.to_datetime(panel["date_id"]).dt.year
    panel["month_sin"] = np.sin(2 * np.pi * panel["month_num"] / 12.0)
    panel["month_cos"] = np.cos(2 * np.pi * panel["month_num"] / 12.0)

    group_keys = ["region_id", "rice_id"]

    lag_frames = []
    for lag in range(1, max_target_lag + 1):
        lag_frames.append(
            panel.groupby(group_keys)["price_rice"].shift(lag).rename(f"price_rice_lag_{lag}")
        )

    for col in numeric_mean_cols:
        if col == "price_rice":
            continue
        for lag in range(1, max_feature_lag + 1):
            lag_frames.append(
                panel.groupby(group_keys)[col].shift(lag).rename(f"{col}_lag_{lag}")
            )

    for col in ["price_rice", *[c for c in numeric_mean_cols if c != "price_rice"]]:
        for window in rolling_windows:
            lag_frames.append(
                panel.groupby(group_keys)[col]
                .transform(lambda s: s.rolling(window, min_periods=1).mean())
                .rename(f"{col}_roll_mean_{window}")
            )
            lag_frames.append(
                panel.groupby(group_keys)[col]
                .transform(lambda s: s.rolling(window, min_periods=1).std())
                .rename(f"{col}_roll_std_{window}")
            )

    feature_block = pd.concat(lag_frames, axis=1)
    panel = pd.concat([panel, feature_block], axis=1)

    prev_price = panel.groupby(group_keys)["price_rice"].shift(1)
    panel["price_rice_diff_1"] = panel["price_rice"] - prev_price
    panel["price_rice_pct_change_1"] = np.where(
        prev_price.notna() & (prev_price != 0),
        (panel["price_rice"] - prev_price) / prev_price,
        np.nan,
    )

    for h in horizons:
        future_price = panel.groupby(group_keys)["price_rice"].shift(-h)
        panel[f"target_t_plus_{h}"] = future_price
        panel[f"direction_t_plus_{h}"] = np.where(
            future_price > panel["price_rice"],
            1,
            np.where(future_price < panel["price_rice"], 0, np.nan),
        )

    panel["row_is_latest"] = False
    latest_idx = panel.groupby(group_keys)["date_id"].idxmax()
    panel.loc[latest_idx, "row_is_latest"] = True

    return panel.reset_index(drop=True)


# + Khai báo schema cuối cùng cho notebook train dùng lại
def infer_model_schema(panel_df: pd.DataFrame) -> SchemaInfo:
    id_cols = ["date_id", "region_id", "rice_id"]
    target_col = "price_rice"
    group_cols = ["region_id", "rice_id"]

    forbidden_cols = set(id_cols + ["row_is_latest"])
    forbidden_cols.update([c for c in panel_df.columns if c.startswith("target_t_plus_")])
    forbidden_cols.update([c for c in panel_df.columns if c.startswith("direction_t_plus_")])

    categorical_cols = panel_df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c not in forbidden_cols]

    numeric_cols = panel_df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in forbidden_cols and c != target_col]

    return SchemaInfo(
        date_col="date_id",
        target_col=target_col,
        group_cols=group_cols,
        id_cols=id_cols,
        numeric_feature_cols=numeric_cols,
        categorical_feature_cols=categorical_cols,
    )


# + Tách X y cho từng horizon và từng bài toán
def prepare_xy(
    panel_df: pd.DataFrame,
    horizon: int,
    task: str = "regression",
) -> Tuple[pd.DataFrame, pd.Series, SchemaInfo]:
    task = task.strip().lower()
    schema = infer_model_schema(panel_df)

    if task == "regression":
        target_name = f"target_t_plus_{horizon}"
    elif task == "classification":
        target_name = f"direction_t_plus_{horizon}"
    else:
        raise ValueError("task chỉ nhận 'regression' hoặc 'classification'.")

    use_cols = (
        schema.id_cols
        + schema.numeric_feature_cols
        + schema.categorical_feature_cols
        + [target_name]
    )

    model_df = panel_df[use_cols].copy()
    model_df = model_df.dropna(subset=[target_name]).reset_index(drop=True)

    feature_cols = schema.numeric_feature_cols + schema.categorical_feature_cols
    X = model_df[feature_cols].copy()
    y = model_df[target_name].copy()

    return X, y, schema