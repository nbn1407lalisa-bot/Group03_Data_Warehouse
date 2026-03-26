# + Tạo dự báo giá gạo 1-2-3 tháng từ model đã huấn luyện
from __future__ import annotations

import os
from typing import List

import joblib
import pandas as pd

from data_processing import (
    load_all_source_tables,
    clean_source_tables,
    build_daily_model_base,
    build_monthly_panel,
)


# + Đọc model bundle đã lưu
def load_model_bundle(model_dir: str, horizon: int, task: str):
    task_prefix = "reg" if task == "regression" else "dir"
    path = os.path.join(model_dir, f"rice_price_{task_prefix}_h{horizon}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy model tại {path}")
    return joblib.load(path)


# + Tạo forecast mới nhất cho từng loại gạo
def predict_latest(
    project_id: str = "dwm-final-487216",
    dataset_id: str = "nong_san_dbscl",
    model_dir: str = "../models",
    horizons: List[int] = [1, 2, 3],
) -> pd.DataFrame:
    source = load_all_source_tables(project_id=project_id, dataset_id=dataset_id)
    source = clean_source_tables(source)

    daily_base = build_daily_model_base(source)
    panel = build_monthly_panel(
        daily_base,
        horizons=(1, 2, 3),
        max_target_lag=6,
        max_feature_lag=3,
        rolling_windows=(2, 3, 6),
    )

    latest_rows = panel.loc[panel["row_is_latest"]].copy().reset_index(drop=True)

    outputs = []
    for horizon in horizons:
        reg_bundle = load_model_bundle(model_dir, horizon, task="regression")
        cls_bundle = load_model_bundle(model_dir, horizon, task="classification")

        X_reg = latest_rows.reindex(columns=reg_bundle["feature_cols"])
        X_cls = latest_rows.reindex(columns=cls_bundle["feature_cols"])

        out = latest_rows[
            ["date_id", "region_id", "rice_id", "region_name", "rice_name", "rice_group", "price_rice"]
        ].copy()

        out["forecast_horizon_month"] = horizon
        out["forecast_month"] = (
            pd.to_datetime(out["date_id"]) + pd.DateOffset(months=horizon)
        ).dt.to_period("M").dt.to_timestamp()

        out["predicted_price"] = reg_bundle["model"].predict(X_reg)
        out["predicted_direction"] = cls_bundle["model"].predict(X_cls)
        out["predicted_direction_label"] = out["predicted_direction"].map(
            {1: "Tăng", 0: "Giảm/Đi ngang"}
        )

        outputs.append(out)

    forecast_df = pd.concat(outputs, ignore_index=True)
    forecast_df["predicted_price"] = forecast_df["predicted_price"].round(2)
    return forecast_df


if __name__ == "__main__":
    forecast_df = predict_latest(
        project_id="dwm-final-487216",
        dataset_id="nong_san_dbscl",
        model_dir="../models",
        horizons=[1, 2, 3],
    )

    os.makedirs("../outputs", exist_ok=True)
    forecast_df.to_csv("../outputs/latest_rice_price_forecast.csv", index=False, encoding="utf-8-sig")
    print(forecast_df.head(20))
    print("Đã lưu forecast tại ../outputs/latest_rice_price_forecast.csv")