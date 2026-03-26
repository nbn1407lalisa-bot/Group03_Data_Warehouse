# + Huấn luyện và chọn mô hình tốt nhất cho dự báo giá gạo 1-2-3 tháng tới
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from data_processing import prepare_xy, infer_model_schema


# + Chia tập train-valid theo kiểu expanding window trên trục thời gian
def build_expanding_splits(
    panel_df: pd.DataFrame,
    date_col: str = "date_id",
    min_train_months: int = 12,
    valid_months: int = 1,
    max_splits: int = 6,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    dates = pd.to_datetime(panel_df[date_col])
    unique_months = sorted(dates.dt.to_period("M").astype(str).unique().tolist())

    splits = []
    for i in range(min_train_months, len(unique_months) - valid_months + 1):
        train_months = unique_months[:i]
        valid_block = unique_months[i : i + valid_months]

        train_idx = panel_df.index[dates.dt.to_period("M").astype(str).isin(train_months)].to_numpy()
        valid_idx = panel_df.index[dates.dt.to_period("M").astype(str).isin(valid_block)].to_numpy()

        if len(train_idx) > 0 and len(valid_idx) > 0:
            splits.append((train_idx, valid_idx))

    if len(splits) > max_splits:
        splits = splits[-max_splits:]

    if not splits:
        raise ValueError("Không đủ số tháng để chia time split. Hãy giảm min_train_months.")

    return splits


# + Tạo OneHotEncoder tương thích nhiều version sklearn
def make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# + Dựng pipeline tiền xử lý cho model
def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    encoding: str = "onehot",
    scale_numeric: bool = False,
) -> ColumnTransformer:
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(steps=num_steps)

    if encoding == "onehot":
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", make_onehot()),
            ]
        )
    else:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]
        )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )


# + Bộ mô hình hồi quy để benchmark
def get_regression_models(numeric_cols: List[str], categorical_cols: List[str]) -> Dict[str, Pipeline]:
    models = {
        "elastic_net": Pipeline(
            steps=[
                ("prep", build_preprocessor(numeric_cols, categorical_cols, encoding="onehot", scale_numeric=True)),
                ("model", ElasticNet(alpha=0.01, l1_ratio=0.3, max_iter=20000, random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("prep", build_preprocessor(numeric_cols, categorical_cols, encoding="onehot", scale_numeric=False)),
                ("model", RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                )),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("prep", build_preprocessor(numeric_cols, categorical_cols, encoding="onehot", scale_numeric=False)),
                ("model", ExtraTreesRegressor(
                    n_estimators=400,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                )),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("prep", build_preprocessor(numeric_cols, categorical_cols, encoding="ordinal", scale_numeric=False)),
                ("model", GradientBoostingRegressor(
                    learning_rate=0.05,
                    n_estimators=250,
                    max_depth=3,
                    random_state=42,
                )),
            ]
        ),
    }

    try:
        from catboost import CatBoostRegressor

        models["catboost"] = Pipeline(
            steps=[
                ("prep", build_preprocessor(numeric_cols, categorical_cols, encoding="ordinal", scale_numeric=False)),
                ("model", CatBoostRegressor(
                    depth=6,
                    learning_rate=0.05,
                    iterations=300,
                    loss_function="RMSE",
                    random_seed=42,
                    verbose=False,
                )),
            ]
        )
    except Exception:
        pass

    return models


# + Bộ mô hình phân loại cho nhãn tăng giảm
def get_classification_models(numeric_cols: List[str], categorical_cols: List[str]) -> Dict[str, Pipeline]:
    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("prep", build_preprocessor(numeric_cols, categorical_cols, encoding="onehot", scale_numeric=True)),
                ("model", LogisticRegression(max_iter=5000)),
            ]
        ),
        "random_forest_cls": Pipeline(
            steps=[
                ("prep", build_preprocessor(numeric_cols, categorical_cols, encoding="onehot", scale_numeric=False)),
                ("model", RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                )),
            ]
        ),
        "extra_trees_cls": Pipeline(
            steps=[
                ("prep", build_preprocessor(numeric_cols, categorical_cols, encoding="onehot", scale_numeric=False)),
                ("model", ExtraTreesClassifier(
                    n_estimators=400,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                )),
            ]
        ),
        "gradient_boosting_cls": Pipeline(
            steps=[
                ("prep", build_preprocessor(numeric_cols, categorical_cols, encoding="ordinal", scale_numeric=False)),
                ("model", GradientBoostingClassifier(
                    learning_rate=0.05,
                    n_estimators=250,
                    max_depth=3,
                    random_state=42,
                )),
            ]
        ),
    }

    try:
        from catboost import CatBoostClassifier

        models["catboost_cls"] = Pipeline(
            steps=[
                ("prep", build_preprocessor(numeric_cols, categorical_cols, encoding="ordinal", scale_numeric=False)),
                ("model", CatBoostClassifier(
                    depth=6,
                    learning_rate=0.05,
                    iterations=300,
                    loss_function="Logloss",
                    random_seed=42,
                    verbose=False,
                )),
            ]
        )
    except Exception:
        pass

    return models


# + Tính các chỉ số cho bài toán hồi quy
def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    denom = np.where(np.abs(y_true) < 1e-8, np.nan, np.abs(y_true))
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


# + Tính chỉ số cho bài toán tăng giảm
def classification_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    return {"accuracy": float(accuracy_score(y_true, y_pred))}


# + Benchmark các model cho một horizon
def evaluate_models_for_horizon(
    panel_df: pd.DataFrame,
    horizon: int,
    min_train_months: int = 12,
    valid_months: int = 1,
) -> pd.DataFrame:
    panel_work = panel_df.copy().reset_index(drop=True)
    splits = build_expanding_splits(
        panel_work,
        date_col="date_id",
        min_train_months=min_train_months,
        valid_months=valid_months,
        max_splits=6,
    )

    schema = infer_model_schema(panel_work)

    # hồi quy
    X_reg, y_reg, _ = prepare_xy(panel_work, horizon=horizon, task="regression")
    reg_df = panel_work.loc[y_reg.index].reset_index(drop=True)
    reg_splits = build_expanding_splits(
        reg_df.assign(date_id=panel_work.loc[y_reg.index, "date_id"].values),
        date_col="date_id",
        min_train_months=min_train_months,
        valid_months=valid_months,
        max_splits=6,
    )

    reg_models = get_regression_models(schema.numeric_feature_cols, schema.categorical_feature_cols)

    rows = []
    for model_name, model in reg_models.items():
        for fold_id, (tr_idx, va_idx) in enumerate(reg_splits, start=1):
            fitted = clone(model)
            fitted.fit(X_reg.iloc[tr_idx], y_reg.iloc[tr_idx])
            pred = fitted.predict(X_reg.iloc[va_idx])

            m = regression_metrics(y_reg.iloc[va_idx], pred)
            m.update(
                {
                    "task": "regression",
                    "horizon": horizon,
                    "model_name": model_name,
                    "fold": fold_id,
                }
            )
            rows.append(m)

    # phân loại
    X_cls, y_cls, _ = prepare_xy(panel_work, horizon=horizon, task="classification")
    cls_df = panel_work.loc[y_cls.index].reset_index(drop=True)
    cls_splits = build_expanding_splits(
        cls_df.assign(date_id=panel_work.loc[y_cls.index, "date_id"].values),
        date_col="date_id",
        min_train_months=min_train_months,
        valid_months=valid_months,
        max_splits=6,
    )

    cls_models = get_classification_models(schema.numeric_feature_cols, schema.categorical_feature_cols)

    for model_name, model in cls_models.items():
        valid_fold_count = 0
        for fold_id, (tr_idx, va_idx) in enumerate(cls_splits, start=1):
            y_train = y_cls.iloc[tr_idx]
            y_valid = y_cls.iloc[va_idx]

            if y_train.nunique() < 2 or y_valid.nunique() < 2:
                continue

            fitted = clone(model)
            fitted.fit(X_cls.iloc[tr_idx], y_train)
            pred = fitted.predict(X_cls.iloc[va_idx])

            m = classification_metrics(y_valid, pred)
            m.update(
                {
                    "task": "classification",
                    "horizon": horizon,
                    "model_name": model_name,
                    "fold": fold_id,
                }
            )
            rows.append(m)
            valid_fold_count += 1

    return pd.DataFrame(rows)


# + Train model tốt nhất cho từng horizon rồi lưu ra thư mục models
def train_best_models(
    panel_df: pd.DataFrame,
    model_dir: str = "../models",
    horizons: Tuple[int, ...] = (1, 2, 3),
    min_train_months: int = 12,
    valid_months: int = 1,
):
    os.makedirs(model_dir, exist_ok=True)

    all_scores = []
    selected_rows = []

    schema = infer_model_schema(panel_df)

    for horizon in horizons:
        score_df = evaluate_models_for_horizon(
            panel_df=panel_df,
            horizon=horizon,
            min_train_months=min_train_months,
            valid_months=valid_months,
        )
        all_scores.append(score_df)

        reg_summary = (
            score_df[score_df["task"] == "regression"]
            .groupby(["task", "horizon", "model_name"], as_index=False)[["rmse", "mae", "mape", "r2"]]
            .mean()
            .sort_values(["rmse", "mae", "mape"], ascending=[True, True, True])
        )

        cls_summary = (
            score_df[score_df["task"] == "classification"]
            .groupby(["task", "horizon", "model_name"], as_index=False)[["accuracy"]]
            .mean()
            .sort_values(["accuracy"], ascending=[False])
        )

        best_reg_name = reg_summary.iloc[0]["model_name"]
        best_cls_name = cls_summary.iloc[0]["model_name"] if not cls_summary.empty else "dummy_classifier"

        X_reg, y_reg, _ = prepare_xy(panel_df, horizon=horizon, task="regression")
        reg_models = get_regression_models(schema.numeric_feature_cols, schema.categorical_feature_cols)
        best_reg_model = clone(reg_models[best_reg_name]).fit(X_reg, y_reg)

        reg_bundle = {
            "task": "regression",
            "horizon": horizon,
            "model_name": best_reg_name,
            "model": best_reg_model,
            "feature_cols": X_reg.columns.tolist(),
        }
        reg_path = os.path.join(model_dir, f"rice_price_reg_h{horizon}.joblib")
        joblib.dump(reg_bundle, reg_path)

        X_cls, y_cls, _ = prepare_xy(panel_df, horizon=horizon, task="classification")
        if y_cls.nunique() < 2:
            cls_model = DummyClassifier(strategy="most_frequent").fit(X_cls, y_cls)
        else:
            cls_models = get_classification_models(schema.numeric_feature_cols, schema.categorical_feature_cols)
            cls_model = clone(cls_models[best_cls_name]).fit(X_cls, y_cls)

        cls_bundle = {
            "task": "classification",
            "horizon": horizon,
            "model_name": best_cls_name,
            "model": cls_model,
            "feature_cols": X_cls.columns.tolist(),
        }
        cls_path = os.path.join(model_dir, f"rice_price_dir_h{horizon}.joblib")
        joblib.dump(cls_bundle, cls_path)

        selected_rows.append(
            {
                "horizon": horizon,
                "best_regression_model": best_reg_name,
                "best_classification_model": best_cls_name,
                "regression_model_path": reg_path,
                "classification_model_path": cls_path,
            }
        )

    all_scores_df = pd.concat(all_scores, ignore_index=True)
    selected_df = pd.DataFrame(selected_rows)

    all_scores_df.to_csv(os.path.join(model_dir, "benchmark_scores.csv"), index=False)
    selected_df.to_csv(os.path.join(model_dir, "selected_models.csv"), index=False)

    metadata = {
        "note": "Direct forecast 1-2-3 tháng từ monthly panel của fact_agriculture_final",
        "selected_models": selected_df.to_dict(orient="records"),
    }
    with open(os.path.join(model_dir, "training_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return all_scores_df, selected_df