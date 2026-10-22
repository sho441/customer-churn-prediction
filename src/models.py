"""Models definition and training/evaluation module for customer churn prediction.

This module provides model dictionaries, scaling requirements, and
Target Encoding functions used in Cross-Validation.
"""

from typing import Dict, List, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.svm import LinearSVC


def get_models(best_lgb_params: dict = None) -> Dict[str, object]:
    """学習に使用する機械学習モデルの辞書を返します.

    Args:
        best_lgb_params (dict, optional): Optuna等で最適化されたLGBMパラメータ。

    Returns:
        Dict[str, object]: モデル名とインスタンスの辞書。
    """
    lgb_params = {
        "random_state": 42,
        "verbose": -1,
    }
    if best_lgb_params:
        lgb_params.update(best_lgb_params)

    models = {
        "LogisticRegression": LogisticRegression(
            random_state=42, max_iter=1000
        ),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42),
        "LightGBM": lgb.LGBMClassifier(**lgb_params),
        "XGBoost": xgb.XGBClassifier(random_state=42, eval_metric="logloss"),
        "CatBoost": CatBoostClassifier(random_state=42, verbose=0),
        "LinearSVC": LinearSVC(random_state=42, max_iter=1000),
    }
    return models


def get_scaling_required_models() -> List[str]:
    """標準化（StandardScaler）が推奨されるモデルのリストを返します."""
    return ["LogisticRegression", "LinearSVC"]


def apply_target_encoding(
    train_fold: pd.DataFrame,
    val_fold: pd.DataFrame,
    cat_cols: List[str],
    target_col: str = "Churn",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """外側/内側のTrainデータからターゲットエンコーディングマップを計算し適用します.

    Args:
        train_fold (pd.DataFrame): 訓練データフォールド。
        val_fold (pd.DataFrame): 検証データフォールド。
        cat_cols (List[str]): エンコード対象のカテゴリカル変数リスト。
        target_col (str): 目的変数の列名。

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: エンコード済みの訓練・検証データフレーム。
    """
    tr_enc = train_fold.copy()
    va_enc = val_fold.copy()

    for col in cat_cols:
        target_name = f"{col}_te"
        tr_series = tr_enc[col].astype(str)
        va_series = va_enc[col].astype(str)

        encoding_map = (
            pd.DataFrame({col: tr_series, target_col: tr_enc[target_col]})
            .groupby(col)[target_col]
            .mean()
            .to_dict()
        )

        tr_mapped = (
            tr_series.map(encoding_map)
            .fillna(0.5)
            .astype("float32")
        )
        va_mapped = (
            va_series.map(encoding_map)
            .fillna(0.5)
            .astype("float32")
        )

        if target_name in tr_enc.columns:
            tr_enc = tr_enc.drop(columns=[target_name])
        if target_name in val_fold.columns and target_name in va_enc.columns:
            va_enc = va_enc.drop(columns=[target_name])

        tr_enc[target_name] = tr_mapped
        va_enc[target_name] = va_mapped

    return tr_enc, va_enc
