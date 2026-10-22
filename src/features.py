"""Feature engineering module for customer churn prediction.

This module provides functions for data cleaning, frequency encoding,
custom charge calculations, service flags, and N-gram combinations.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd


def clean_churn_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    original_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """データフレームの基本的なクリーニングとターゲット変数の変換を行う."""
    train = train_df.copy()
    test = test_df.copy()
    orig = original_df.copy()

    # Churn カラムの Yes/No を 1/0 (int型) に変換
    churn_map = {"Yes": 1, "No": 0}
    if "Churn" in train.columns:
        train["Churn"] = train["Churn"].map(churn_map).astype(int)
    if "Churn" in orig.columns:
        orig["Churn"] = orig["Churn"].map(churn_map).astype(int)

    # TotalCharges 列を数値型に変換（変換不能な値は NaN）
    orig["TotalCharges"] = pd.to_numeric(
        orig["TotalCharges"], errors="coerce"
    )

    # 不要な customerID 列の削除
    if "customerID" in orig.columns:
        orig = orig.drop(columns=["customerID"])

    return train, test, orig


def add_frequency_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    original_df: pd.DataFrame,
    categorical_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """カテゴリカル変数に対して全データ基準の出現頻度（Frequency）を付与する."""
    train = train_df.copy()
    test = test_df.copy()
    orig = original_df.copy()

    all_df = pd.concat([train, test, orig], axis=0).reset_index(drop=True)
    added_features = []

    for col in categorical_cols:
        freq_col_name = f"{col}_freq"
        freq_map = all_df[col].value_counts().to_dict()

        train[freq_col_name] = train[col].map(freq_map).fillna(0).astype("float32")
        test[freq_col_name] = test[col].map(freq_map).fillna(0).astype("float32")
        orig[freq_col_name] = orig[freq_col_name].map(freq_map).fillna(0).astype("float32")

        if freq_col_name not in added_features:
            added_features.append(freq_col_name)

    return train, test, orig, added_features


def add_charge_and_service_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    original_df: pd.DataFrame,
    median_val: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """料金関連の特徴量およびサービス加入フラグを追加する."""
    dfs = [train_df, test_df, original_df]
    added_features = []

    for df in dfs:
        # 実効平均月額料金
        df["effective_monthly_charges"] = np.where(
            df["tenure"] > 0, df["TotalCharges"] / df["tenure"], 0.0
        ).astype("float32")

        # 推定基準月額料金
        df["estimated_baseline_charges"] = (
            median_val * df["MonthlyCharges"]
        ).astype("float32")

        # サービスフラグ
        df["is_fiber_optic"] = (
            df["InternetService"] == "Fiber optic"
        ).astype("float32")
        df["has_phone_service"] = (
            df["PhoneService"] == "Yes"
        ).astype("float32")

    feat_names = [
        "effective_monthly_charges",
        "estimated_baseline_charges",
        "is_fiber_optic",
        "has_phone_service",
    ]
    added_features.extend(feat_names)

    return train_df, test_df, original_df, added_features


def add_ngram_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    original_df: pd.DataFrame,
    base_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """主要なカテゴリカル変数を組み合わせた Bi-gram / Tri-gram 特徴量を生成する."""
    dfs = [train_df, test_df, original_df]
    added_features = []

    top_col0 = train_df[base_cols[0]].value_counts().head(8).index.tolist()
    top_col1 = train_df[base_cols[1]].value_counts().head(8).index.tolist()
    top_t2 = train_df[base_cols[2]].value_counts().head(6).index.tolist()

    for df in dfs:
        val0 = df[base_cols[0]].apply(lambda x: x if x in top_col0 else "Other")
        val1 = df[base_cols[1]].apply(lambda x: x if x in top_col1 else "Other")
        df["bigram_feature"] = (val0.astype(str) + "_" + val1.astype(str)).astype("category")

        vt0 = df[base_cols[0]].apply(lambda x: x if x in top_col0 else "Other")
        vt1 = df[base_cols[1]].apply(lambda x: x if x in top_col1 else "Other")
        vt2 = df[base_cols[2]].apply(lambda x: x if x in top_t2 else "Other")
        df["trigram_feature"] = (
            vt0.astype(str) + "_" + vt1.astype(str) + "_" + vt2.astype(str)
        ).astype("category")

        df["combined_ngram_feature"] = (
            df["bigram_feature"].astype(str)
            + "__"
            + df["trigram_feature"].astype(str)
        ).astype("category")

    ngram_cols = ["bigram_feature", "trigram_feature", "combined_ngram_feature"]
    added_features.extend(ngram_cols)

    return train_df, test_df, original_df, added_features
