"""Main training and inference script for customer churn prediction.

This script executes data loading, feature engineering, cross-validation,
Optuna hyperparameter tuning, model blending, and creates a submission file.
"""

from typing import Dict, List
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# src ディレクトリからモジュールをインポート
from src.features import (
    add_charge_and_service_features,
    add_frequency_features,
    add_ngram_features,
    clean_churn_data,
)
from src.models import apply_target_encoding, get_models, get_scaling_required_models

# Optunaのログを抑制
optuna.logging.set_verbosity(optuna.logging.WARNING)


def main():
    """メイン実行関数."""
    print("--- 1. データの読み込み ---")
    # ※適宜、ご自身のデータパスに合わせて修正してください
    train_path = "./data/train.csv"
    test_path = "./data/test.csv"
    original_path = "./data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    original_df = pd.read_csv(original_path)

    print("--- 2. データクリーニング ---")
    train_df, test_df, original_df = clean_churn_data(
        train_df, test_df, original_df
    )

    # 識別子（id）の退避
    test_id = test_df["id"].copy() if "id" in test_df.columns else None

    print("--- 3. 特徴量エンジニアリング ---")
    # カテゴリカル変数の定義
    exclude_cols = ["id", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    categorical_cols = [
        col for col in train_df.columns if col not in exclude_cols
    ][:16]

    # 1. 頻度エンコーディング
    train_df, test_df, original_df, freq_feats = add_frequency_features(
        train_df, test_df, original_df, categorical_cols
    )

    # 2. TotalCharges の欠損値補填の準備（中間値の計算例）
    no_churn_df = train_df[train_df["Churn"] == 0].copy()
    calc_total = no_churn_df["tenure"] * no_churn_df["MonthlyCharges"]
    is_eq = np.isclose(calc_total, no_churn_df["TotalCharges"], rtol=1e-3, atol=1e-2)
    mismatch_df = no_churn_df[~is_eq & (no_churn_df["tenure"] > 0)].copy()
    ratio = mismatch_df["TotalCharges"] / (
        mismatch_df["tenure"] * mismatch_df["MonthlyCharges"]
    )
    filtered_ratio = ratio[(ratio >= 0.0) & (ratio <= 1.0)]
    median_val = filtered_ratio.median()

    # original_df の欠損補填
    missing_mask = original_df["TotalCharges"].isnull()
    original_df.loc[missing_mask, "TotalCharges"] = median_val * (
        original_df.loc[missing_mask, "tenure"]
        * original_df.loc[missing_mask, "MonthlyCharges"]
    )

    # 3. 料金・サービスフラグ特徴量
    train_df, test_df, original_df, charge_feats = add_charge_and_service_features(
        train_df, test_df, original_df, median_val
    )

    # 4. N-gram 特徴量
    base_cats = categorical_cols[:4]
    train_df, test_df, original_df, ngram_feats = add_ngram_features(
        train_df, test_df, original_df, base_cats
    )

    # すべての特徴量を統合
    all_features = list(
        set(
            categorical_cols
            + ["tenure", "MonthlyCharges", "TotalCharges"]
            + freq_feats
            + charge_feats
            + ngram_feats
        )
    )

    cat_cols_to_encode = list(set(categorical_cols + ngram_feats))
    base_features = [
        f for f in all_features
        if f not in ["Churn", "customerID", "id"] and f in train_df.columns
    ]

    print("--- 4. Optuna による LightGBM のハイパーパラメータチューニング ---")
    outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "random_state": 42,
            "verbose": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        fold_scores = []
        for tr_idx, va_idx in outer_cv.split(train_df, train_df["Churn"]):
            tr_fold = train_df.iloc[tr_idx].copy()
            va_fold = train_df.iloc[va_idx].copy()

            tr_enc, va_enc = apply_target_encoding(
                tr_fold, va_fold, cat_cols_to_encode
            )
            te_cols = [f"{c}_te" for c in cat_cols_to_encode]
            curr_feats = [
                f for f in base_features if f not in cat_cols_to_encode
            ] + te_cols
            curr_feats = [f for f in curr_feats if f in tr_enc.columns]

            X_tr, y_tr = tr_enc[curr_feats], tr_enc["Churn"]
            X_va, y_va = va_enc[curr_feats], va_enc["Churn"]

            model = lgb.LGBMClassifier(**params, n_estimators=500)
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
            )
            preds = model.predict_proba(X_va)[:, 1]
            fold_scores.append(roc_auc_score(y_va, preds))
        return np.mean(fold_scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)  # 試行回数は必要に応じて調整
    best_params = study.best_params
    print(f"Best Optuna Score: {study.best_value:.4f}")

    print("--- 5. アンサンブルモデルの学習と OOF 予測 ---")
    ensemble_models = get_models(best_lgb_params=best_params)
    scaling_needed = set(get_scaling_required_models())

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = {name: np.zeros(len(train_df)) for name in ensemble_models.keys()}
    y_trues = np.zeros(len(train_df))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_df, train_df["Churn"])):
        tr_fold = train_df.iloc[tr_idx].copy()
        va_fold = train_df.iloc[va_idx].copy()

        tr_enc, va_enc = apply_target_encoding(
            tr_fold, va_fold, cat_cols_to_encode
        )
        te_cols = [f"{c}_te" for c in cat_cols_to_encode]
        curr_feats = [
            f for f in base_features if f not in cat_cols_to_encode
        ] + te_cols
        curr_feats = [f for f in curr_feats if f in tr_enc.columns]

        X_tr, y_tr = tr_enc[curr_feats], tr_enc["Churn"]
        X_va, y_va = va_enc[curr_feats], va_enc["Churn"]
        y_trues[va_idx] = y_va.values

        for name, model in ensemble_models.items():
            clf = clone(model)
            if name in scaling_needed:
                scaler = StandardScaler()
                X_tr_p = scaler.fit_transform(X_tr)
                X_va_p = scaler.transform(X_va)
            else:
                X_tr_p, X_va_p = X_tr, X_va

            if name == "LightGBM":
                clf.fit(
                    X_tr_p,
                    y_tr,
                    eval_set=[(X_va_p, y_va)],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
            else:
                clf.fit(X_tr_p, y_tr)

            if hasattr(clf, "predict_proba"):
                preds = clf.predict_proba(X_va_p)[:, 1]
            else:
                decision = clf.decision_function(X_va_p)
                preds = 1 / (1 + np.exp(-decision))
            oof_preds[name][va_idx] = preds

    print("\n--- 単体モデルの OOF ROC-AUC スコア ---")
    for name, preds in oof_preds.items():
        print(f"{name:20s}: {roc_auc_score(y_trues, preds):.5f}")

    print("--- 6. 重み付きアンサンブルの最適化 ---")
    model_names = list(ensemble_models.keys())

    def obj_weights(weights):
        w = np.array(weights)
        w = w / np.sum(w)
        blended = np.zeros(len(y_trues))
        for i, name in enumerate(model_names):
            blended += w[i] * oof_preds[name]
        return -roc_auc_score(y_trues, blended)

    init_w = [1.0 / len(model_names)] * len(model_names)
    bounds = [(0.0, 1.0)] * len(model_names)
    constraints = {"type": "eq", "fun": lambda w: 1 - sum(w)}

    res = minimize(
        obj_weights, init_w, method="SLSQP", bounds=bounds, constraints=constraints
    )
    best_weights = res.x / np.sum(res.x)

    print("\n最適化されたモデルごとの重み:")
    for name, w in zip(model_names, best_weights):
        print(f"  {name:20s}: {w:.4f}")

    opt_blended_preds = np.zeros(len(y_trues))
    for i, name in enumerate(model_names):
        opt_blended_preds += best_weights[i] * oof_preds[name]
    print(f"\nOptimal Weighted Ensemble OOF AUC: {roc_auc_score(y_trues, opt_blended_preds):.5f}")


if __name__ == "__main__":
    main()
