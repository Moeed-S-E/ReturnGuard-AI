"""Train the ReturnGuard LightGBM model and persist all artifacts.

Reproduces the notebook pipeline as a standalone script so the model,
metrics, and policy_config.json can be regenerated without Jupyter.

    python train_model.py

Writes to artifacts/:
    model.joblib, metrics.json, policy_config.json
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

SEED = 42
HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"

LABEL_MAP = {"Legitimate": 0, "Policy Abuser": 1, "Fraudulent Return": 2, "Wardrobing": 3}
INT_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

CAT_COLS = ["customer_segment", "country", "platform", "device_type",
            "payment_method", "product_category", "return_reason", "shipping_carrier"]
BOOL_COLS = ["is_high_value_item", "discount_used", "item_returned_opened",
             "return_packaging_intact", "photo_evidence_provided", "tracking_number_valid",
             "address_change_before_delivery", "refund_to_different_account",
             "multiple_accounts_flag", "review_left_after_return"]
NUMERIC_RAW = ["age", "account_age_days", "avg_order_value_usd",
               "refund_amount_requested_usd", "days_to_return",
               "total_orders_lifetime", "total_returns_lifetime", "return_rate_pct",
               "customer_support_contacts", "previous_dispute_count",
               "wishlist_to_cart_time_hrs"]
ENGINEERED = ["order_to_return_days", "return_ratio", "risk_exposure",
              "category_risk_index", "dispute_frequency"]
FEATURE_COLS = NUMERIC_RAW + BOOL_COLS + CAT_COLS + ENGINEERED

TUNE = {0: ("thresh_legitimate", "precision"), 1: ("thresh_policy_abuser", "recall"),
        2: ("thresh_fraudulent", "recall"), 3: ("thresh_wardrobing", "recall")}
THRESHOLDS = np.round(np.arange(0.30, 0.91, 0.05), 2)


def _resolve_data() -> Path:
    for p in [HERE.parent / "datasets" / "ecommerce_return_abuse_dataset.csv",
              HERE / "datasets" / "ecommerce_return_abuse_dataset.csv"]:
        if p.exists():
            return p
    raise FileNotFoundError("ecommerce_return_abuse_dataset.csv not found under datasets/")


def engineer_features(df: pd.DataFrame, cat_risk_map=None):
    df = df.copy()
    if "abuse_label" not in df.columns:
        df["abuse_label"] = df["abuse_type"].map(LABEL_MAP)
    df["order_to_return_days"] = (
        (pd.to_datetime(df["return_date"]) - pd.to_datetime(df["order_date"]))
        .dt.days.clip(lower=0).astype(int))
    df["return_ratio"] = df["total_returns_lifetime"] / df["total_orders_lifetime"].clip(lower=1)
    df["risk_exposure"] = df["refund_amount_requested_usd"] * (1 + df["previous_dispute_count"])
    if cat_risk_map is None:
        cat_risk_map = (df.groupby("product_category")["abuse_label"]
                        .apply(lambda s: (s == 2).mean()).to_dict())
    df["category_risk_index"] = df["product_category"].map(cat_risk_map).fillna(0.0)
    df["dispute_frequency"] = df["previous_dispute_count"] / (df["account_age_days"] / 30).clip(lower=1)
    return df, cat_risk_map


def _xy(part):
    X = part[FEATURE_COLS].copy()
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
    return X, part["abuse_label"].astype(int)


def tune_threshold(probs_val, y_val, class_idx, metric):
    truth = (np.asarray(y_val) == class_idx).astype(int)
    best_t, best_s = 0.50, -1.0
    for t in THRESHOLDS:
        preds = (probs_val[:, class_idx] >= t).astype(int)
        s = (recall_score(truth, preds, zero_division=0) if metric == "recall"
             else precision_score(truth, preds, zero_division=0))
        if s > best_s:
            best_s, best_t = s, float(t)
    return best_t, best_s


def train():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(_resolve_data())
    df["abuse_label"] = df["abuse_type"].map(LABEL_MAP)

    tmp, test = train_test_split(df, test_size=0.2, stratify=df["abuse_label"], random_state=SEED)
    train_df, val_df = train_test_split(tmp, test_size=0.1/0.8,
                                        stratify=tmp["abuse_label"], random_state=SEED)
    train_df, cat_risk_map = engineer_features(train_df, None)
    val_df, _ = engineer_features(val_df, cat_risk_map)
    test_df, _ = engineer_features(test_df, cat_risk_map)

    X_train, y_train = _xy(train_df)
    X_val, y_val = _xy(val_df)
    X_test, y_test = _xy(test_df)

    model = LGBMClassifier(objective="multiclass", num_class=4, class_weight="balanced",
                           n_estimators=1000, learning_rate=0.05, num_leaves=63,
                           random_state=SEED, n_jobs=-1)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="multi_logloss",
              categorical_feature=CAT_COLS,
              callbacks=[early_stopping(50), log_evaluation(100)])

    y_pred = model.predict(X_test)
    names = [INT_TO_LABEL[i] for i in range(4)]
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    per_class = f1_score(y_test, y_pred, average=None).tolist()
    print(f"Macro F1: {macro_f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=names, digits=3))

    joblib.dump(model, ARTIFACTS / "model.joblib")
    (ARTIFACTS / "metrics.json").write_text(json.dumps({
        "macro_f1": macro_f1,
        "per_class_f1": {names[i]: per_class[i] for i in range(4)},
        "label_map": LABEL_MAP,
        "best_iteration": int(model.best_iteration_ or 1000),
    }, indent=2))

    probs_val = model.predict_proba(X_val)
    policy = {"fee_pkr": 950}
    for idx, (key, metric) in TUNE.items():
        t, s = tune_threshold(probs_val, y_val.values, idx, metric)
        policy[key] = t
        print(f"{INT_TO_LABEL[idx]:18s} {key:22s} = {t:.2f} ({metric}={s:.3f})")

    (ARTIFACTS / "policy_config.json").write_text(json.dumps({
        "policy_config": policy,
        "cat_risk_map": {str(k): float(v) for k, v in cat_risk_map.items()},
        "feature_cols": FEATURE_COLS,
    }, indent=2))
    print("Saved model.joblib, metrics.json, policy_config.json to", ARTIFACTS)
    return model


if __name__ == "__main__":
    train()
