"""ReturnGuard policy engine (modular, model-aligned).

Mirrors the validated logic in returnguard_prototype.ipynb. Loads the trained
LightGBM model and the persisted feature/threshold config, then scores a return
request into a policy action with SHAP-based reasons.

Artifacts expected in artifacts/:
    model.joblib        trained LGBMClassifier
    policy_config.json  tuned thresholds + fee + cat_risk_map + feature lists

Run train_model.py (or the notebook) first to produce these.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

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

SEVERITY = ["Fraudulent Return", "Wardrobing", "Policy Abuser", "Legitimate"]
THRESH_KEY = {
    "Fraudulent Return": "thresh_fraudulent",
    "Wardrobing": "thresh_wardrobing",
    "Policy Abuser": "thresh_policy_abuser",
    "Legitimate": "thresh_legitimate",
}
ACTION_MAP = {
    "Fraudulent Return": "FRAUD_HOLD",
    "Wardrobing": "VERIFY_TAGS",
    "Policy Abuser": "FEE_DEDUCTION",
    "Legitimate": "AUTO_APPROVE",
    "Uncertain": "LIGHT_VERIFY",
}
BANNER = {
    "AUTO_APPROVE": "Green", "VERIFY_TAGS": "Yellow", "FEE_DEDUCTION": "Orange",
    "FRAUD_HOLD": "Red", "LIGHT_VERIFY": "Yellow",
}
CUSTOMER_MESSAGES = {
    "AUTO_APPROVE":  "Your return has been approved. Refund will be processed in 3-5 business days.",
    "VERIFY_TAGS":   "Quick check needed: keep all tags and packaging intact, or upload a photo of the tag.",
    "FEE_DEDUCTION": "Return authorized. A PKR 950 handling and restocking fee will be deducted from your refund.",
    "FRAUD_HOLD":    "Your request is under review. Our team will contact you within 24 hours.",
    "LIGHT_VERIFY":  "Your return is being reviewed. We may reach out for additional information.",
}
MERCHANT_NOTES = {
    "AUTO_APPROVE":  "Low-risk return. Approve automatically.",
    "VERIFY_TAGS":   "Wardrobing risk. Verify product condition and tags before refund.",
    "FEE_DEDUCTION": "Policy-abuse pattern. Apply restocking fee per policy.",
    "FRAUD_HOLD":    "High fraud probability. Hold and route to manual review.",
    "LIGHT_VERIFY":  "Uncertain profile. Light verification recommended.",
}

DEFAULT_POLICY_CONFIG = {
    "thresh_legitimate": 0.60, "thresh_wardrobing": 0.50,
    "thresh_policy_abuser": 0.45, "thresh_fraudulent": 0.40, "fee_pkr": 950,
}


def _class_shap(sv, class_idx):
    """Return 1-D SHAP values for one sample across shap output formats."""
    if isinstance(sv, list):
        return sv[class_idx][0]
    sv = np.asarray(sv)
    if sv.ndim == 3:          # (samples, features, classes)
        return sv[0, :, class_idx]
    return sv[0]


class PolicyEngine:
    """Loads the trained model + config and scores return requests."""

    def __init__(self, artifacts_dir: Path | str = ARTIFACTS):
        self.dir = Path(artifacts_dir)
        model_path = self.dir / "model.joblib"
        if not model_path.exists():
            raise RuntimeError(
                f"Model not found at {model_path}. Train the model first "
                "(run train_model.py or the notebook)."
            )
        self.model = joblib.load(model_path)
        self.explainer = shap.TreeExplainer(self.model)

        cfg_path = self.dir / "policy_config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            self.policy_config = cfg.get("policy_config", dict(DEFAULT_POLICY_CONFIG))
            self.cat_risk_map = cfg.get("cat_risk_map", {})
            self.feature_cols = cfg.get("feature_cols", FEATURE_COLS)
        else:
            self.policy_config = dict(DEFAULT_POLICY_CONFIG)
            self.cat_risk_map = {}
            self.feature_cols = FEATURE_COLS

    def _build_row(self, features: dict) -> pd.DataFrame:
        row = pd.DataFrame([features])
        o2r = (pd.to_datetime(row["return_date"].iloc[0]) -
               pd.to_datetime(row["order_date"].iloc[0])).days
        row["order_to_return_days"] = max(0, int(o2r))
        row["return_ratio"] = (row["total_returns_lifetime"].iloc[0] /
                               max(row["total_orders_lifetime"].iloc[0], 1))
        row["risk_exposure"] = (row["refund_amount_requested_usd"].iloc[0] *
                                (1 + row["previous_dispute_count"].iloc[0]))
        row["category_risk_index"] = self.cat_risk_map.get(
            row["product_category"].iloc[0], 0.0)
        row["dispute_frequency"] = (row["previous_dispute_count"].iloc[0] /
                                    max(row["account_age_days"].iloc[0] / 30, 1))
        X = row[self.feature_cols].copy()
        for c in CAT_COLS:
            if c in X.columns:
                X[c] = X[c].astype("category")
        return X

    def predict_return(self, features: dict) -> dict:
        t0 = time.perf_counter()
        X = self._build_row(features)

        probs_arr = self.model.predict_proba(X)[0]
        probs = {INT_TO_LABEL[i]: float(p) for i, p in enumerate(probs_arr)}

        predicted_class = "Uncertain"
        for cls in SEVERITY:
            if probs[cls] >= self.policy_config[THRESH_KEY[cls]]:
                predicted_class = cls
                break

        action = ACTION_MAP[predicted_class]
        fee_pkr = self.policy_config["fee_pkr"] if action == "FEE_DEDUCTION" else 0

        cls_idx = LABEL_MAP.get(predicted_class, int(np.argmax(probs_arr)))
        shap_row = _class_shap(self.explainer.shap_values(X), cls_idx)
        top3 = np.argsort(np.abs(shap_row))[-3:][::-1]
        top_reasons = [(self.feature_cols[i], float(shap_row[i])) for i in top3]

        return {
            "probs": probs,
            "predicted_class": predicted_class,
            "action": action,
            "fee_pkr": fee_pkr,
            "customer_message": CUSTOMER_MESSAGES[action],
            "merchant_note": MERCHANT_NOTES[action],
            "banner": BANNER[action],
            "top_reasons": top_reasons,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }


_ENGINE = None

def get_policy_engine() -> PolicyEngine:
    """Lazy singleton accessor for the Streamlit app."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = PolicyEngine()
    return _ENGINE
