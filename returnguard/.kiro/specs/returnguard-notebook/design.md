# Design Document: ReturnGuard Notebook

## Overview

ReturnGuard Notebook is a single Jupyter `.ipynb` file that implements an end-to-end
explainable-AI return-abuse detection pipeline. It covers data ingestion and validation,
exploratory data analysis, feature engineering, stratified train/validation/test splitting,
LightGBM multiclass training with native categoricals, per-class threshold auto-tuning,
SHAP-based explainability (four waterfall plots), a `predict_return()` policy engine, and
four demo-profile validation runs. The notebook is the canonical prototype for a later
conversion to modular `.py` modules and a Streamlit UI.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                  returnguard_prototype.ipynb                         │
│                                                                     │
│  Section 1: Setup & Imports                                         │
│     seed fixation (random, numpy, LightGBM)                        │
│     library version assertions                                      │
│                                                                     │
│  Section 2: Load & Validate Data                                    │
│     CSV → raw DataFrame                                             │
│     class membership guard → ValueError                             │
│     diagnostic display                                              │
│                                                                     │
│  Section 3: Exploratory Data Analysis                               │
│     class-count bar chart                                           │
│     Pearson correlation heatmap                                     │
│     missing-value summary                                           │
│     distribution plots (KDE / histogram)                           │
│     cross-tabulation (product_category, return_reason)              │
│                                                                     │
│  Section 4: Feature Engineering                                     │
│     order_to_return_days, return_ratio, risk_exposure              │
│     category_risk_index (train-only, mapped to all splits)         │
│     dispute_frequency                                               │
│     feature list printed                                            │
│                                                                     │
│  Section 5: Train / Validation / Test Split                         │
│     stratified 70 / 10 / 20                                         │
│     random_state = 42                                               │
│     class-distribution verification per split                      │
│                                                                     │
│  Section 6: Model Training                                          │
│     LGBMClassifier (multiclass, balanced, native categoricals)      │
│     early stopping on validation (patience 50)                     │
│     test-set evaluation: macro F1, per-class metrics, confusion     │
│     save model.joblib + metrics.json to artifacts/                  │
│                                                                     │
│  Section 7: Threshold Tuning                                        │
│     POLICY_CONFIG dict (initial values)                             │
│     sweep [0.30 → 0.90 step 0.05] per class on validation probs    │
│     update POLICY_CONFIG in-place                                   │
│     severity-ordered policy eval on test set                       │
│                                                                     │
│  Section 8: SHAP Explainability                                     │
│     TreeExplainer(model) → cached                                   │
│     shap_values on test set                                         │
│     4 waterfall plots (one per class, high-confidence sample)      │
│                                                                     │
│  Section 9: Policy Engine                                           │
│     POLICY_CONFIG (single source of truth)                          │
│     predict_return(features: dict) -> dict                          │
│                                                                     │
│  Section 10: Demo Validation                                        │
│     4 demo profiles → PASS / FAIL                                   │
└─────────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴────────────────┐
              ▼                                ▼
    returnguard/artifacts/         (future)
      model.joblib                 modular .py files
      metrics.json                 Streamlit UI
```

---

## Components

### 1. Setup & Imports Cell

All third-party imports are placed in a single cell. Seeds are fixed at `42` for Python
`random`, `numpy.random`, and LightGBM's `random_state`. Library versions for `pandas`,
`numpy`, `lightgbm`, `shap`, `sklearn`, and `joblib` are printed.

```python
import random, time, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, recall_score, precision_score
)

# Seed fixation
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Version diagnostics
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("lightgbm:", lgb.__version__)
print("shap:", shap.__version__)
print("sklearn:", sklearn.__version__ if hasattr(sklearn, '__version__') else sklearn.version)
print("joblib:", joblib.__version__)
```

### 2. Data Loader and Validator

Resolves the CSV path relative to the notebook's parent directory using `Path(__file__).parent`
(or `Path.cwd()` in a notebook context).

```python
def load_and_validate(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at expected path: {csv_path}")
    df = pd.read_csv(csv_path)
    EXPECTED_CLASSES = {"Legitimate", "Policy Abuser", "Fraudulent Return", "Wardrobing"}
    found = set(df["abuse_type"].unique())
    missing_classes = EXPECTED_CLASSES - found
    if missing_classes:
        raise ValueError(f"abuse_type is missing required class(es): {missing_classes}")
    return df
```

Key display outputs: shape, class counts + percentages, missing values per column, dtypes.

### 3. EDA Module

Pure display functions — no side-effects on the DataFrame:

| Function | Output |
|---|---|
| `plot_class_distribution(df)` | Bar chart with percentage labels |
| `plot_correlation_heatmap(df)` | Pearson heatmap, diverging colormap (`RdBu_r`) |
| `summarize_missing(df)` | Table with count and `%` only for columns with ≥ 1 null |
| `plot_numeric_distributions(df, cols)` | KDE/histogram per numeric column |
| `plot_categorical_crosstab(df, cats)` | Grouped bar chart per categorical |

### 4. Feature Engineering Pipeline

All five engineered features are computed by a single `engineer_features(df, cat_risk_map=None)`
function. `category_risk_index` requires `cat_risk_map` to be pre-computed from training rows
only; when `None` it is computed from the passed DataFrame (training use only).

```python
def engineer_features(df: pd.DataFrame, cat_risk_map: dict | None = None) -> pd.DataFrame:
    df = df.copy()
    df["order_to_return_days"] = (
        pd.to_datetime(df["return_date"]) - pd.to_datetime(df["order_date"])
    ).dt.days.clip(lower=0).astype(int)
    df["return_ratio"] = df["total_returns_lifetime"] / df["total_orders_lifetime"].clip(lower=1)
    df["risk_exposure"] = df["refund_amount_requested_usd"] * (1 + df["previous_dispute_count"])
    if cat_risk_map is None:
        cat_risk_map = (
            df.groupby("product_category")["abuse_label"]
            .apply(lambda s: (s == 2).mean())
            .to_dict()
        )
    df["category_risk_index"] = df["product_category"].map(cat_risk_map).fillna(0.0)
    df["dispute_frequency"] = (
        df["previous_dispute_count"]
        / (df["account_age_days"] / 30).clip(lower=1)
    )
    return df, cat_risk_map
```

`abuse_label` uses integer encoding: `Legitimate=0`, `Fraudulent Return=2`, etc. The exact
mapping is persisted in `metrics.json`.

**Excluded columns**: `customer_id`, `order_id`, `order_date`, `return_date`, `abuse_type`
(string target), `abuse_label` is the target.

### 5. Train / Validation / Test Split

```python
def split_data(df, target="abuse_label", val_size=0.1, test_size=0.2, seed=42):
    X = df[FEATURE_COLS]
    y = df[target]
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    val_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_adjusted, stratify=y_tmp, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
```

`category_risk_index` is computed from training rows and mapped to validation/test using the
returned `cat_risk_map`.

### 6. LightGBM Training

**Categorical columns** (cast to `"category"` dtype before fit):
`customer_segment`, `country`, `platform`, `device_type`, `payment_method`,
`product_category`, `return_reason`, `shipping_carrier`.

**Boolean/binary columns** (`int8`):
`is_high_value_item`, `discount_used`, `item_returned_opened`, `return_packaging_intact`,
`photo_evidence_provided`, `tracking_number_valid`, `address_change_before_delivery`,
`refund_to_different_account`, `multiple_accounts_flag`, `review_left_after_return`.

**Key hyperparameters**:

```python
params = dict(
    objective="multiclass",
    num_class=4,
    class_weight="balanced",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=63,
    random_state=42,
    n_jobs=-1,
)
callbacks = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
model = lgb.LGBMClassifier(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    categorical_feature=CAT_COLS,
    callbacks=callbacks,
)
```

**Artifact saving**:

```python
ARTIFACTS = Path("returnguard/artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)
joblib.dump(model, ARTIFACTS / "model.joblib")
with open(ARTIFACTS / "metrics.json", "w") as f:
    json.dump(metrics_dict, f, indent=2)
```

### 7. Threshold Tuning

```python
POLICY_CONFIG = {
    "thresh_legitimate": 0.60,
    "thresh_wardrobing": 0.50,
    "thresh_policy_abuser": 0.45,
    "thresh_fraudulent": 0.40,
    "fee_pkr": 950,
}

TUNING_MAP = {
    0: ("thresh_legitimate",   "precision"),  # Legitimate → maximize precision
    1: ("thresh_policy_abuser","recall"),      # Policy Abuser → maximize recall
    2: ("thresh_fraudulent",   "recall"),      # Fraudulent Return → maximize recall
    3: ("thresh_wardrobing",   "recall"),      # Wardrobing → maximize recall
}
THRESHOLDS = np.arange(0.30, 0.91, 0.05)
```

The sweep evaluates each threshold candidate against validation-set predictions:

```python
def tune_threshold(probs_val, y_val, class_idx, metric, thresholds):
    best_thresh, best_score = 0.50, -1.0
    for t in thresholds:
        preds = (probs_val[:, class_idx] >= t).astype(int)
        truth = (y_val == class_idx).astype(int)
        if metric == "recall":
            score = recall_score(truth, preds, zero_division=0)
        else:
            score = precision_score(truth, preds, zero_division=0)
        if score > best_score:
            best_score, best_thresh = score, t
    return best_thresh, best_score
```

**Severity-ordered policy decision** (applied to test set for final evaluation):

```
if prob[Fraudulent] >= thresh_fraudulent  → FRAUD_HOLD
elif prob[Wardrobing] >= thresh_wardrobing → VERIFY_TAGS
elif prob[PolicyAbuser] >= thresh_policy_abuser → FEE_DEDUCTION
elif prob[Legitimate] >= thresh_legitimate → AUTO_APPROVE
else → LIGHT_VERIFY (Uncertain)
```

### 8. SHAP Explainability

```python
# Cached after training — one instance for the entire notebook session
explainer = shap.TreeExplainer(model)
shap_values_test = explainer.shap_values(X_test)
# shap_values_test: list of 4 arrays (one per class), each shape (n_samples, n_features)
```

Waterfall plot per class:

```python
CLASS_NAMES = {0: "Legitimate", 1: "Policy Abuser", 2: "Fraudulent Return", 3: "Wardrobing"}

def get_representative_sample(probs, class_idx, threshold=0.70):
    high_conf = np.where(probs[:, class_idx] >= threshold)[0]
    if len(high_conf) == 0:
        return np.argmax(probs[:, class_idx])
    return high_conf[np.argmax(probs[high_conf, class_idx])]

for cls_idx, cls_name in CLASS_NAMES.items():
    sample_idx = get_representative_sample(probs_test, cls_idx)
    sv = shap_values_test[cls_idx][sample_idx]  # 1-D array
    exp = shap.Explanation(
        values=sv,
        base_values=explainer.expected_value[cls_idx],
        data=X_test.iloc[sample_idx],
        feature_names=X_test.columns.tolist(),
    )
    plt.figure()
    plt.title(f"SHAP Waterfall — {cls_name}")
    shap.plots.waterfall(exp, show=False)
    plt.tight_layout()
    plt.show()
```

### 9. Policy Engine

```python
# Class label ↔ integer mapping
LABEL_MAP = {
    "Legitimate": 0, "Policy Abuser": 1, "Fraudulent Return": 2, "Wardrobing": 3
}
INT_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

ACTION_MAP = {
    "Fraudulent Return": "FRAUD_HOLD",
    "Wardrobing":        "VERIFY_TAGS",
    "Policy Abuser":     "FEE_DEDUCTION",
    "Legitimate":        "AUTO_APPROVE",
    "Uncertain":         "LIGHT_VERIFY",
}

CUSTOMER_MESSAGES = {
    "AUTO_APPROVE":   "Your return has been approved. Refund will be processed in 3-5 business days.",
    "VERIFY_TAGS":    "Your return is under review. Please ensure all tags and packaging are intact.",
    "FEE_DEDUCTION":  "A restocking fee of PKR 950 will be deducted from your refund.",
    "FRAUD_HOLD":     "Your return request requires additional verification. Our team will contact you within 24 hours.",
    "LIGHT_VERIFY":   "Your return is being reviewed. We may reach out for additional information.",
}

MERCHANT_NOTES = {
    "AUTO_APPROVE":   "Low-risk return. Approve automatically.",
    "VERIFY_TAGS":    "Wardrobing risk detected. Verify product condition and tags before issuing refund.",
    "FEE_DEDUCTION":  "Policy abuse pattern. Apply restocking fee per policy.",
    "FRAUD_HOLD":     "High fraud probability. Place return on hold and initiate manual review.",
    "LIGHT_VERIFY":   "Uncertain risk profile. Light verification recommended.",
}

def predict_return(features: dict) -> dict:
    t0 = time.perf_counter()
    row = pd.DataFrame([features])

    # Cast categoricals
    for col in CAT_COLS:
        if col in row.columns:
            row[col] = row[col].astype("category")

    # Engineered features
    row["order_to_return_days"] = max(
        0,
        (pd.to_datetime(row["return_date"].iloc[0]) -
         pd.to_datetime(row["order_date"].iloc[0])).days
    )
    row["return_ratio"] = (
        row["total_returns_lifetime"].iloc[0] /
        max(row["total_orders_lifetime"].iloc[0], 1)
    )
    row["risk_exposure"] = (
        row["refund_amount_requested_usd"].iloc[0] *
        (1 + row["previous_dispute_count"].iloc[0])
    )
    row["category_risk_index"] = CAT_RISK_MAP.get(
        row["product_category"].iloc[0], 0.0
    )
    row["dispute_frequency"] = (
        row["previous_dispute_count"].iloc[0] /
        max(row["account_age_days"].iloc[0] / 30, 1)
    )

    X_row = row[FEATURE_COLS]
    probs_arr = model.predict_proba(X_row)[0]
    probs = {INT_TO_LABEL[i]: float(p) for i, p in enumerate(probs_arr)}

    # Severity-ordered threshold evaluation
    severity_order = ["Fraudulent Return", "Wardrobing", "Policy Abuser", "Legitimate"]
    thresh_keys = {
        "Fraudulent Return": "thresh_fraudulent",
        "Wardrobing":        "thresh_wardrobing",
        "Policy Abuser":     "thresh_policy_abuser",
        "Legitimate":        "thresh_legitimate",
    }
    predicted_class = "Uncertain"
    for cls in severity_order:
        if probs[cls] >= POLICY_CONFIG[thresh_keys[cls]]:
            predicted_class = cls
            break

    action = ACTION_MAP[predicted_class]
    fee_pkr = POLICY_CONFIG["fee_pkr"] if action == "FEE_DEDUCTION" else 0

    # SHAP top-3 reasons (cached explainer)
    sv = explainer.shap_values(X_row)
    cls_idx = LABEL_MAP.get(predicted_class, 0)
    shap_vals = sv[cls_idx][0] if isinstance(sv, list) else sv[0, :, cls_idx]
    top3_idx = np.argsort(np.abs(shap_vals))[-3:][::-1]
    top_reasons = [(FEATURE_COLS[i], float(shap_vals[i])) for i in top3_idx]

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "probs":            probs,
        "predicted_class":  predicted_class,
        "action":           action,
        "fee_pkr":          fee_pkr,
        "customer_message": CUSTOMER_MESSAGES[action],
        "merchant_note":    MERCHANT_NOTES[action],
        "top_reasons":      top_reasons,
        "latency_ms":       latency_ms,
    }
```

### 10. Demo Profile Validator

```python
DEMO_PROFILES = [
    {
        "name": "Honest Shopper",
        "expected_action": "AUTO_APPROVE",
        "features": {
            "age": 35, "account_age_days": 1200, "customer_segment": "Gold",
            "country": "US", "platform": "Web Browser", "device_type": "MacBook",
            "payment_method": "Credit Card", "product_category": "Books",
            "avg_order_value_usd": 45.0, "refund_amount_requested_usd": 40.0,
            "is_high_value_item": 0, "discount_used": 0,
            "order_date": "2023-06-01", "return_date": "2023-06-10",
            "return_reason": "Defective/Broken",
            "total_orders_lifetime": 50, "total_returns_lifetime": 2,
            "return_rate_pct": 4.0, "item_returned_opened": 0,
            "return_packaging_intact": 1, "photo_evidence_provided": 1,
            "tracking_number_valid": 1, "shipping_carrier": "FedEx",
            "address_change_before_delivery": 0, "refund_to_different_account": 0,
            "multiple_accounts_flag": 0, "customer_support_contacts": 0,
            "previous_dispute_count": 0, "wishlist_to_cart_time_hrs": 120.0,
            "review_left_after_return": 1,
        }
    },
    {
        "name": "Wardrober",
        "expected_action": "VERIFY_TAGS",
        "features": { ... }  # high-value clothing, short return window, opened item
    },
    {
        "name": "Policy Abuser",
        "expected_action": "FEE_DEDUCTION",
        "features": { ... }  # many lifetime returns, multiple disputes
    },
    {
        "name": "Fraudster",
        "expected_action": "FRAUD_HOLD",
        "features": { ... }  # refund to different account, multiple accounts flag, large refund
    },
]
```

Validation loop:

```python
for profile in DEMO_PROFILES:
    result = predict_return(profile["features"])
    passed = result["action"] == profile["expected_action"]
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n{status} — {profile['name']}")
    print(f"  predicted_class : {result['predicted_class']}")
    print(f"  action          : {result['action']}  (expected: {profile['expected_action']})")
    print(f"  customer_message: {result['customer_message']}")
    if not passed:
        print(f"  probs: {result['probs']}")
```

---

## Data Models

### Raw Dataset Schema

| Column | Type | Notes |
|---|---|---|
| `order_id` | str | Excluded from model |
| `customer_id` | str | Excluded from model |
| `age` | int | |
| `account_age_days` | int | |
| `customer_segment` | str (cat) | New / Bronze / Silver / Gold |
| `country` | str (cat) | |
| `platform` | str (cat) | |
| `device_type` | str (cat) | |
| `payment_method` | str (cat) | |
| `product_category` | str (cat) | |
| `avg_order_value_usd` | float | |
| `refund_amount_requested_usd` | float | |
| `is_high_value_item` | int (0/1) | |
| `discount_used` | int (0/1) | |
| `order_date` | str | ISO date |
| `return_date` | str | ISO date |
| `days_to_return` | int | Raw column, also re-derived |
| `return_reason` | str (cat) | |
| `total_orders_lifetime` | int | |
| `total_returns_lifetime` | int | |
| `return_rate_pct` | float | |
| `item_returned_opened` | int (0/1) | |
| `return_packaging_intact` | int (0/1) | |
| `photo_evidence_provided` | int (0/1) | |
| `tracking_number_valid` | int (0/1) | |
| `shipping_carrier` | str (cat) | |
| `address_change_before_delivery` | int (0/1) | |
| `refund_to_different_account` | int (0/1) | |
| `multiple_accounts_flag` | int (0/1) | |
| `customer_support_contacts` | int | |
| `previous_dispute_count` | int | |
| `wishlist_to_cart_time_hrs` | float | |
| `review_left_after_return` | int (0/1) | |
| `abuse_type` | str | 4-class target |
| `abuse_label` | int | 0–3 encoding of abuse_type |

### Engineered Features

| Feature | Formula | Purpose |
|---|---|---|
| `order_to_return_days` | `max(0, (return_date − order_date).days)` | Normalized return speed |
| `return_ratio` | `total_returns / max(total_orders, 1)` | Lifetime return rate |
| `risk_exposure` | `refund_amount * (1 + dispute_count)` | Monetary + dispute risk |
| `category_risk_index` | Per-category mean fraud rate (train only) | Category-level base rate |
| `dispute_frequency` | `dispute_count / max(account_age_months, 1)` | Dispute rate over time |

### POLICY_CONFIG Dict

```python
POLICY_CONFIG = {
    "thresh_legitimate":    float,  # tuned on validation set, default 0.60
    "thresh_wardrobing":    float,  # tuned on validation set, default 0.50
    "thresh_policy_abuser": float,  # tuned on validation set, default 0.45
    "thresh_fraudulent":    float,  # tuned on validation set, default 0.40
    "fee_pkr":              int,    # 950 — fixed, not tuned
}
```

### predict_return() Return Dict

```python
{
    "probs":            {"Legitimate": float, "Policy Abuser": float,
                         "Fraudulent Return": float, "Wardrobing": float},
    "predicted_class":  str,   # one of the 4 class names or "Uncertain"
    "action":           str,   # AUTO_APPROVE | VERIFY_TAGS | FEE_DEDUCTION |
                               # FRAUD_HOLD | LIGHT_VERIFY
    "fee_pkr":          int,   # 950 if FEE_DEDUCTION, else 0
    "customer_message": str,
    "merchant_note":    str,
    "top_reasons":      list,  # [(feature_name: str, shap_value: float), ×3]
    "latency_ms":       float, # >= 0
}
```

---

## Interfaces

### Notebook → Artifact Store

| Artifact | Path | Format | Contents |
|---|---|---|---|
| Trained model | `returnguard/artifacts/model.joblib` | joblib | `LGBMClassifier` instance |
| Metrics | `returnguard/artifacts/metrics.json` | JSON | macro F1, per-class F1, label map |

### Notebook → Streamlit UI (future)

The Streamlit UI will `import predict_return` from a module extracted from the Policy Engine
section. All constants (`POLICY_CONFIG`, `CAT_COLS`, `FEATURE_COLS`, `CAT_RISK_MAP`,
`INT_TO_LABEL`, loaded `model`, cached `explainer`) will be module-level singletons.

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| CSV file not found | `FileNotFoundError` with expected path in message |
| `abuse_type` missing a class | `ValueError` listing missing class names |
| Division by zero in `return_ratio` | Denominator clipped to minimum 1 |
| Division by zero in `dispute_frequency` | Denominator clipped to minimum 1 |
| Unknown `product_category` in policy call | `category_risk_index` defaults to `0.0` via `.fillna(0.0)` |
| No high-confidence SHAP sample (≥ 0.70) | Fall back to `argmax` over class probability column |
| SHAP output is 3-D array | Slice along last axis: `shap_values_test[:, :, cls_idx]` |
| `artifacts/` directory missing | `Path.mkdir(parents=True, exist_ok=True)` before save |

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions
of a system — essentially, a formal statement about what the system should do. Properties
serve as the bridge between human-readable specifications and machine-verifiable correctness
guarantees.*

### Property 1: Validation rejects incomplete class sets

*For any* DataFrame where the `abuse_type` column is missing at least one of the four required
class values (`Legitimate`, `Policy Abuser`, `Fraudulent Return`, `Wardrobing`), the validation
function SHALL raise a `ValueError`.

**Validates: Requirements 1.4**

---

### Property 2: Missing-value summary covers exactly the columns with nulls

*For any* DataFrame, `summarize_missing()` SHALL return a summary that contains exactly
those columns with at least one null value and SHALL NOT include columns with zero nulls.

**Validates: Requirements 2.3, 2.4**

---

### Property 3: `order_to_return_days` is non-negative and correct

*For any* `order_date` and `return_date` strings parseable as ISO dates, the computed
`order_to_return_days` SHALL equal `max(0, (return_date − order_date).days)`.

**Validates: Requirements 3.1**

---

### Property 4: `return_ratio` is safe against zero denominators

*For any* non-negative integers `total_returns_lifetime` and `total_orders_lifetime`
(including when `total_orders_lifetime = 0`), `return_ratio` SHALL equal
`total_returns_lifetime / max(total_orders_lifetime, 1)` and SHALL NOT raise a
`ZeroDivisionError`.

**Validates: Requirements 3.2**

---

### Property 5: `risk_exposure` equals the correct product

*For any* non-negative `refund_amount_requested_usd` and non-negative integer
`previous_dispute_count`, `risk_exposure` SHALL equal
`refund_amount_requested_usd * (1 + previous_dispute_count)`.

**Validates: Requirements 3.3**

---

### Property 6: `category_risk_index` is computed from training rows only

*For any* train/validation split of the dataset, the `category_risk_index` value assigned
to any validation-set row SHALL equal the per-category mean fraud rate computed exclusively
from the training rows for that category. No information from the validation or test rows
SHALL influence the index value.

**Validates: Requirements 3.4, 4.4**

---

### Property 7: `dispute_frequency` is safe against zero denominators

*For any* non-negative `previous_dispute_count` and positive `account_age_days`,
`dispute_frequency` SHALL equal `previous_dispute_count / max(account_age_days / 30, 1)`
and SHALL NOT raise a `ZeroDivisionError`.

**Validates: Requirements 3.5**

---

### Property 8: Stratified split preserves class proportions

*For any* dataset with at least 10 samples per class, the train/validation/test splits SHALL
have sizes approximately 70 %, 10 %, and 20 % of the total, and the per-class proportions
in each split SHALL be within 5 percentage points of the overall dataset proportions.

**Validates: Requirements 4.1, 4.2**

---

### Property 9: Split determinism with fixed seed

*For any* valid dataset, executing the split function twice with `random_state=42` SHALL
produce identical index sets for all three subsets.

**Validates: Requirements 4.2, 10.2**

---

### Property 10: Model serialization round-trip preserves predictions

*For any* feature vector from the test set, the probability predictions from the model
reloaded from `model.joblib` SHALL be element-wise equal to predictions from the original
in-memory model (within floating-point tolerance of 1e-6).

**Validates: Requirements 5.6**

---

### Property 11: Threshold tuning selects the optimal value in the sweep range

*For any* validation probability matrix and class index, the threshold returned by
`tune_threshold()` SHALL be the value in `[0.30, 0.35, …, 0.90]` that maximizes the
target metric (recall or precision) for that class on the validation set.

**Validates: Requirements 6.2, 6.3**

---

### Property 12: Severity-ordered policy decision is consistent

*For any* probability dict and `POLICY_CONFIG`, the action returned by the severity-order
evaluation SHALL be the action corresponding to the first class — in order Fraudulent
Return → Wardrobing → Policy Abuser → Legitimate — whose probability meets or exceeds
its configured threshold; if none do, the action SHALL be `LIGHT_VERIFY`.

**Validates: Requirements 6.6, 8.3**

---

### Property 13: `predict_return()` always returns a complete result dict

*For any* valid feature dict (all required columns present), `predict_return()` SHALL return
a dict containing exactly the keys: `probs`, `predicted_class`, `action`, `fee_pkr`,
`customer_message`, `merchant_note`, `top_reasons`, `latency_ms`, with correct types
(`probs` is a dict of four floats summing to ≈ 1.0, `fee_pkr` is an int, `latency_ms` is
a non-negative float, `top_reasons` is a list of exactly 3 tuples).

**Validates: Requirements 8.1, 8.2**

---

### Property 14: `fee_pkr` is 950 if and only if action is FEE_DEDUCTION

*For any* feature dict, if `predict_return()` returns `action = "FEE_DEDUCTION"` then
`fee_pkr` SHALL equal `POLICY_CONFIG["fee_pkr"]` (950); for any other action, `fee_pkr`
SHALL equal 0.

**Validates: Requirements 8.4**

---

### Property 15: Latency is non-negative

*For any* valid feature dict, `predict_return()` SHALL return `latency_ms >= 0`.

**Validates: Requirements 8.5**

---

### Property 16: SHAP representative sample selection uses highest-confidence sample

*For any* probability matrix and class index, `get_representative_sample()` SHALL return
the index of the test-set row with the highest predicted probability for that class, either
among rows with probability ≥ 0.70 (preferred) or among all rows (fallback).

**Validates: Requirements 7.5, 7.6**
