# Technical Requirements Document (TRD): ReturnGuard

## 1. Stack
Python 3.10+, pandas, numpy, scikit-learn, LightGBM, SHAP, matplotlib, joblib, Streamlit. IDE: AWS Kiro (spec mode). Optional: AWS Lambda/S3 as a described extension only.

## 2. Architecture
```
CSV / synthetic data -> data_pipeline.py -> features
features -> train_model.py -> model.joblib + metrics.json
model.joblib -> policy_engine.py (predict + SHAP + policy)
policy_engine.py -> app.py (Streamlit: Customer Portal | Merchant Dashboard)
```

## 3. Dataset
- Kaggle "E-Commerce Return Abuse Detection" (about 60,000 rows; 4-class target `abuse_type`).
- Class mix: Legitimate 70.1%, Policy Abuser 11.9%, Fraudulent 10.2%, Wardrobing 7.7%.
- **Verify real column names after download.** The code must read a column map (config dict) so names can be adjusted in one place.
- Place file at `data/returns.csv`. If absent, generate synthetic data (section 4).

## 4. Synthetic Fallback
Generate 5,000+ rows (not 1,000, so the model has enough signal) with the same class mix and plausible patterns:
- Wardrobing: short order-to-return days (0-2), high price, apparel/luxury, high return rate.
- Abuser: high historical return rate (>0.7), long delta, low-price repeated items.
- Fraud: delta ~0, prior disputes >= 3, address mismatch, electronics.
- Legitimate: low return rate, moderate delta.
Add noise/overlap so the classes are not trivially separable. Label output as synthetic in the UI.

## 5. Features
| Feature | Definition |
|---------|-----------|
| order_to_return_days | return_date - purchase_date (days, float) |
| return_ratio | total_returns / max(total_purchases, 1) |
| risk_exposure | item_price * (1 + prior_disputes_count) |
| category_risk_index | per-category risk, computed from TRAIN split only (avoid leakage) |
| dispute_frequency | prior_disputes_count / max(account_age_days/30, 1) |
| account_age_days, item_price, historical_return_rate | raw numeric |
| product_category, payment_method | encoded (one-hot or LightGBM categorical) |

Do not use customer_id as a feature. Do not use protected attributes (gender, religion, etc.) if present.
Tree models do not need scaling; skip scaling.

## 6. Model
- LightGBM `objective=multiclass`, `num_class=4`, `class_weight='balanced'`.
- Stratified 80/20 split (plus validation fold for threshold tuning); random_state fixed.
- Quick tuning: n_estimators 200-500, learning_rate 0.05, num_leaves 31-63, early stopping.
- Metrics: Macro F1, per-class precision/recall, confusion matrix, saved to `metrics.json`.
- Label map: 0 Legitimate, 1 Policy Abuser, 2 Fraudulent Return, 3 Wardrobing.

## 7. Policy Engine API
```python
predict_return(features: dict) -> {
  "probs": {"Legitimate": float, ...},
  "predicted_class": str,
  "action": "AUTO_APPROVE" | "VERIFY_TAGS" | "FEE_DEDUCTION" | "FRAUD_HOLD" | "LIGHT_VERIFY",
  "fee_pkr": int,
  "customer_message": str,
  "merchant_note": str,
  "top_reasons": [(feature, shap_value), ...],
  "latency_ms": float
}
```
Rules per PRD section 5, thresholds in a single `POLICY_CONFIG` dict.

## 8. Explainability
- `shap.TreeExplainer(model)` created once, cached.
- Multiclass handling: select SHAP values for the predicted class (handle both list and 3-D array output across shap versions).
- Render waterfall via `shap.plots.waterfall(shap.Explanation(...))`; convert top 3 to English sentences.

## 9. Impact Calculations (assumptions displayed in UI)
- Value protected = sum(item_price) of Fraud-hold + Wardrobing-flagged returns that would otherwise be refunded.
- Handling cost saved = prevented consignments x $30 (assumption from research; configurable).
- CO2 avoided = avoided trips x configurable kg/trip (default 1.0 kg, placeholder assumption, labeled as estimate).

## 10. Performance and Reliability
- Model loaded once with `st.cache_resource`; data with `st.cache_data`.
- Latency measured with `time.perf_counter` around predict + SHAP; show in UI.
- Offline-capable; no network calls at runtime.

## 11. Testing
- Unit: feature engineering, policy thresholds (boundary values), SHAP shape.
- Integration: each of 4 demo profiles returns expected action.
- Smoke: `streamlit run app.py` starts and both views render.

## 12. Repository Layout
```
returnguard/
  data/returns.csv (optional)
  data_pipeline.py
  train_model.py
  policy_engine.py
  app.py
  demo_profiles.py
  tests/test_policy.py
  artifacts/model.joblib, metrics.json
  requirements.txt
  .kiro/specs/ (requirements.md, design.md, tasks.md)
```
