# Implementation Plan: ReturnGuard Notebook

## Overview

Build a single Jupyter notebook (`returnguard/returnguard_prototype.ipynb`) that implements the full ReturnGuard explainable-AI return-abuse detection pipeline in Python. The notebook runs top-to-bottom without manual intervention, covers all 10 sections defined in the design, trains a LightGBM multiclass classifier, tunes per-class thresholds, generates four SHAP waterfall plots, and exposes a `predict_return()` policy engine validated against four demo profiles. Artifacts (`model.joblib`, `metrics.json`) are saved to `returnguard/artifacts/`.

---

## Tasks

- [ ] 1. Set up notebook skeleton and Section 1 (Setup & Imports)
  - [x] 1.1 Create `returnguard/returnguard_prototype.ipynb` with all 10 Markdown heading cells in order
    - Create the `.ipynb` file with the correct JSON structure
    - Add one Markdown cell per section: (1) Setup & Imports, (2) Load & Validate Data, (3) Exploratory Data Analysis, (4) Feature Engineering, (5) Train/Validation/Test Split, (6) Model Training, (7) Threshold Tuning, (8) SHAP Explainability, (9) Policy Engine, (10) Demo Validation
    - _Requirements: 10.1_
  - [x] 1.2 Write the Setup & Imports code cell
    - Import all libraries in one cell: `random`, `time`, `json`, `os`, `pathlib.Path`, `numpy`, `pandas`, `matplotlib.pyplot`, `seaborn`, `lightgbm`, `shap`, `joblib`, `sklearn.model_selection`, `sklearn.metrics`
    - Fix seeds: `SEED=42`, `random.seed(SEED)`, `np.random.seed(SEED)`
    - Print library versions for `pandas`, `numpy`, `lightgbm`, `shap`, `sklearn`, `joblib`
    - _Requirements: 10.2, 10.5, 10.6_

- [ ] 2. Section 2 — Load & Validate Data
  - [x] 2.1 Implement `load_and_validate()` and the dataset loading cell
    - Resolve CSV path relative to notebook parent using `Path.cwd().parent / "datasets/ecommerce_return_abuse_dataset.csv"` (with a `..` fallback pattern)
    - Raise `FileNotFoundError` with descriptive path message if file not found
    - Validate that `abuse_type` contains exactly `{"Legitimate", "Policy Abuser", "Fraudulent Return", "Wardrobing"}`; raise `ValueError` listing missing values if not
    - Display: shape, `abuse_type` class distribution (counts + %), missing values per column, dtypes
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  - [ ]* 2.2 Write property test for validation rejects incomplete class sets
    - **Property 1: Validation rejects incomplete class sets**
    - **Validates: Requirements 1.4**
    - Test that `load_and_validate()` raises `ValueError` for DataFrames missing one or more of the four required class values

- [ ] 3. Section 3 — Exploratory Data Analysis
  - [x] 3.1 Implement EDA plotting functions and cells
    - `plot_class_distribution(df)`: bar chart of `abuse_type` counts with percentage labels on each bar
    - `plot_correlation_heatmap(df)`: Pearson heatmap of all numeric columns, diverging colormap `RdBu_r`
    - `summarize_missing(df)`: display table of count + `%` only for columns with ≥ 1 null; skip columns with zero nulls
    - `plot_numeric_distributions(df, cols)`: KDE/histogram for `days_to_return`, `return_rate_pct`, `avg_order_value_usd`, `refund_amount_requested_usd`, `total_returns_lifetime`
    - `plot_categorical_crosstab(df, cats)`: grouped bar chart of `abuse_type` vs `product_category` and `return_reason`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_
  - [ ]* 3.2 Write property test for missing-value summary correctness
    - **Property 2: Missing-value summary covers exactly the columns with nulls**
    - **Validates: Requirements 2.3, 2.4**
    - Test `summarize_missing()` with synthetic DataFrames: verify it includes only columns with ≥ 1 null and excludes fully-populated columns

- [ ] 4. Section 4 — Feature Engineering
  - [x] 4.1 Implement `engineer_features()` and the label encoding cell
    - Create `abuse_label` integer encoding: `Legitimate=0`, `Policy Abuser=1`, `Fraudulent Return=2`, `Wardrobing=3`; persist `LABEL_MAP` and `INT_TO_LABEL`
    - Implement `engineer_features(df, cat_risk_map=None)` returning `(df_out, cat_risk_map)`:
      - `order_to_return_days`: `(return_date - order_date).dt.days.clip(lower=0).astype(int)`
      - `return_ratio`: `total_returns_lifetime / total_orders_lifetime.clip(lower=1)`
      - `risk_exposure`: `refund_amount_requested_usd * (1 + previous_dispute_count)`
      - `category_risk_index`: computed from training rows only when `cat_risk_map=None`; otherwise uses passed map; fill unknown categories with `0.0`
      - `dispute_frequency`: `previous_dispute_count / (account_age_days / 30).clip(lower=1)`
    - Cast categorical columns to `"category"` dtype; define `CAT_COLS` list
    - Define `FEATURE_COLS` list (excluding `customer_id`, `order_id`, `order_date`, `return_date`, `abuse_type`, `abuse_label`); print the list
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_
  - [x]* 4.2 Write property tests for engineered feature formulas
    - **Property 3: `order_to_return_days` is non-negative and correct**
    - **Validates: Requirements 3.1**
    - **Property 4: `return_ratio` is safe against zero denominators**
    - **Validates: Requirements 3.2**
    - **Property 5: `risk_exposure` equals the correct product**
    - **Validates: Requirements 3.3**
    - **Property 7: `dispute_frequency` is safe against zero denominators**
    - **Validates: Requirements 3.5**
    - Use Hypothesis to generate arbitrary non-negative inputs; verify formulas hold and no `ZeroDivisionError` is raised

- [ ] 5. Section 5 — Train/Validation/Test Split
  - [x] 5.1 Implement `split_data()` and the split cell
    - Split into 70 % train / 10 % validation / 20 % test using stratified sampling on `abuse_label`, `random_state=42`
    - Compute `cat_risk_map` from training rows only via `engineer_features(X_train_raw, cat_risk_map=None)`, then apply map to validation and test rows
    - Print class distribution (counts + %) for each of the three splits
    - _Requirements: 4.1, 4.2, 4.3, 4.4_
  - [ ]* 5.2 Write property tests for split correctness
    - **Property 6: `category_risk_index` is computed from training rows only**
    - **Validates: Requirements 3.4, 4.4**
    - **Property 8: Stratified split preserves class proportions**
    - **Validates: Requirements 4.1, 4.2**
    - **Property 9: Split determinism with fixed seed**
    - **Validates: Requirements 4.2, 10.2**

- [ ] 6. Checkpoint — Verify data pipeline end-to-end
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Section 6 — LightGBM Model Training
  - [x] 7.1 Implement model training cell with native categoricals and early stopping
    - Cast all `CAT_COLS` to `"category"` dtype on all three splits before fitting
    - Instantiate `LGBMClassifier(objective="multiclass", num_class=4, class_weight="balanced", n_estimators=1000, learning_rate=0.05, num_leaves=63, random_state=42, n_jobs=-1)`
    - Fit with `eval_set=[(X_val, y_val)]`, `categorical_feature=CAT_COLS`, callbacks: `lgb.early_stopping(stopping_rounds=50)` and `lgb.log_evaluation(period=100)`
    - _Requirements: 5.1, 5.2, 5.3, 5.4_
  - [x] 7.2 Implement test-set evaluation and artifact saving
    - Compute and display: macro F1, per-class precision/recall (classification report), confusion matrix on test set
    - Create `returnguard/artifacts/` with `Path.mkdir(parents=True, exist_ok=True)`
    - Save `model.joblib` via `joblib.dump(model, ARTIFACTS / "model.joblib")`
    - Build `metrics_dict` with macro F1, per-class F1, and `LABEL_MAP`; save to `returnguard/artifacts/metrics.json`
    - _Requirements: 5.5, 5.6, 5.7, 5.8_
  - [ ]* 7.3 Write property test for model serialization round-trip
    - **Property 10: Model serialization round-trip preserves predictions**
    - **Validates: Requirements 5.6**
    - Load `model.joblib` into a new variable; assert `predict_proba` outputs match original within 1e-6 tolerance on a sample of test rows

- [ ] 8. Section 7 — Probability Threshold Auto-Tuning
  - [x] 8.1 Implement `tune_threshold()` and the POLICY_CONFIG sweep cell
    - Define initial `POLICY_CONFIG` dict with keys: `thresh_legitimate=0.60`, `thresh_wardrobing=0.50`, `thresh_policy_abuser=0.45`, `thresh_fraudulent=0.40`, `fee_pkr=950`
    - Implement `tune_threshold(probs_val, y_val, class_idx, metric, thresholds)` sweeping `np.arange(0.30, 0.91, 0.05)`; maximize recall for classes 1, 2, 3 and precision for class 0
    - Update `POLICY_CONFIG` in-place with tuned values; print each tuned threshold and achieved score
    - Evaluate severity-ordered policy on test set; report per-class accuracy
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_
  - [ ]* 8.2 Write property test for threshold tuning optimality
    - **Property 11: Threshold tuning selects the optimal value in the sweep range**
    - **Validates: Requirements 6.2, 6.3**
    - Generate synthetic probability arrays; verify `tune_threshold()` returns the threshold achieving the best score among all candidates in the sweep range

- [ ] 9. Section 8 — SHAP Explainability
  - [x] 9.1 Implement cached TreeExplainer and four waterfall plots
    - Create `explainer = shap.TreeExplainer(model)` after training (cached — one instance)
    - Compute `shap_values_test = explainer.shap_values(X_test)` on the test set
    - Handle both list output (index by class) and 3-D array output (slice last axis) per design error-handling spec
    - Implement `get_representative_sample(probs, class_idx, threshold=0.70)`: prefer samples with prob ≥ 0.70, fall back to argmax
    - Render four `shap.plots.waterfall` plots (one per class); add descriptive title above each; use `show=False` to preserve inline rendering
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8_
  - [ ]* 9.2 Write property test for representative sample selection
    - **Property 16: SHAP representative sample selection uses highest-confidence sample**
    - **Validates: Requirements 7.5, 7.6**
    - Test `get_representative_sample()` with synthetic probability matrices: verify it returns the highest-confidence index among samples ≥ 0.70, and falls back to global argmax when none qualify

- [ ] 10. Section 9 — Policy Engine
  - [x] 10.1 Implement `predict_return(features: dict) -> dict`
    - Define all module-level constants: `LABEL_MAP`, `INT_TO_LABEL`, `ACTION_MAP`, `CUSTOMER_MESSAGES`, `MERCHANT_NOTES`; reference only `POLICY_CONFIG` for thresholds — no hardcoded values inside the function
    - Function body: build `pd.DataFrame([features])`, cast `CAT_COLS`, compute all five engineered features using `CAT_RISK_MAP`, select `FEATURE_COLS`, call `model.predict_proba()`, build `probs` dict
    - Severity-ordered threshold evaluation: Fraudulent Return → Wardrobing → Policy Abuser → Legitimate → Uncertain
    - Compute `top_reasons` using cached `explainer` (handle list vs 3-D array); return top-3 `(feature_name, shap_value)` tuples by absolute SHAP value
    - Set `fee_pkr=POLICY_CONFIG["fee_pkr"]` if action is `FEE_DEDUCTION`, else `0`
    - Measure latency with `time.perf_counter`; return complete result dict
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_
  - [ ]* 10.2 Write property tests for predict_return() contract
    - **Property 12: Severity-ordered policy decision is consistent**
    - **Validates: Requirements 6.6, 8.3**
    - **Property 13: `predict_return()` always returns a complete result dict**
    - **Validates: Requirements 8.1, 8.2**
    - **Property 14: `fee_pkr` is 950 if and only if action is FEE_DEDUCTION**
    - **Validates: Requirements 8.4**
    - **Property 15: Latency is non-negative**
    - **Validates: Requirements 8.5**

- [ ] 11. Checkpoint — Verify policy engine with unit calls
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 12. Section 10 — Demo Profile Validation
  - [x] 12.1 Implement four demo profile dicts and validation loop
    - Define `DEMO_PROFILES` list with four complete feature dicts:
      - **Honest Shopper**: `expected_action="AUTO_APPROVE"` — low return history, valid tracking, no disputes, legitimate return reason, packaging intact
      - **Wardrober**: `expected_action="VERIFY_TAGS"` — high-value clothing, short return window, item opened, many lifetime returns
      - **Policy Abuser**: `expected_action="FEE_DEDUCTION"` — high return rate, multiple disputes, discount used, frequent returns
      - **Fraudster**: `expected_action="FRAUD_HOLD"` — `refund_to_different_account=1`, `multiple_accounts_flag=1`, large refund, `address_change_before_delivery=1`
    - Validation loop: call `predict_return()` for each; print `predicted_class`, `action`, `probs`, `customer_message`; print `✅ PASS` or `❌ FAIL`; on FAIL also print full `probs` dict
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [ ] 13. Final Checkpoint — Full notebook run validation
  - Ensure all tests pass, ask the user if questions arise.

---

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- The notebook must be fully executable top-to-bottom with `Kernel → Restart & Run All`; no cell should require internet access
- All imports remain in Section 1; no new imports in later cells (_Requirements: 10.4, 10.5_)
- `category_risk_index` must always be computed from training rows only and stored in `CAT_RISK_MAP` for use in `predict_return()` — this is the main data-leakage guard
- Demo profile feature values may be adjusted to achieve expected actions; model weights are never modified (_Requirements: 9.5_)
- Property tests (tasks marked `*`) use Hypothesis for generative testing where applicable

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["2.1", "3.1", "4.1"] },
    { "id": 2, "tasks": ["2.2", "3.2", "4.2", "5.1"] },
    { "id": 3, "tasks": ["5.2", "7.1"] },
    { "id": 4, "tasks": ["7.2", "8.1"] },
    { "id": 5, "tasks": ["7.3", "8.2", "9.1"] },
    { "id": 6, "tasks": ["9.2", "10.1"] },
    { "id": 7, "tasks": ["10.2", "12.1"] }
  ]
}
```
