import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st, settings
from sklearn.model_selection import train_test_split

# Property 1: Validation rejects incomplete class sets
def load_and_validate(df: pd.DataFrame):
    EXPECTED_CLASSES = {"Legitimate", "Policy Abuser", "Fraudulent Return", "Wardrobing"}
    found = set(df["abuse_type"].unique())
    missing_classes = EXPECTED_CLASSES - found
    if missing_classes:
        raise ValueError(f"abuse_type is missing required class(es): {missing_classes}")
    return df

@given(classes_present=st.sets(st.sampled_from(["Legitimate", "Policy Abuser", "Fraudulent Return", "Wardrobing"]), min_size=1, max_size=3))
def test_validation_rejects_incomplete_class_sets(classes_present):
    df = pd.DataFrame({"abuse_type": list(classes_present)})
    with pytest.raises(ValueError):
        load_and_validate(df)

# Property 2: Missing-value summary covers exactly the columns with nulls
def summarize_missing(df):
    missing = df.isnull().sum()
    return missing[missing > 0]

@given(
    null_a=st.integers(min_value=0, max_value=10),
    null_b=st.integers(min_value=0, max_value=10),
    null_c=st.integers(min_value=0, max_value=10)
)
def test_missing_value_summary(null_a, null_b, null_c):
    df = pd.DataFrame({
        "A": [1]*(10 - null_a) + [np.nan]*null_a,
        "B": [1]*(10 - null_b) + [np.nan]*null_b,
        "C": [1]*(10 - null_c) + [np.nan]*null_c,
    })
    summary = summarize_missing(df)
    expected_cols = []
    if null_a > 0: expected_cols.append("A")
    if null_b > 0: expected_cols.append("B")
    if null_c > 0: expected_cols.append("C")
    assert set(summary.index) == set(expected_cols)

# Property 6, 8, 9: Splits
def split_data(df, target="abuse_label", val_size=0.1, test_size=0.2, seed=42):
    X = df
    y = df[target]
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    val_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_adjusted, stratify=y_tmp, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

@settings(deadline=None)
@given(seed=st.integers(min_value=0, max_value=100))
def test_split_determinism(seed):
    np.random.seed(0)
    df = pd.DataFrame({
        "abuse_label": np.random.choice([0,1,2,3], 200),
        "product_category": np.random.choice(["A", "B", "C"], 200)
    })
    X_tr1, X_v1, X_te1, y_tr1, y_v1, y_te1 = split_data(df, seed=42)
    X_tr2, X_v2, X_te2, y_tr2, y_v2, y_te2 = split_data(df, seed=42)
    assert X_tr1.index.equals(X_tr2.index)
    assert X_v1.index.equals(X_v2.index)
    assert X_te1.index.equals(X_te2.index)

def test_stratified_split_preserves_proportions():
    np.random.seed(0)
    df = pd.DataFrame({
        "abuse_label": np.random.choice([0,1,2,3], 1000, p=[0.5, 0.2, 0.2, 0.1]),
    })
    total_props = df["abuse_label"].value_counts(normalize=True)
    X_tr, X_v, X_te, y_tr, y_v, y_te = split_data(df)
    
    tr_props = y_tr.value_counts(normalize=True)
    v_props = y_v.value_counts(normalize=True)
    te_props = y_te.value_counts(normalize=True)
    
    for c in [0,1,2,3]:
        assert abs(total_props[c] - tr_props.get(c, 0)) < 0.05
        assert abs(total_props[c] - v_props.get(c, 0)) < 0.05
        assert abs(total_props[c] - te_props.get(c, 0)) < 0.05

def test_category_risk_index_training_only():
    df = pd.DataFrame({
        "abuse_label": [2, 0, 2, 0, 2, 1, 0, 2],
        "product_category": ["A", "A", "B", "B", "C", "C", "C", "A"]
    })
    train_df = df.iloc[:4]
    val_df = df.iloc[4:]
    
    cat_risk_map = (
        train_df.groupby("product_category")["abuse_label"]
        .apply(lambda s: (s == 2).mean())
        .to_dict()
    )
    assert cat_risk_map["A"] == 0.5
    assert cat_risk_map["B"] == 0.5
    
    val_mapped = val_df["product_category"].map(cat_risk_map).fillna(0.0)
    assert val_mapped.iloc[0] == 0.0
    assert val_mapped.iloc[-1] == 0.5

# Property 10: Model serialization
def test_model_serialization_roundtrip():
    import lightgbm as lgb
    import joblib
    import os
    
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 4, 100)
    
    model = lgb.LGBMClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, "test_model.joblib")
    model_loaded = joblib.load("test_model.joblib")
    
    preds_orig = model.predict_proba(X)
    preds_loaded = model_loaded.predict_proba(X)
    
    np.testing.assert_allclose(preds_orig, preds_loaded, atol=1e-6)
    os.remove("test_model.joblib")

# Property 11: Threshold tuning optimality
def tune_threshold(probs_val, y_val, class_idx, metric, thresholds):
    from sklearn.metrics import recall_score, precision_score
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

@given(seed=st.integers(min_value=0, max_value=100))
def test_threshold_tuning_optimality(seed):
    from sklearn.metrics import recall_score
    np.random.seed(seed)
    probs_val = np.random.rand(50, 4)
    y_val = np.random.randint(0, 4, 50)
    
    thresholds = np.arange(0.30, 0.91, 0.05)
    best_t, best_s = tune_threshold(probs_val, y_val, 1, "recall", thresholds)
    
    for t in thresholds:
        preds = (probs_val[:, 1] >= t).astype(int)
        truth = (y_val == 1).astype(int)
        s = recall_score(truth, preds, zero_division=0)
        assert s <= best_s

# Property 16: SHAP sample selection
def get_representative_sample(probs, class_idx, threshold=0.70):
    high_conf = np.where(probs[:, class_idx] >= threshold)[0]
    if len(high_conf) == 0:
        return np.argmax(probs[:, class_idx])
    return high_conf[np.argmax(probs[high_conf, class_idx])]

@given(probs=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=10, max_size=50))
def test_get_representative_sample(probs):
    probs_arr = np.array(probs).reshape(-1, 1)
    idx = get_representative_sample(probs_arr, 0)
    
    high_conf = np.where(probs_arr[:, 0] >= 0.70)[0]
    if len(high_conf) > 0:
        assert idx in high_conf
        assert probs_arr[idx, 0] == np.max(probs_arr[high_conf, 0])
    else:
        assert probs_arr[idx, 0] == np.max(probs_arr[:, 0])

# Property 13 and 15: predict_return contract
import time
def predict_return_mock(features):
    t0 = time.perf_counter()
    action = "AUTO_APPROVE"
    fee_pkr = 950 if action == "FEE_DEDUCTION" else 0
    latency_ms = (time.perf_counter() - t0) * 1000.0
    
    return {
        "probs": {"Fraudulent Return": 0.1, "Wardrobing": 0.1, "Policy Abuser": 0.1, "Legitimate": 0.8},
        "predicted_class": action,
        "action": action,
        "fee_pkr": fee_pkr,
        "customer_message": "test",
        "merchant_note": "test",
        "top_reasons": [("a", 0.1), ("b", 0.1), ("c", 0.1)],
        "latency_ms": latency_ms,
    }

def test_predict_return_contract():
    res = predict_return_mock({})
    expected_keys = {"probs", "predicted_class", "action", "fee_pkr", "customer_message", "merchant_note", "top_reasons", "latency_ms"}
    assert set(res.keys()) == expected_keys
    assert isinstance(res["fee_pkr"], int)
    assert res["latency_ms"] >= 0
    assert len(res["top_reasons"]) == 3

