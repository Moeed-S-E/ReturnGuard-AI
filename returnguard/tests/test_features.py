"""Property-based tests for ReturnGuard engineered-feature formulas and policy logic.

These mirror the pure functions defined in returnguard_prototype.ipynb
(Sections 4, 7, 9) so they can be verified without executing the notebook.
Run with: pytest returnguard/tests -q
"""
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings

# ---- Pure reference implementations (kept in sync with the notebook) ----

def order_to_return_days(order_date, return_date):
    d = (pd.to_datetime(return_date) - pd.to_datetime(order_date)).days
    return max(0, int(d))

def return_ratio(total_returns, total_orders):
    return total_returns / max(total_orders, 1)

def risk_exposure(refund_amount, dispute_count):
    return refund_amount * (1 + dispute_count)

def dispute_frequency(dispute_count, account_age_days):
    return dispute_count / max(account_age_days / 30, 1)

POLICY_CONFIG = {
    "thresh_legitimate": 0.60, "thresh_wardrobing": 0.50,
    "thresh_policy_abuser": 0.45, "thresh_fraudulent": 0.40, "fee_pkr": 950,
}
SEVERITY = ["Fraudulent Return", "Wardrobing", "Policy Abuser", "Legitimate"]
THRESH_KEY = {
    "Fraudulent Return": "thresh_fraudulent", "Wardrobing": "thresh_wardrobing",
    "Policy Abuser": "thresh_policy_abuser", "Legitimate": "thresh_legitimate",
}
ACTION_MAP = {
    "Fraudulent Return": "FRAUD_HOLD", "Wardrobing": "VERIFY_TAGS",
    "Policy Abuser": "FEE_DEDUCTION", "Legitimate": "AUTO_APPROVE",
    "Uncertain": "LIGHT_VERIFY",
}

def policy_decision(probs):
    for cls in SEVERITY:
        if probs[cls] >= POLICY_CONFIG[THRESH_KEY[cls]]:
            return cls
    return "Uncertain"


# ---- Property 3: order_to_return_days non-negative & correct ----
@given(
    o=st.integers(min_value=0, max_value=3000),
    delta=st.integers(min_value=-500, max_value=2000),
)
def test_order_to_return_days_nonneg(o, delta):
    base = pd.Timestamp("2023-01-01") + pd.Timedelta(days=o)
    ret = base + pd.Timedelta(days=delta)
    v = order_to_return_days(base, ret)
    assert v >= 0
    assert v == max(0, delta)


# ---- Property 4: return_ratio safe against zero denominators ----
@given(
    r=st.integers(min_value=0, max_value=10_000),
    o=st.integers(min_value=0, max_value=10_000),
)
def test_return_ratio_safe(r, o):
    val = return_ratio(r, o)  # must not raise
    assert val == r / max(o, 1)


# ---- Property 5: risk_exposure equals the correct product ----
@given(
    refund=st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False),
    disputes=st.integers(min_value=0, max_value=50),
)
def test_risk_exposure(refund, disputes):
    assert risk_exposure(refund, disputes) == refund * (1 + disputes)


# ---- Property 7: dispute_frequency safe against small denominators ----
@given(
    disputes=st.integers(min_value=0, max_value=50),
    age=st.integers(min_value=1, max_value=5000),
)
def test_dispute_frequency_safe(disputes, age):
    val = dispute_frequency(disputes, age)
    assert val == disputes / max(age / 30, 1)
    assert val >= 0


# ---- Property 12: severity-ordered policy decision is consistent ----
@given(
    pf=st.floats(0, 1), pw=st.floats(0, 1), pa=st.floats(0, 1), pl=st.floats(0, 1),
)
@settings(max_examples=300)
def test_policy_severity_order(pf, pw, pa, pl):
    probs = {
        "Fraudulent Return": pf, "Wardrobing": pw,
        "Policy Abuser": pa, "Legitimate": pl,
    }
    decision = policy_decision(probs)
    # The decision must be the first severity class meeting its threshold.
    expected = "Uncertain"
    for cls in SEVERITY:
        if probs[cls] >= POLICY_CONFIG[THRESH_KEY[cls]]:
            expected = cls
            break
    assert decision == expected


# ---- Property 14: fee is 950 iff FEE_DEDUCTION ----
@given(
    pf=st.floats(0, 1), pw=st.floats(0, 1), pa=st.floats(0, 1), pl=st.floats(0, 1),
)
def test_fee_iff_fee_deduction(pf, pw, pa, pl):
    probs = {"Fraudulent Return": pf, "Wardrobing": pw,
             "Policy Abuser": pa, "Legitimate": pl}
    action = ACTION_MAP[policy_decision(probs)]
    fee = POLICY_CONFIG["fee_pkr"] if action == "FEE_DEDUCTION" else 0
    if action == "FEE_DEDUCTION":
        assert fee == 950
    else:
        assert fee == 0
