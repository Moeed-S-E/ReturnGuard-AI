"""Threshold boundary tests for the severity-ordered policy decision.

Covers DoD boundary pairs: just-below vs at-threshold for each class, using the
PRD starting thresholds (Fraud 0.50, Wardrobing 0.60, Abuser 0.55, Legit 0.75).
These verify the decision rule is correct exactly at the boundary (>=), not the
runtime-tuned values (which are validated separately in test_policy.py).
"""
import pytest

# PRD starting thresholds (the boundaries called out in the DoD)
PRD_CONFIG = {
    "thresh_fraudulent": 0.50,
    "thresh_wardrobing": 0.60,
    "thresh_policy_abuser": 0.55,
    "thresh_legitimate": 0.75,
}
SEVERITY = ["Fraudulent Return", "Wardrobing", "Policy Abuser", "Legitimate"]
THRESH_KEY = {
    "Fraudulent Return": "thresh_fraudulent", "Wardrobing": "thresh_wardrobing",
    "Policy Abuser": "thresh_policy_abuser", "Legitimate": "thresh_legitimate",
}
ACTION = {
    "Fraudulent Return": "FRAUD_HOLD", "Wardrobing": "VERIFY_TAGS",
    "Policy Abuser": "FEE_DEDUCTION", "Legitimate": "AUTO_APPROVE",
    "Uncertain": "LIGHT_VERIFY",
}


def decide(probs, cfg=PRD_CONFIG):
    for cls in SEVERITY:
        if probs[cls] >= cfg[THRESH_KEY[cls]]:
            return ACTION[cls]
    return "LIGHT_VERIFY"


def _probs(fraud=0.0, ward=0.0, abuser=0.0, legit=0.0):
    return {"Fraudulent Return": fraud, "Wardrobing": ward,
            "Policy Abuser": abuser, "Legitimate": legit}


# ---- Fraud boundary 0.49 / 0.50 ----
def test_fraud_below_boundary():
    assert decide(_probs(fraud=0.49)) == "LIGHT_VERIFY"

def test_fraud_at_boundary():
    assert decide(_probs(fraud=0.50)) == "FRAUD_HOLD"


# ---- Wardrobing boundary 0.59 / 0.60 ----
def test_wardrobing_below_boundary():
    assert decide(_probs(ward=0.59)) == "LIGHT_VERIFY"

def test_wardrobing_at_boundary():
    assert decide(_probs(ward=0.60)) == "VERIFY_TAGS"


# ---- Policy Abuser boundary 0.54 / 0.55 ----
def test_abuser_below_boundary():
    assert decide(_probs(abuser=0.54)) == "LIGHT_VERIFY"

def test_abuser_at_boundary():
    assert decide(_probs(abuser=0.55)) == "FEE_DEDUCTION"


# ---- Legitimate boundary 0.74 / 0.75 ----
def test_legit_below_boundary():
    assert decide(_probs(legit=0.74)) == "LIGHT_VERIFY"

def test_legit_at_boundary():
    assert decide(_probs(legit=0.75)) == "AUTO_APPROVE"


# ---- Severity precedence: fraud wins even if others also exceed ----
def test_severity_precedence_fraud_first():
    assert decide(_probs(fraud=0.60, ward=0.90, abuser=0.90, legit=0.90)) == "FRAUD_HOLD"

def test_severity_precedence_wardrobing_over_abuser():
    assert decide(_probs(ward=0.70, abuser=0.90, legit=0.90)) == "VERIFY_TAGS"
