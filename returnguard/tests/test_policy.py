import pytest
from policy_engine import POLICY_CONFIG

def test_policy_thresholds():
    assert 'Fraudulent Return' in POLICY_CONFIG['thresholds']
    assert 'Wardrobing' in POLICY_CONFIG['thresholds']
    assert 'Policy Abuser' in POLICY_CONFIG['thresholds']
    assert 'Legitimate' in POLICY_CONFIG['thresholds']
    assert POLICY_CONFIG['fee_pkr'] > 0
