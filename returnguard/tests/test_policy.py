"""Integration tests for the modular PolicyEngine.

Requires trained artifacts (model.joblib, policy_config.json). If they are
absent the tests are skipped so the suite still runs in a clean checkout.
"""
import time
from pathlib import Path

import pytest

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
pytestmark = pytest.mark.skipif(
    not (ARTIFACTS / "model.joblib").exists(),
    reason="trained model.joblib not present; run train_model.py first",
)


def _base(**over):
    b = dict(
        order_id="D", customer_id="D", age=35, account_age_days=1200,
        customer_segment="Gold", country="US", platform="Web Browser",
        device_type="MacBook", payment_method="Credit Card",
        product_category="Books", avg_order_value_usd=45.0,
        refund_amount_requested_usd=40.0, is_high_value_item=0, discount_used=0,
        order_date="2023-06-01", return_date="2023-06-10", days_to_return=9,
        return_reason="Defective/Broken", total_orders_lifetime=50,
        total_returns_lifetime=2, return_rate_pct=4.0, item_returned_opened=0,
        return_packaging_intact=1, photo_evidence_provided=1,
        tracking_number_valid=1, shipping_carrier="FedEx",
        address_change_before_delivery=0, refund_to_different_account=0,
        multiple_accounts_flag=0, customer_support_contacts=0,
        previous_dispute_count=0, wishlist_to_cart_time_hrs=120.0,
        review_left_after_return=1,
    )
    b.update(over)
    return b


HONEST = _base(customer_segment="Bronze", product_category="Clothing",
               avg_order_value_usd=156.0, refund_amount_requested_usd=140.0,
               order_date="2023-06-01", return_date="2023-06-16", days_to_return=15,
               return_rate_pct=5.3, total_orders_lifetime=40, total_returns_lifetime=2,
               wishlist_to_cart_time_hrs=36.0, photo_evidence_provided=0,
               review_left_after_return=0)
WARDROBER = _base(customer_segment="Silver", product_category="Clothing",
                  payment_method="Buy Now Pay Later", return_reason="Changed Mind",
                  is_high_value_item=1, avg_order_value_usd=279.0,
                  refund_amount_requested_usd=264.0, order_date="2023-06-01",
                  return_date="2023-07-11", days_to_return=40, item_returned_opened=1,
                  total_returns_lifetime=13, total_orders_lifetime=33,
                  return_rate_pct=40.4, previous_dispute_count=2,
                  customer_support_contacts=3, wishlist_to_cart_time_hrs=2.6,
                  photo_evidence_provided=0, review_left_after_return=0)
ABUSER = _base(customer_segment="Silver", product_category="Clothing",
               return_reason="Changed Mind", avg_order_value_usd=110.0,
               refund_amount_requested_usd=104.0, order_date="2023-06-01",
               return_date="2023-06-17", days_to_return=16, total_orders_lifetime=65,
               total_returns_lifetime=38, return_rate_pct=61.4, previous_dispute_count=2,
               customer_support_contacts=3, item_returned_opened=1,
               return_packaging_intact=0, wishlist_to_cart_time_hrs=36.3,
               photo_evidence_provided=0, review_left_after_return=0)
FRAUDSTER = _base(customer_segment="Silver", product_category="Clothing",
                  payment_method="Buy Now Pay Later", return_reason="Item Not Received",
                  is_high_value_item=1, avg_order_value_usd=432.0,
                  refund_amount_requested_usd=430.0, order_date="2023-06-01",
                  return_date="2023-06-04", days_to_return=3, previous_dispute_count=3,
                  photo_evidence_provided=1, tracking_number_valid=0,
                  return_rate_pct=47.4, total_orders_lifetime=26,
                  total_returns_lifetime=12, customer_support_contacts=3,
                  wishlist_to_cart_time_hrs=2.6, review_left_after_return=0)


@pytest.fixture(scope="module")
def engine():
    from policy_engine import PolicyEngine
    return PolicyEngine()


@pytest.mark.parametrize("features,expected", [
    (HONEST, "AUTO_APPROVE"),
    (WARDROBER, "VERIFY_TAGS"),
    (ABUSER, "FEE_DEDUCTION"),
    (FRAUDSTER, "FRAUD_HOLD"),
])
def test_demo_profiles(engine, features, expected):
    r = engine.predict_return(features)
    assert r["action"] == expected


def test_result_contract(engine):
    r = engine.predict_return(HONEST)
    keys = {"probs", "predicted_class", "action", "fee_pkr", "customer_message",
            "merchant_note", "banner", "top_reasons", "latency_ms"}
    assert keys <= set(r)
    assert len(r["top_reasons"]) == 3
    assert r["latency_ms"] >= 0
    assert abs(sum(r["probs"].values()) - 1.0) < 1e-6


def test_fee_only_for_fee_deduction(engine):
    assert engine.predict_return(ABUSER)["fee_pkr"] == 950
    assert engine.predict_return(HONEST)["fee_pkr"] == 0
