"""ReturnGuard Streamlit app — Customer Portal + Merchant Dashboard.

    streamlit run app.py

Loads the trained model via policy_engine.PolicyEngine (cached once).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from policy_engine import get_policy_engine

st.set_page_config(page_title="ReturnGuard", page_icon="🛡️", layout="wide")

BANNER_STYLE = {
    "Green":  ("#1b7f3b", "✅"),
    "Yellow": ("#9a7d0a", "🔎"),
    "Orange": ("#b35900", "⚠️"),
    "Red":    ("#9b1c1c", "⛔"),
}


@st.cache_resource(show_spinner="Loading model…")
def _engine():
    return get_policy_engine()


def _base_profile(**over):
    base = dict(
        order_id="DEMO", customer_id="DEMO", age=35, account_age_days=1200,
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
    base.update(over)
    return base


DEMO_PROFILES = {
    "Fatima — Honest Shopper": _base_profile(
        customer_segment="Bronze", product_category="Clothing",
        avg_order_value_usd=156.0, refund_amount_requested_usd=140.0,
        order_date="2023-06-01", return_date="2023-06-16", days_to_return=15,
        return_rate_pct=5.3, total_orders_lifetime=40, total_returns_lifetime=2,
        wishlist_to_cart_time_hrs=36.0, photo_evidence_provided=0,
        review_left_after_return=0),
    "Wardrober — Luxury Return": _base_profile(
        customer_segment="Silver", product_category="Clothing",
        payment_method="Buy Now Pay Later", return_reason="Changed Mind",
        is_high_value_item=1, avg_order_value_usd=279.0,
        refund_amount_requested_usd=264.0, order_date="2023-06-01",
        return_date="2023-07-11", days_to_return=40, item_returned_opened=1,
        total_returns_lifetime=13, total_orders_lifetime=33, return_rate_pct=40.4,
        previous_dispute_count=2, customer_support_contacts=3,
        wishlist_to_cart_time_hrs=2.6, photo_evidence_provided=0,
        review_left_after_return=0),
    "Policy Abuser — Serial Returner": _base_profile(
        customer_segment="Silver", product_category="Clothing",
        return_reason="Changed Mind", avg_order_value_usd=110.0,
        refund_amount_requested_usd=104.0, order_date="2023-06-01",
        return_date="2023-06-17", days_to_return=16, total_orders_lifetime=65,
        total_returns_lifetime=38, return_rate_pct=61.4, previous_dispute_count=2,
        customer_support_contacts=3, item_returned_opened=1,
        return_packaging_intact=0, wishlist_to_cart_time_hrs=36.3,
        photo_evidence_provided=0, review_left_after_return=0),
    "Fraudster — Item Not Received": _base_profile(
        customer_segment="Silver", product_category="Clothing",
        payment_method="Buy Now Pay Later", return_reason="Item Not Received",
        is_high_value_item=1, avg_order_value_usd=432.0,
        refund_amount_requested_usd=430.0, order_date="2023-06-01",
        return_date="2023-06-04", days_to_return=3, previous_dispute_count=3,
        photo_evidence_provided=1, tracking_number_valid=0, return_rate_pct=47.4,
        total_orders_lifetime=26, total_returns_lifetime=12,
        customer_support_contacts=3, wishlist_to_cart_time_hrs=2.6,
        review_left_after_return=0),
}


def banner(result):
    color, icon = BANNER_STYLE.get(result["banner"], ("#444", "ℹ️"))
    st.markdown(
        f"<div style='background:{color};color:white;padding:16px;border-radius:8px;"
        f"font-size:16px'>{icon} <b>{result['action']}</b><br>{result['customer_message']}"
        f"</div>", unsafe_allow_html=True)


def customer_portal():
    st.header("Fast Return & Exchange Center")
    st.caption("Submit a return and get an instant resolution.")

    choice = st.selectbox("Demo profile", list(DEMO_PROFILES.keys()))
    features = DEMO_PROFILES[choice]

    c1, c2, c3 = st.columns(3)
    c1.metric("Item category", features["product_category"])
    c2.metric("Refund (USD)", f"{features['refund_amount_requested_usd']:.0f}")
    c3.metric("Days to return", features["days_to_return"])

    if st.button("Submit Return Request", type="primary"):
        result = _engine().predict_return(features)
        banner(result)
        if result["fee_pkr"]:
            st.info(f"A handling/restocking fee of PKR {result['fee_pkr']} applies.")
        st.caption(f"Decided in {result['latency_ms']:.0f} ms")


def merchant_dashboard():
    st.header("Merchant Risk Dashboard")
    rows = []
    for name, feats in DEMO_PROFILES.items():
        r = _engine().predict_return(feats)
        rows.append({
            "profile": name, "category": feats["product_category"],
            "refund_usd": feats["refund_amount_requested_usd"],
            "class": r["predicted_class"], "action": r["action"],
            "fee_pkr": r["fee_pkr"], "latency_ms": round(r["latency_ms"], 1),
        })
    df = pd.DataFrame(rows)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Returns analyzed", len(df))
    protected = df.loc[df["action"].isin(["FRAUD_HOLD", "VERIFY_TAGS"]), "refund_usd"].sum()
    k2.metric("Value protected (USD)", f"{protected:.0f}")
    k3.metric("Fees applied (PKR)", int(df["fee_pkr"].sum()))
    honest = (df["action"] == "AUTO_APPROVE").mean() * 100
    k4.metric("Auto-approved %", f"{honest:.0f}%")

    st.dataframe(df, use_container_width=True)

    sel = st.selectbox("Inspect a return", df["profile"].tolist())
    result = _engine().predict_return(DEMO_PROFILES[sel])
    st.subheader(f"{sel} → {result['action']}")
    st.write("**Merchant note:**", result["merchant_note"])
    st.write("**Class probabilities**")
    st.bar_chart(pd.Series(result["probs"]))
    st.write("**Top reasons (SHAP)**")
    for feat, val in result["top_reasons"]:
        st.write(f"- `{feat}`: {val:+.3f}")

    with st.expander("Assumptions"):
        st.write("Handling cost and CO2 figures are illustrative estimates. "
                 "Thresholds are tuned on a held-out validation split. "
                 "Model trained on a public synthetic benchmark dataset.")


def main():
    st.sidebar.title("🛡️ ReturnGuard")
    view = st.sidebar.radio("View", ["Customer Portal", "Merchant Dashboard"])
    st.sidebar.caption("Explainable AI return-policy engine")
    if view == "Customer Portal":
        customer_portal()
    else:
        merchant_dashboard()


if __name__ == "__main__":
    main()
