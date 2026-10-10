"""ReturnGuard — full MVP Streamlit app.

Two views:
  * Customer Portal   — submit a return, get an instant explained resolution.
  * Merchant Dashboard — batch-scored returns, KPIs, business-impact panel,
                         SHAP reasons, and manual override with audit note.

    streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from policy_engine import get_policy_engine

st.set_page_config(page_title="ReturnGuard", page_icon="🛡️", layout="wide")

HERE = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    HERE.parent / "datasets" / "ecommerce_return_abuse_dataset.csv",
    HERE / "datasets" / "ecommerce_return_abuse_dataset.csv",
]

# ---- Impact assumptions (shown on screen, configurable) ----
USD_TO_PKR = 278.0
HANDLING_COST_USD = 30.0      # research estimate, per prevented consignment
CO2_PER_TRIP_KG = 1.0         # placeholder estimate per avoided trip

BANNER_STYLE = {
    "Green":  ("#1b7f3b", "✅"), "Yellow": ("#9a7d0a", "🔎"),
    "Orange": ("#b35900", "⚠️"), "Red":    ("#9b1c1c", "⛔"),
}


@st.cache_resource(show_spinner="Loading model…")
def engine():
    return get_policy_engine()


@st.cache_data(show_spinner="Loading sample returns…")
def sample_returns(n=40):
    for p in DATA_CANDIDATES:
        if p.exists():
            df = pd.read_csv(p)
            return df.sample(min(n, len(df)), random_state=7).reset_index(drop=True)
    return pd.DataFrame()


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


# ------------------------------------------------------------------ Customer
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
        result = engine().predict_return(features)
        banner(result)
        if result["fee_pkr"]:
            st.info(f"A handling/restocking fee of PKR {result['fee_pkr']} applies.")
        with st.expander("Why this decision?"):
            for feat, val in result["top_reasons"]:
                st.write(f"- `{feat}`: {val:+.3f}")
        st.caption(f"Decided in {result['latency_ms']:.0f} ms")


# ------------------------------------------------------------------ Merchant
def _score_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    eng = engine()
    for _, rec in df.iterrows():
        feats = rec.to_dict()
        feats.setdefault("order_date", "2023-06-01")
        feats.setdefault("return_date", "2023-06-10")
        r = eng.predict_return(feats)
        rows.append({
            "order_id": rec.get("order_id", "—"),
            "category": rec.get("product_category", "—"),
            "refund_usd": float(rec.get("refund_amount_requested_usd", 0) or 0),
            "class": r["predicted_class"], "action": r["action"],
            "fee_pkr": r["fee_pkr"],
            "true_label": rec.get("abuse_type", "—"),
        })
    return pd.DataFrame(rows)


def impact_panel(scored: pd.DataFrame):
    flagged = scored[scored["action"].isin(["FRAUD_HOLD", "VERIFY_TAGS"])]
    value_protected_usd = flagged["refund_usd"].sum()
    prevented = len(flagged)
    handling_saved_usd = prevented * HANDLING_COST_USD
    co2_avoided_kg = prevented * CO2_PER_TRIP_KG

    st.subheader("Business impact (illustrative)")
    i1, i2, i3 = st.columns(3)
    i1.metric("Value protected", f"PKR {value_protected_usd * USD_TO_PKR:,.0f}",
              help=f"~${value_protected_usd:,.0f} in refunds on flagged fraud/wardrobing returns")
    i2.metric("Handling cost saved", f"${handling_saved_usd:,.0f}",
              help=f"{prevented} prevented consignments × ${HANDLING_COST_USD}")
    i3.metric("CO₂ avoided", f"{co2_avoided_kg:,.1f} kg",
              help=f"{prevented} avoided trips × {CO2_PER_TRIP_KG} kg (estimate)")
    st.caption("Assumptions: handling $30/return, CO₂ 1.0 kg/trip, USD→PKR "
               f"{USD_TO_PKR:.0f}. Figures are estimates for demonstration.")


def merchant_dashboard():
    st.header("Merchant Risk Dashboard")

    source = st.radio("Data source", ["Demo profiles", "Sample from dataset"],
                      horizontal=True)
    if source == "Demo profiles":
        base = pd.DataFrame([{**{"order_id": name}, **f} for name, f in DEMO_PROFILES.items()])
    else:
        base = sample_returns(40)
        if base.empty:
            st.warning("Dataset not found; showing demo profiles instead.")
            base = pd.DataFrame([{**{"order_id": name}, **f} for name, f in DEMO_PROFILES.items()])

    scored = _score_frame(base)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Returns analyzed", len(scored))
    k2.metric("Flagged (fraud/wardrobe)",
              int(scored["action"].isin(["FRAUD_HOLD", "VERIFY_TAGS"]).sum()))
    k3.metric("Fees applied (PKR)", int(scored["fee_pkr"].sum()))
    k4.metric("Auto-approved %",
              f"{(scored['action'] == 'AUTO_APPROVE').mean() * 100:.0f}%")

    impact_panel(scored)

    st.subheader("Analyzed returns")
    f1, f2 = st.columns(2)
    cls_filter = f1.multiselect("Filter class", sorted(scored["class"].unique()))
    act_filter = f2.multiselect("Filter action", sorted(scored["action"].unique()))
    view = scored.copy()
    if cls_filter:
        view = view[view["class"].isin(cls_filter)]
    if act_filter:
        view = view[view["action"].isin(act_filter)]
    st.dataframe(view, use_container_width=True)

    st.subheader("Inspect & override")
    idx = st.selectbox("Select a return (row index)", view.index.tolist()
                       if len(view) else scored.index.tolist())
    rec = base.loc[idx].to_dict()
    rec.setdefault("order_date", "2023-06-01")
    rec.setdefault("return_date", "2023-06-10")
    result = engine().predict_return(rec)

    cL, cR = st.columns([1, 1])
    with cL:
        st.write(f"**AI decision:** {result['action']}  ({result['predicted_class']})")
        st.write("**Merchant note:**", result["merchant_note"])
        st.bar_chart(pd.Series(result["probs"]))
    with cR:
        st.write("**Top reasons (SHAP)**")
        for feat, val in result["top_reasons"]:
            st.write(f"- `{feat}`: {val:+.3f}")
        st.markdown("**Manual override**")
        new_action = st.selectbox("Override action",
                                  ["(keep AI decision)", "AUTO_APPROVE", "VERIFY_TAGS",
                                   "FEE_DEDUCTION", "FRAUD_HOLD", "LIGHT_VERIFY"])
        note = st.text_input("Audit note (required to override)")
        if st.button("Apply override"):
            if new_action == "(keep AI decision)":
                st.info("No override applied.")
            elif not note.strip():
                st.error("An audit note is required to override.")
            else:
                st.session_state.setdefault("audit_log", []).append(
                    {"order_id": rec.get("order_id", "—"),
                     "from": result["action"], "to": new_action, "note": note})
                st.success(f"Override recorded: {result['action']} → {new_action}")

    if st.session_state.get("audit_log"):
        st.subheader("Override audit log")
        st.dataframe(pd.DataFrame(st.session_state["audit_log"]),
                     use_container_width=True)

    with st.expander("Assumptions & model notes"):
        st.write("- Thresholds tuned on a held-out validation split.")
        st.write("- Model trained on a public synthetic benchmark; retrain on "
                 "real merchant data before production.")
        st.write("- Fraud class routes to human review; never auto-punished.")
        st.write("- No sensitive attributes used as model features.")


def main():
    st.sidebar.title("🛡️ ReturnGuard")
    view = st.sidebar.radio("View", ["Customer Portal", "Merchant Dashboard"])
    st.sidebar.caption("Explainable AI return-policy engine")
    st.sidebar.markdown("---")
    st.sidebar.caption("Severity order: Fraud → Wardrobe → Abuser → Legit")
    if view == "Customer Portal":
        customer_portal()
    else:
        merchant_dashboard()


if __name__ == "__main__":
    main()
