"""ReturnGuard — full MVP Streamlit app.

Two views:
  * Customer Portal   — submit a return, get an instant explained resolution.
  * Merchant Dashboard — batch-scored returns, KPIs, business-impact panel,
                         SHAP reasons, and manual override with audit note.

    streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path
import time

import pandas as pd
import streamlit as st

from policy_engine import get_policy_engine

st.set_page_config(page_title="ReturnGuard", page_icon="security", layout="wide")

# -- Premium Custom CSS Injection & Material Icons --
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0');

html, body, [class*="css"]  {
    font-family: 'Outfit', sans-serif;
}

.material-symbols-outlined {
  vertical-align: bottom;
  margin-right: 6px;
  font-size: 1.2em;
}

/* Hide Streamlit default components to look like raw HTML */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* Remove default padding for a flush, web-like appearance */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px !important;
}

/* Premium gradient background */
.stApp {
    background: radial-gradient(circle at 15% 50%, #1e1b4b, #0f172a);
    color: #f8fafc;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: rgba(15, 23, 42, 0.6) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(20px);
}

/* Card-like containers for metrics */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 40px rgba(0, 0, 0, 0.3);
}

/* Tooltip container styling to look modern */
.tooltip-custom {
    border-bottom: 1px dotted #94a3b8;
    cursor: help;
    color: #cbd5e1;
}

/* Buttons with gradient and glow */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.6);
    color: white;
}

/* Dataframe styling */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.1);
}

/* Headers */
h1, h2, h3 {
    color: #f8fafc !important;
    font-weight: 700 !important;
}

/* Tab Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: transparent;
    border-radius: 4px 4px 0 0;
    gap: 1px;
    padding-top: 10px;
    padding-bottom: 10px;
}

/* Banner animations */
@keyframes slideUpFade {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
.animated-banner {
    animation: slideUpFade 0.6s cubic-bezier(0.16, 1, 0.3, 1);
}
</style>
""", unsafe_allow_html=True)


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
    "Green":  ("linear-gradient(135deg, #059669 0%, #10b981 100%)", "check_circle", "rgba(16, 185, 129, 0.3)"), 
    "Yellow": ("linear-gradient(135deg, #d97706 0%, #f59e0b 100%)", "search", "rgba(245, 158, 11, 0.3)"),
    "Orange": ("linear-gradient(135deg, #ea580c 0%, #f97316 100%)", "warning", "rgba(249, 115, 22, 0.3)"), 
    "Red":    ("linear-gradient(135deg, #b91c1c 0%, #ef4444 100%)", "block", "rgba(239, 68, 68, 0.3)"),
}


@st.cache_resource(show_spinner="Initializing AI Engine...")
def engine():
    return get_policy_engine()


@st.cache_data(show_spinner="Loading return data...")
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
    action = result.get('action', 'LIGHT_VERIFY')
    if action == "AUTO_APPROVE":
        b_key = "Green"
    elif action == "VERIFY_TAGS":
        b_key = "Yellow"
    elif action == "FEE_DEDUCTION":
        b_key = "Orange"
    elif action == "FRAUD_HOLD":
        b_key = "Red"
    else:
        b_key = "Yellow"

    bg_grad, icon, shadow = BANNER_STYLE.get(b_key, ("linear-gradient(135deg, #4b5563 0%, #374151 100%)", "info", "rgba(75, 85, 99, 0.3)"))
    st.markdown(
        f"""
        <div class='animated-banner' style='background:{bg_grad}; color:white; padding:24px; border-radius:16px; 
             box-shadow: 0 10px 25px {shadow}; margin-top: 20px; border: 1px solid rgba(255,255,255,0.2);'
             title='The final decision issued by the AI based on its risk evaluation.'>
            <h3 style='margin:0 0 10px 0; color:white !important; display:flex; align-items:center; gap:10px;'>
                <span class="material-symbols-outlined">{icon}</span> {result['action']}
            </h3>
            <p style='font-size:18px; margin:0; opacity: 0.95;'>{result['customer_message']}</p>
        </div>
        """, unsafe_allow_html=True)


# ------------------------------------------------------------------ Customer
def customer_portal():
    st.markdown("<h1><span class='material-symbols-outlined'>shopping_bag</span> Fast Return & Exchange Center</h1>", unsafe_allow_html=True)
    st.caption("Submit a return request and our AI will process it instantly. Hover over elements to learn what they mean!")

    st.markdown("<br>", unsafe_allow_html=True)
    choice = st.selectbox(
        "Select a customer profile to test the AI:", 
        list(DEMO_PROFILES.keys()),
        help="Choose a simulated customer persona to see how the AI handles different return scenarios (from honest shoppers to potential fraud)."
    )
    features = DEMO_PROFILES[choice]

    st.markdown("### <span class='material-symbols-outlined'>receipt_long</span> Order Details", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Item category", 
        features["product_category"],
        help="The type of product being returned. Some categories (like electronics or high-end clothing) have a higher base risk for abuse."
    )
    c2.metric(
        "Refund Amount", 
        f"${features['refund_amount_requested_usd']:.0f}",
        help="The amount the customer expects to be refunded. Higher amounts represent larger financial exposure."
    )
    c3.metric(
        "Days since purchase", 
        features["days_to_return"],
        help="Number of days between the order date and the return request. Very short or very long return windows can indicate irregular behavior."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Submit Return Request", type="primary", use_container_width=True, help="Click to send this data to the AI model for an instant decision."):
        with st.spinner("Analyzing return patterns and history..."):
            time.sleep(0.8) # Micro-animation feel
            result = engine().predict_return(features)
        
        banner(result)
        
        if result["fee_pkr"]:
            st.warning(f"A handling/restocking fee of **PKR {result['fee_pkr']}** applies according to our policy.")
            
        with st.expander("Why did the AI make this decision?"):
            st.markdown(
                "The AI explains its reasoning using **SHAP Values** — a transparent way to see exactly which factors pushed the decision towards or away from suspicion.<br>"
                "<span class='tooltip-custom' title='Green means it made the AI MORE suspicious (riskier). Red means it made the AI LESS suspicious (safer).'>Hover here for a quick tip on reading these values.</span>", 
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)
            for feat, val in result["top_reasons"]:
                color = "#10b981" if val < 0 else "#ef4444"
                st.markdown(
                    f"- `{feat}`: <span style='color:{color}; font-weight:bold;' title='This feature contributed {val:+.3f} to the AI\\'s risk score for the selected action.'>{val:+.3f}</span>", 
                    unsafe_allow_html=True
                )
        st.caption(f"Decided in {result['latency_ms']:.1f} ms")


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

    i1, i2, i3 = st.columns(3)
    i1.metric(
        "Value protected (PKR)", 
        f"{value_protected_usd * USD_TO_PKR:,.0f}",
        help=f"Calculated by summing the refund amounts for all returns flagged as FRAUD or WARDROBING. Represents the direct financial loss prevented."
    )
    i2.metric(
        "Handling cost saved (USD)", 
        f"${handling_saved_usd:,.0f}",
        help=f"Estimated logistical cost saved by not shipping and processing invalid returns ({prevented} prevented items × ${HANDLING_COST_USD} handling fee)."
    )
    i3.metric(
        "CO₂ emissions avoided", 
        f"{co2_avoided_kg:,.1f} kg",
        help=f"Estimated environmental impact prevented by avoiding unnecessary return shipments ({prevented} prevented items × {CO2_PER_TRIP_KG} kg per trip)."
    )
    st.caption("Assumptions: handling $30/return, CO₂ 1.0 kg/trip. Figures are illustrative estimates.")


def merchant_dashboard():
    st.markdown("<h1><span class='material-symbols-outlined'>dashboard</span> Merchant Risk Dashboard</h1>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        source = st.radio(
            "Select Data Source", 
            ["Demo profiles", "Sample from dataset"], 
            horizontal=True,
            help="Choose 'Demo profiles' to evaluate hand-crafted examples, or 'Sample from dataset' to randomly pull 40 real returns from the underlying dataset."
        )
    
    if source == "Demo profiles":
        base = pd.DataFrame([{**{"order_id": name}, **f} for name, f in DEMO_PROFILES.items()])
    else:
        base = sample_returns(40)
        if base.empty:
            st.warning("Dataset not found; showing demo profiles instead.")
            base = pd.DataFrame([{**{"order_id": name}, **f} for name, f in DEMO_PROFILES.items()])

    with st.spinner("Analyzing data batch..."):
        scored = _score_frame(base)

    st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Overview & Impact", "Return Data", "Inspect & Override"])

    with tab1:
        st.subheader("Key Performance Indicators")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            "Returns analyzed", 
            len(scored),
            help="Total number of return requests processed in this batch."
        )
        k2.metric(
            "Flagged for Review", 
            int(scored["action"].isin(["FRAUD_HOLD", "VERIFY_TAGS"]).sum()),
            help="Number of returns halted because the AI detected strong signals of Fraud or Wardrobing."
        )
        k3.metric(
            "Fees applied (PKR)", 
            f"{int(scored['fee_pkr'].sum()):,}",
            help="Total restocking fees automatically applied to users identified as chronic Policy Abusers."
        )
        k4.metric(
            "Auto-approved %", 
            f"{(scored['action'] == 'AUTO_APPROVE').mean() * 100:.0f}%",
            help="Percentage of returns deemed low-risk and approved instantly without human intervention."
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Business Impact")
        impact_panel(scored)

    with tab2:
        st.subheader("Analyzed Returns Table")
        st.markdown(
            "<span class='tooltip-custom' title='Use these dropdowns to filter the table below to find specific types of returns.'>Filter the data below to spot trends.</span>", 
            unsafe_allow_html=True
        )
        f1, f2 = st.columns(2)
        cls_filter = f1.multiselect(
            "Filter by Prediction Class", 
            sorted(scored["class"].unique()),
            help="Filter by the raw AI classification (e.g., 'Legitimate' vs 'Fraudulent Return')."
        )
        act_filter = f2.multiselect(
            "Filter by Action", 
            sorted(scored["action"].unique()),
            help="Filter by the final business action taken (e.g., 'AUTO_APPROVE' vs 'FRAUD_HOLD')."
        )
        
        view = scored.copy()
        if cls_filter:
            view = view[view["class"].isin(cls_filter)]
        if act_filter:
            view = view[view["action"].isin(act_filter)]
            
        st.dataframe(view, use_container_width=True, height=400)

    with tab3:
        st.subheader("Deep Dive & Override")
        st.markdown(
            "Inspect an individual return to see *exactly* why the AI made its decision, and manually override it if necessary.",
            help="This section is designed for human risk analysts to review complex cases and document manual interventions."
        )
        
        view = scored.copy()
        if cls_filter:
            view = view[view["class"].isin(cls_filter)]
        if act_filter:
            view = view[view["action"].isin(act_filter)]
            
        idx = st.selectbox(
            "Select a return (row index) to inspect", 
            view.index.tolist() if len(view) else scored.index.tolist(),
            help="Pick a row from the data to view its detailed AI scorecard."
        )
        
        if idx is not None:
            rec = base.loc[idx].to_dict()
            rec.setdefault("order_date", "2023-06-01")
            rec.setdefault("return_date", "2023-06-10")
            result = engine().predict_return(rec)

            st.markdown(f"### <span class='material-symbols-outlined'>assignment</span> Inspecting Order: `{rec.get('order_id', 'Unknown')}`", unsafe_allow_html=True)
            
            cL, cR = st.columns([1, 1])
            with cL:
                st.markdown(f"**AI decision:** `{result['action']}`")
                st.markdown(f"**Predicted Class:** `{result['predicted_class']}`")
                st.markdown(f"**Merchant Note:** _{result['merchant_note']}_")
                
                st.markdown(
                    "#### Confidence Scores",
                    help="The raw probability that the return belongs to each specific risk category. The AI uses thresholds on these scores to determine the final action."
                )
                st.bar_chart(pd.Series(result["probs"]), height=250)
                
            with cR:
                st.markdown(
                    "#### Top Factors (SHAP Values)",
                    help="The three features from the customer's profile that had the largest mathematical impact on the AI's risk score."
                )
                for feat, val in result["top_reasons"]:
                    color = "#10b981" if val < 0 else "#ef4444"
                    st.markdown(f"- `{feat}`: <span style='color:{color}; font-weight:bold;' title='Feature weight'>{val:+.3f}</span>", unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("#### Manual Override", help="Use this form to change the AI's decision and log it for compliance.")
                new_action = st.selectbox("Override Action",
                                          ["(keep AI decision)", "AUTO_APPROVE", "VERIFY_TAGS",
                                           "FEE_DEDUCTION", "FRAUD_HOLD", "LIGHT_VERIFY"])
                note = st.text_input("Audit note (required to override)", help="A brief explanation for why you are overriding the AI. This is saved to the system audit log.")
                if st.button("Apply Override", type="primary"):
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
                st.markdown("---")
                st.subheader("Override Audit Log")
                st.dataframe(pd.DataFrame(st.session_state["audit_log"]), use_container_width=True)

    with st.sidebar.expander("Model Transparency Notes"):
        st.markdown("""
        - **Thresholds**: Evaluated dynamically based on acceptable business risk.
        - **Data**: Trained on historical e-commerce returns.
        - **Safety**: Fraud instances route to human review; never auto-punished without cause.
        - **Fairness**: Demographics are strictly excluded from the decision engine.
        """)


def main():
    st.sidebar.markdown(
        "<h1 style='text-align: center; margin-bottom: 0;'><span class='material-symbols-outlined' style='font-size:1.5rem'>security</span> ReturnGuard</h1>", 
        unsafe_allow_html=True
    )
    st.sidebar.markdown(
        "<p style='text-align: center; color: #94a3b8; font-size: 0.9rem; margin-bottom: 2rem;'>Explainable AI return-policy engine</p>", 
        unsafe_allow_html=True
    )
    
    view = st.sidebar.radio(
        "Navigation", 
        ["Customer Portal", "Merchant Dashboard"],
        help="Switch between the customer-facing returns flow and the internal risk analyst dashboard."
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Severity evaluation order: Fraud → Wardrobe → Abuser → Legit")
    
    if view == "Customer Portal":
        customer_portal()
    else:
        merchant_dashboard()


if __name__ == "__main__":
    main()
