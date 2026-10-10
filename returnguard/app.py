import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from policy_engine import get_policy_engine
import time
import demo_profiles

st.set_page_config(layout="wide", page_title="ReturnGuard")

@st.cache_resource
def load_engine():
    return get_policy_engine()

engine = load_engine()

def customer_portal():
    st.title("Fast Return & Exchange Center")
    
    profile_name = st.selectbox("Demo Profile", list(demo_profiles.PROFILES.keys()))
    profile = demo_profiles.PROFILES[profile_name]
    
    st.write(f"**Order ID:** {profile['order_id']}")
    st.write(f"**Item:** {profile['item']} - PKR {profile['price']}")
    st.write(f"**Reason:** {profile['reason']}")
    
    if st.button("Submit Return Request"):
        result = engine.predict_return(profile['features'])
        
        banner_color = {
            'Green': 'success',
            'Yellow': 'warning',
            'Orange': 'warning',
            'Red': 'error'
        }.get(result['banner'], 'info')
        
        if result['banner'] == 'Green':
            st.success(result['customer_message'])
            st.image("https://api.qrserver.com/v1/create-qr-code/?size=150x150&data=ReturnQR", width=150)
        elif result['banner'] == 'Yellow':
            st.warning(result['customer_message'])
        elif result['banner'] == 'Orange':
            st.warning(result['customer_message'])
        elif result['banner'] == 'Red':
            st.error(result['customer_message'])
        
        st.write(f"*Latency: {result['latency_ms']:.1f} ms*")

def merchant_dashboard():
    st.title("Merchant Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Returns Analyzed", "1,245")
    col2.metric("Value Protected", "PKR 450,000")
    col3.metric("CO2 Avoided", "124 kg")
    col4.metric("Honest Approval %", "72%")
    
    st.subheader("Recent Returns")
    
    data = []
    for name, p in demo_profiles.PROFILES.items():
        res = engine.predict_return(p['features'])
        data.append({
            'Order': p['order_id'],
            'Customer': name,
            'Category': p['features']['product_category'],
            'Price': p['price'],
            'Class': res['predicted_class'],
            'Action': res['action']
        })
    df = pd.DataFrame(data)
    st.dataframe(df)
    
    st.subheader("Return Analysis")
    selected_order = st.selectbox("Select Order to Analyze", df['Order'])
    
    profile_name = next(name for name, p in demo_profiles.PROFILES.items() if p['order_id'] == selected_order)
    profile = demo_profiles.PROFILES[profile_name]
    
    result = engine.predict_return(profile['features'])
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("### Predictions")
        st.bar_chart(pd.Series(result['probs']))
        
        st.write("### SHAP Waterfall")
        exp = result['shap_explanation']
        class_map = {0: 'Legitimate', 1: 'Policy Abuser', 2: 'Fraudulent Return', 3: 'Wardrobing'}
        pred_idx = list(class_map.values()).index(result['predicted_class']) if result['predicted_class'] != 'Uncertain' else np.argmax(list(result['probs'].values()))
        
        fig, ax = plt.subplots(figsize=(8, 4))
        if len(exp.shape) == 3:
            shap.plots.waterfall(exp[0, :, pred_idx], show=False)
        else:
            shap.plots.waterfall(exp[0], show=False)
        st.pyplot(fig)
        
    with col_b:
        st.write("### Explanation")
        for reason, val in result['top_reasons']:
            direction = "increased" if val > 0 else "decreased"
            st.write(f"- **{reason}** {direction} the risk score by {abs(val):.2f}.")
            
        st.write("### Manual Override")
        st.text_input("Override Note")
        st.button("Submit Override")
        
    st.write("---")
    st.expander("Assumptions").write("Handling cost: $30/return. CO2 factor: 1.0 kg/trip. Thresholds: Legitimate >= 0.75, Abuser >= 0.55, Wardrobing >= 0.60, Fraud >= 0.50.")

view = st.sidebar.radio("View", ["Customer Portal", "Merchant Dashboard"])
if view == "Customer Portal":
    customer_portal()
else:
    merchant_dashboard()
