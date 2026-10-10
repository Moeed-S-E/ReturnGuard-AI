import joblib
import pandas as pd
import numpy as np
import shap
import time
import os

POLICY_CONFIG = {
    'thresholds': {
        'Fraudulent Return': 0.50,
        'Wardrobing': 0.60,
        'Policy Abuser': 0.55,
        'Legitimate': 0.75
    },
    'fee_pkr': 950
}

class PolicyEngine:
    def __init__(self):
        if not os.path.exists('artifacts/model.joblib'):
            raise RuntimeError("Model file not found. Please train the model first.")
        self.model = joblib.load('artifacts/model.joblib')
        self.explainer = shap.TreeExplainer(self.model)
        self.classes = ['Legitimate', 'Policy Abuser', 'Fraudulent Return', 'Wardrobing']
        self.class_map = {0: 'Legitimate', 1: 'Policy Abuser', 2: 'Fraudulent Return', 3: 'Wardrobing'}
        
    def predict_return(self, features: dict):
        start_time = time.perf_counter()
        
        df = pd.DataFrame([features])
        if 'product_category' in df.columns:
            df['product_category'] = df['product_category'].astype('category')
        if 'payment_method' in df.columns:
            df['payment_method'] = df['payment_method'].astype('category')
            
        probas = self.model.predict_proba(df)[0]
        prob_dict = {self.class_map[i]: float(probas[i]) for i in range(4)}
        
        predicted_class = "Uncertain"
        action = "LIGHT_VERIFY"
        fee_pkr = 0
        customer_message = "Your request is under review. Our team will contact you within 24 hours."
        merchant_note = "Requires lightweight verification."
        banner = "Yellow"
        
        # Severity-first evaluation
        if prob_dict['Fraudulent Return'] >= POLICY_CONFIG['thresholds']['Fraudulent Return']:
            predicted_class = 'Fraudulent Return'
            action = 'FRAUD_HOLD'
            customer_message = "Your request is under review. Our team will contact you within 24 hours."
            merchant_note = "High fraud probability. Manual review required."
            banner = "Red"
        elif prob_dict['Wardrobing'] >= POLICY_CONFIG['thresholds']['Wardrobing']:
            predicted_class = 'Wardrobing'
            action = 'VERIFY_TAGS'
            customer_message = "Quick check needed: drop the item at a store with tags attached, or upload a photo of the tag."
            merchant_note = "Wardrobing suspected. Check for tags."
            banner = "Yellow"
        elif prob_dict['Policy Abuser'] >= POLICY_CONFIG['thresholds']['Policy Abuser']:
            predicted_class = 'Policy Abuser'
            action = 'FEE_DEDUCTION'
            fee_pkr = POLICY_CONFIG['fee_pkr']
            customer_message = f"Return authorized. A PKR {fee_pkr} handling and restocking fee will be deducted from your refund."
            merchant_note = "Policy abuse pattern detected. Fee applied."
            banner = "Orange"
        elif prob_dict['Legitimate'] >= POLICY_CONFIG['thresholds']['Legitimate']:
            predicted_class = 'Legitimate'
            action = 'AUTO_APPROVE'
            customer_message = "Your return is approved! Show this QR code to the courier. Full refund on handoff."
            merchant_note = "Low risk. Auto-approved."
            banner = "Green"
        else:
            predicted_class = 'Uncertain'
            action = 'LIGHT_VERIFY'
            customer_message = "Quick check needed: upload a photo of the tag."
            merchant_note = "Uncertain classification. Light verification requested."
            banner = "Yellow"
            
        shap_values_obj = self.explainer(df)
        
        try:
            if len(shap_values_obj.shape) == 3:
                # Shape: (1, n_features, n_classes)
                class_idx = list(self.class_map.values()).index(predicted_class) if predicted_class != 'Uncertain' else np.argmax(probas)
                shap_vals = shap_values_obj.values[0, :, class_idx]
            else:
                shap_vals = shap_values_obj.values[0]
        except Exception:
            shap_vals = np.zeros(df.shape[1])
            
        top_reasons = []
        feature_names = df.columns.tolist()
        sorted_indices = np.argsort(np.abs(shap_vals))[::-1]
        for idx in sorted_indices[:3]:
            top_reasons.append((feature_names[idx], float(shap_vals[idx])))
            
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            "probs": prob_dict,
            "predicted_class": predicted_class,
            "action": action,
            "fee_pkr": fee_pkr,
            "customer_message": customer_message,
            "merchant_note": merchant_note,
            "top_reasons": top_reasons,
            "latency_ms": latency_ms,
            "banner": banner,
            "shap_explanation": shap_values_obj
        }

def get_policy_engine():
    if not os.path.exists('artifacts/model.joblib'):
        print("Model not found. Triggering training...")
        import train_model
        train_model.train()
    return PolicyEngine()
