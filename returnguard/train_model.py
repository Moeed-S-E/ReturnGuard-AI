import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import joblib
import json
import os
from data_pipeline import load_data, feature_engineering

def train():
    df = load_data('data/returns.csv')
    df = feature_engineering(df)
    
    label_map = {'Legitimate': 0, 'Policy Abuser': 1, 'Fraudulent Return': 2, 'Wardrobing': 3}
    y = df['abuse_type'].map(label_map)
    
    features = [
        'order_to_return_days', 'return_ratio', 'risk_exposure', 'dispute_frequency',
        'account_age_days', 'item_price', 'historical_return_rate', 'category_risk_index',
        'product_category', 'payment_method'
    ]
    X = df[features]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=4,
        class_weight='balanced',
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42
    )
    
    # We use Early Stopping
    callbacks = [lgb.early_stopping(stopping_rounds=20)]
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=callbacks
    )
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'macro_f1': f1_score(y_test, y_pred, average='macro'),
        'precision': precision_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(model, 'artifacts/model.joblib')
    
    with open('artifacts/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Training completed. Macro F1: {metrics['macro_f1']:.4f}")

if __name__ == "__main__":
    train()
