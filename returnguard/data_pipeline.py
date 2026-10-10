import os
import pandas as pd
import numpy as np

def generate_synthetic_data(n_samples=5000):
    np.random.seed(42)
    classes = ['Legitimate', 'Policy Abuser', 'Fraudulent Return', 'Wardrobing']
    probs = [0.701, 0.119, 0.102, 0.078]
    
    y = np.random.choice(classes, size=n_samples, p=probs)
    
    data = []
    for label in y:
        if label == 'Legitimate':
            order_to_return_days = np.random.uniform(5, 20)
            item_price = np.random.uniform(20, 100)
            prior_disputes_count = np.random.poisson(0.1)
            account_age_days = np.random.uniform(100, 1000)
            total_purchases = np.random.randint(1, 20)
            total_returns = np.random.randint(0, int(total_purchases * 0.2) + 1)
            product_category = np.random.choice(['Apparel', 'Electronics', 'Home', 'Beauty'])
            payment_method = np.random.choice(['Credit Card', 'Debit Card', 'PayPal'])
            address_mismatch = 0
            
        elif label == 'Wardrobing':
            order_to_return_days = np.random.uniform(0, 2)
            item_price = np.random.uniform(150, 500)
            prior_disputes_count = np.random.poisson(0.5)
            account_age_days = np.random.uniform(30, 500)
            total_purchases = np.random.randint(5, 50)
            total_returns = int(total_purchases * np.random.uniform(0.5, 0.9))
            product_category = np.random.choice(['Apparel', 'Luxury'])
            payment_method = np.random.choice(['Credit Card'])
            address_mismatch = 0
            
        elif label == 'Policy Abuser':
            order_to_return_days = np.random.uniform(15, 30)
            item_price = np.random.uniform(10, 50)
            prior_disputes_count = np.random.poisson(1.0)
            account_age_days = np.random.uniform(100, 1000)
            total_purchases = np.random.randint(10, 100)
            total_returns = int(total_purchases * np.random.uniform(0.7, 1.0))
            product_category = np.random.choice(['Apparel', 'Home', 'Beauty'])
            payment_method = np.random.choice(['Credit Card', 'PayPal'])
            address_mismatch = 0
            
        elif label == 'Fraudulent Return':
            order_to_return_days = np.random.uniform(0, 1)
            item_price = np.random.uniform(200, 1000)
            prior_disputes_count = np.random.randint(3, 10)
            account_age_days = np.random.uniform(1, 60)
            total_purchases = np.random.randint(1, 5)
            total_returns = total_purchases
            product_category = np.random.choice(['Electronics'])
            payment_method = np.random.choice(['Credit Card'])
            address_mismatch = 1
            
        return_ratio = total_returns / max(total_purchases, 1)
        risk_exposure = item_price * (1 + prior_disputes_count)
        dispute_frequency = prior_disputes_count / max(account_age_days / 30, 1)
        historical_return_rate = return_ratio
        
        data.append({
            'order_to_return_days': order_to_return_days,
            'item_price': item_price,
            'prior_disputes_count': prior_disputes_count,
            'account_age_days': account_age_days,
            'total_purchases': total_purchases,
            'total_returns': total_returns,
            'product_category': product_category,
            'payment_method': payment_method,
            'address_mismatch': address_mismatch,
            'return_ratio': return_ratio,
            'risk_exposure': risk_exposure,
            'dispute_frequency': dispute_frequency,
            'historical_return_rate': historical_return_rate,
            'abuse_type': label
        })
        
    df = pd.DataFrame(data)
    cat_risk = df.groupby('product_category')['abuse_type'].apply(lambda x: (x != 'Legitimate').mean()).to_dict()
    df['category_risk_index'] = df['product_category'].map(cat_risk)
    
    return df

def load_data(filepath='data/returns.csv'):
    if os.path.exists(filepath):
        print(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        return df
    else:
        print("Data file not found. Generating synthetic fallback...")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df = generate_synthetic_data()
        df.to_csv(filepath, index=False)
        return df

def feature_engineering(df, is_train=True, cat_risk_map=None):
    df = df.copy()
    
    if 'product_category' in df.columns:
        df['product_category'] = df['product_category'].astype('category')
    if 'payment_method' in df.columns:
        df['payment_method'] = df['payment_method'].astype('category')
        
    return df
