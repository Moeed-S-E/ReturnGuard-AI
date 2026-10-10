PROFILES = {
    "Fatima (Honest)": {
        "order_id": "#1001",
        "item": "Men's Casual Shirt",
        "price": 2200,
        "reason": "Doesn't fit",
        "features": {
            "order_to_return_days": 7.0,
            "return_ratio": 0.04,
            "risk_exposure": 2200.0,
            "dispute_frequency": 0.0,
            "account_age_days": 300.0,
            "item_price": 2200.0,
            "historical_return_rate": 0.04,
            "category_risk_index": 0.1,
            "product_category": "Apparel",
            "payment_method": "Credit Card"
        }
    },
    "Sara (Wardrober)": {
        "order_id": "#2042",
        "item": "Luxury Evening Gown",
        "price": 45000,
        "reason": "Changed my mind",
        "features": {
            "order_to_return_days": 1.0,
            "return_ratio": 0.72,
            "risk_exposure": 45000.0 * 2.0,
            "dispute_frequency": 0.1,
            "account_age_days": 100.0,
            "item_price": 45000.0,
            "historical_return_rate": 0.72,
            "category_risk_index": 0.5,
            "product_category": "Luxury",
            "payment_method": "Credit Card"
        }
    },
    "Ali (Abuser)": {
        "order_id": "#3089",
        "item": "Shoes",
        "price": 3500,
        "reason": "Defective",
        "features": {
            "order_to_return_days": 28.0,
            "return_ratio": 0.91,
            "risk_exposure": 3500.0 * (1 + 2),
            "dispute_frequency": 0.5,
            "account_age_days": 400.0,
            "item_price": 3500.0,
            "historical_return_rate": 0.91,
            "category_risk_index": 0.2,
            "product_category": "Apparel",
            "payment_method": "Credit Card"
        }
    },
    "Zain (Fraud)": {
        "order_id": "#4401",
        "item": "Smart Watch",
        "price": 15000,
        "reason": "Not as described",
        "features": {
            "order_to_return_days": 0.0,
            "return_ratio": 1.0,
            "risk_exposure": 15000.0 * (1 + 4),
            "dispute_frequency": 4.0 / max(10.0 / 30.0, 1.0),
            "account_age_days": 10.0,
            "item_price": 15000.0,
            "historical_return_rate": 1.0,
            "category_risk_index": 0.8,
            "product_category": "Electronics",
            "payment_method": "Credit Card"
        }
    }
}
