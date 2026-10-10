"""Data loading and schema-aligned synthetic fallback for ReturnGuard.

The real dataset lives at datasets/ecommerce_return_abuse_dataset.csv. If it is
missing, generate_synthetic_data() produces rows with the SAME column schema and
plausible per-class patterns so the rest of the pipeline (train_model.py,
policy_engine.py) works unchanged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REAL_PATHS = [
    HERE.parent / "datasets" / "ecommerce_return_abuse_dataset.csv",
    HERE / "datasets" / "ecommerce_return_abuse_dataset.csv",
]

CLASSES = ["Legitimate", "Policy Abuser", "Fraudulent Return", "Wardrobing"]
LABEL_MAP = {"Legitimate": 0, "Policy Abuser": 1, "Fraudulent Return": 2, "Wardrobing": 3}
CLASS_MIX = [0.701, 0.119, 0.102, 0.078]

CATEGORIES = ["Clothing", "Electronics", "Home & Kitchen", "Books", "Toys",
              "Beauty", "Shoes", "Sports", "Jewelry", "Furniture", "Grocery", "Tools"]
SEGMENTS = ["New", "Bronze", "Silver", "Gold"]
COUNTRIES = ["US", "CA", "FR", "DE", "UK"]
PLATFORMS = ["Web Browser", "Tablet App", "Mobile App"]
DEVICES = ["iPhone", "iPad", "MacBook", "Android", "Windows PC"]
PAYMENTS = ["Credit Card", "PayPal", "Crypto", "Buy Now Pay Later", "Debit Card"]
CARRIERS = ["FedEx", "USPS", "UPS", "OnTrac", "DHL"]
REASONS = ["Defective/Broken", "Wrong Item Sent", "Changed Mind",
           "No Longer Needed", "Item Not Received"]


def _rng(seed=42):
    return np.random.default_rng(seed)


def generate_synthetic_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate schema-matched synthetic rows with class-dependent patterns."""
    rng = _rng(seed)
    labels = rng.choice(CLASSES, size=n_samples, p=CLASS_MIX)
    rows = []
    base = pd.Timestamp("2023-01-01")
    for lbl in labels:
        if lbl == "Legitimate":
            d2r = int(rng.integers(5, 25)); rate = rng.uniform(1, 12)
            orders = int(rng.integers(20, 80)); hv = 0
            aov = rng.uniform(40, 220); disputes = int(rng.poisson(0.1))
            opened = 0; intact = 1; photo = 0; track = 1
            wishlist = rng.uniform(20, 60); reason = "Defective/Broken"
            diff_acct = 0; multi = 0; addr = 0
        elif lbl == "Wardrobing":
            d2r = int(rng.integers(30, 55)); rate = rng.uniform(30, 55)
            orders = int(rng.integers(20, 50)); hv = 1
            aov = rng.uniform(200, 450); disputes = int(rng.poisson(1.0))
            opened = 1; intact = 1; photo = 0; track = 1
            wishlist = rng.uniform(1, 6); reason = "Changed Mind"
            diff_acct = 0; multi = 0; addr = 0
        elif lbl == "Policy Abuser":
            d2r = int(rng.integers(10, 25)); rate = rng.uniform(55, 80)
            orders = int(rng.integers(40, 100)); hv = 0
            aov = rng.uniform(60, 160); disputes = int(rng.poisson(1.5))
            opened = 1; intact = 0; photo = 0; track = 1
            wishlist = rng.uniform(20, 50); reason = "Changed Mind"
            diff_acct = 0; multi = 0; addr = 0
        else:  # Fraudulent Return
            d2r = int(rng.integers(0, 6)); rate = rng.uniform(35, 60)
            orders = int(rng.integers(10, 35)); hv = 1
            aov = rng.uniform(300, 650); disputes = int(rng.integers(3, 7))
            opened = 0; intact = 1; photo = 1; track = 0
            wishlist = rng.uniform(1, 5); reason = "Item Not Received"
            diff_acct = int(rng.random() < 0.3); multi = int(rng.random() < 0.3)
            addr = int(rng.random() < 0.3)

        returns = min(orders, max(1, int(orders * rate / 100)))
        order_date = base + pd.Timedelta(days=int(rng.integers(0, 300)))
        return_date = order_date + pd.Timedelta(days=d2r)
        rows.append(dict(
            order_id=f"SYN{rng.integers(1_000_000, 9_999_999)}",
            customer_id=f"CUST{rng.integers(100000, 999999)}",
            age=int(rng.integers(18, 70)), account_age_days=int(rng.integers(60, 2000)),
            customer_segment=rng.choice(SEGMENTS), country=rng.choice(COUNTRIES),
            platform=rng.choice(PLATFORMS), device_type=rng.choice(DEVICES),
            payment_method=rng.choice(PAYMENTS), product_category=rng.choice(CATEGORIES),
            avg_order_value_usd=round(aov, 2),
            refund_amount_requested_usd=round(aov * rng.uniform(0.9, 1.0), 2),
            is_high_value_item=hv, discount_used=int(rng.random() < 0.3),
            order_date=order_date.strftime("%Y-%m-%d"),
            return_date=return_date.strftime("%Y-%m-%d"), days_to_return=d2r,
            return_reason=reason, total_orders_lifetime=orders,
            total_returns_lifetime=returns, return_rate_pct=round(rate, 1),
            item_returned_opened=opened, return_packaging_intact=intact,
            photo_evidence_provided=photo, tracking_number_valid=track,
            shipping_carrier=rng.choice(CARRIERS), address_change_before_delivery=addr,
            refund_to_different_account=diff_acct, multiple_accounts_flag=multi,
            customer_support_contacts=int(rng.integers(0, 5)),
            previous_dispute_count=disputes,
            wishlist_to_cart_time_hrs=round(wishlist, 1),
            review_left_after_return=int(rng.random() < 0.2),
            abuse_type=lbl, abuse_label=LABEL_MAP[lbl],
        ))
    return pd.DataFrame(rows)


def load_data(prefer_real: bool = True) -> pd.DataFrame:
    """Load the real dataset if present, else synthesize a schema-matched set."""
    if prefer_real:
        for p in REAL_PATHS:
            if p.exists():
                print(f"Loading real dataset: {p}")
                return pd.read_csv(p)
    print("Real dataset not found — generating synthetic fallback (5000 rows).")
    return generate_synthetic_data()


if __name__ == "__main__":
    df = load_data()
    print(df.shape)
    print(df["abuse_type"].value_counts())
