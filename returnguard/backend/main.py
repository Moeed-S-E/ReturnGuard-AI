"""ReturnGuard REST API (FastAPI).

Endpoints
  GET  /                 service info
  GET  /v1/health        health check (no auth)
  GET  /v1/plans         list subscription plans (no auth)
  POST /v1/subscribe     'purchase' a plan -> issues an API key (payment stubbed)
  POST /v1/predict       score a return request (requires X-API-Key)
  GET  /v1/usage         usage/quota for the calling key (requires X-API-Key)

Auth: pass the key in the `X-API-Key` header. Quota + a simple per-minute
rate limit are enforced per key. Free tier works immediately via /v1/subscribe
(plan=free) or the built-in demo key.
"""
from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Make the parent returnguard package importable (policy_engine lives there).
BACKEND_DIR = Path(__file__).resolve().parent
PKG_ROOT = BACKEND_DIR.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from policy_engine import PolicyEngine  # noqa: E402

from . import store  # noqa: E402
from .plans import PLANS, get_plan  # noqa: E402

_engine: Optional[PolicyEngine] = None
_rate_buckets: dict[str, deque] = defaultdict(deque)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    store.init_db()
    store.ensure_demo_key()
    _engine = PolicyEngine()
    yield


app = FastAPI(
    title="ReturnGuard API",
    version="1.0.0",
    description="Explainable AI return-abuse scoring as a service.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ----------------------------- models -----------------------------
class ReturnFeatures(BaseModel):
    """Raw return-request features. Mirrors the dataset schema."""
    order_id: str = "API"
    customer_id: str = "API"
    age: int = 35
    account_age_days: int = 365
    customer_segment: str = "Bronze"
    country: str = "US"
    platform: str = "Web Browser"
    device_type: str = "iPhone"
    payment_method: str = "Credit Card"
    product_category: str = "Clothing"
    avg_order_value_usd: float = 100.0
    refund_amount_requested_usd: float = 95.0
    is_high_value_item: int = 0
    discount_used: int = 0
    order_date: str = "2023-06-01"
    return_date: str = "2023-06-10"
    days_to_return: int = 9
    return_reason: str = "Defective/Broken"
    total_orders_lifetime: int = 20
    total_returns_lifetime: int = 2
    return_rate_pct: float = 10.0
    item_returned_opened: int = 0
    return_packaging_intact: int = 1
    photo_evidence_provided: int = 0
    tracking_number_valid: int = 1
    shipping_carrier: str = "FedEx"
    address_change_before_delivery: int = 0
    refund_to_different_account: int = 0
    multiple_accounts_flag: int = 0
    customer_support_contacts: int = 0
    previous_dispute_count: int = 0
    wishlist_to_cart_time_hrs: float = 24.0
    review_left_after_return: int = 0


class SubscribeRequest(BaseModel):
    plan: str = Field("free", description="Plan id: free | starter | pro")
    email: Optional[str] = None


# ----------------------------- auth -----------------------------
def require_key(x_api_key: str | None = Header(default=None)):
    if not x_api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED,
                            "Missing X-API-Key header.")
    row = store.get_key(x_api_key)
    if row is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid API key.")
    plan = get_plan(row["plan_id"])

    # per-minute rate limit
    now = time.time()
    bucket = _rate_buckets[x_api_key]
    while bucket and now - bucket[0] > 60:
        bucket.popleft()
    if len(bucket) >= plan.rate_per_min:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                            f"Rate limit: {plan.rate_per_min} req/min for {plan.name}.")
    bucket.append(now)

    # monthly quota
    used = store.usage_this_period(x_api_key)
    if used >= plan.monthly_quota:
        raise HTTPException(status.HTTP_402_PAYMENT_REQUIRED,
                            f"Monthly quota of {plan.monthly_quota} reached for "
                            f"{plan.name}. Upgrade your plan.")
    return x_api_key


# ----------------------------- routes -----------------------------
@app.get("/")
def root():
    return {"service": "ReturnGuard API", "version": "1.0.0",
            "docs": "/docs", "health": "/v1/health"}


@app.get("/v1/health")
def health():
    return {"status": "ok", "model_loaded": _engine is not None}


@app.get("/v1/plans")
def list_plans():
    return {
        "plans": [
            {"id": p.id, "name": p.name, "monthly_quota": p.monthly_quota,
             "price_usd": p.price_usd, "rate_per_min": p.rate_per_min,
             "description": p.description}
            for p in PLANS.values()
        ]
    }


@app.post("/v1/subscribe")
def subscribe(req: SubscribeRequest):
    if req.plan not in PLANS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            f"Unknown plan '{req.plan}'. Choose: {list(PLANS)}")
    plan = get_plan(req.plan)
    # Payment is stubbed. Paid plans are issued free during the demo period.
    issued = store.issue_key(plan.id, req.email)
    return {
        **issued,
        "price_usd": plan.price_usd,
        "payment": "stubbed (free during demo)" if plan.price_usd else "free",
        "message": f"Subscribed to {plan.name}. Use the api_key in the "
                   f"X-API-Key header.",
    }


@app.get("/v1/usage")
def usage(api_key: str = Depends(require_key)):
    stats = store.key_stats(api_key)
    if stats is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Key not found.")
    return stats


@app.post("/v1/predict")
def predict(features: ReturnFeatures, api_key: str = Depends(require_key)):
    result = _engine.predict_return(features.model_dump())
    store.increment_usage(api_key, 1)
    # top_reasons tuples -> JSON-friendly list
    result["top_reasons"] = [{"feature": f, "shap_value": v}
                             for f, v in result["top_reasons"]]
    return result
