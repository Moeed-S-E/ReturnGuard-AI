"""Subscription plans for the ReturnGuard API.

Each plan defines a monthly request quota and a price. "Purchasing" a plan
issues an API key bound to that plan. Payments are stubbed (free for now);
the price fields are shown in the UI and are the integration point for a real
gateway (e.g. Stripe) later.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    id: str
    name: str
    monthly_quota: int          # requests per calendar month
    price_usd: float            # 0 = free
    rate_per_min: int           # simple per-minute burst limit
    description: str


PLANS = {
    "free": Plan(
        id="free", name="Free", monthly_quota=500, price_usd=0.0,
        rate_per_min=30,
        description="For evaluation and small projects. 500 scored returns/month.",
    ),
    "starter": Plan(
        id="starter", name="Starter", monthly_quota=10_000, price_usd=49.0,
        rate_per_min=120,
        description="For growing stores. 10,000 scored returns/month.",
    ),
    "pro": Plan(
        id="pro", name="Pro", monthly_quota=100_000, price_usd=299.0,
        rate_per_min=600,
        description="For high-volume merchants. 100,000 scored returns/month.",
    ),
}

DEFAULT_PLAN = "free"


def get_plan(plan_id: str) -> Plan:
    return PLANS.get(plan_id, PLANS[DEFAULT_PLAN])
