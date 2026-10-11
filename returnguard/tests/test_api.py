"""Integration tests for the ReturnGuard FastAPI backend.

Skipped if the trained model artifact is missing.
"""
from pathlib import Path
import pytest

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
pytestmark = pytest.mark.skipif(
    not (ARTIFACTS / "model.joblib").exists(),
    reason="trained model.joblib not present; run train_model.py first",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from backend.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/v1/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_plans_listed(client):
    ids = [p["id"] for p in client.get("/v1/plans").json()["plans"]]
    assert set(ids) == {"free", "starter", "pro"}


def test_predict_requires_key(client):
    assert client.post("/v1/predict", json={}).status_code == 401


def test_subscribe_and_predict(client):
    sub = client.post("/v1/subscribe", json={"plan": "free"}).json()
    key = sub["api_key"]
    assert sub["monthly_quota"] == 500
    r = client.post("/v1/predict", json={"product_category": "Clothing"},
                    headers={"X-API-Key": key})
    assert r.status_code == 200
    body = r.json()
    assert body["action"] in {"AUTO_APPROVE", "VERIFY_TAGS", "FEE_DEDUCTION",
                              "FRAUD_HOLD", "LIGHT_VERIFY"}
    assert len(body["top_reasons"]) == 3
    usage = client.get("/v1/usage", headers={"X-API-Key": key}).json()
    assert usage["used_this_month"] >= 1


def test_invalid_key_rejected(client):
    assert client.post("/v1/predict", json={"product_category": "Clothing"},
                       headers={"X-API-Key": "bad"}).status_code == 401
