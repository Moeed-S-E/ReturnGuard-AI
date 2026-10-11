# ReturnGuard API (backend)

FastAPI service exposing the ReturnGuard model as a subscription REST API.

## Run

    cd returnguard
    pip install -r requirements.txt -r backend/requirements.txt
    bash run_api.sh
    # interactive docs are served at /docs on the running service

## Auth and subscriptions

Pass your key in the `X-API-Key` header. Get a key by subscribing to a plan
(payment is stubbed, free during the demo):

    POST /v1/subscribe   body: {"plan":"starter","email":"you@store.com"}
    -> {"api_key":"rg_...","plan":"starter","monthly_quota":10000}

Plans (see backend/plans.py): free (500/mo, $0), starter (10k/mo, $49),
pro (100k/mo, $299). A monthly quota and a per-minute rate limit are enforced
per key.

## Endpoints

| Method | Path          | Auth | Purpose |
|--------|---------------|------|---------|
| GET    | /v1/health    | no   | health check |
| GET    | /v1/plans     | no   | list plans |
| POST   | /v1/subscribe | no   | issue an API key for a plan |
| POST   | /v1/predict   | key  | score a return request |
| GET    | /v1/usage     | key  | quota and usage for your key |

Predict response: probs, predicted_class, action, fee_pkr, customer_message,
merchant_note, banner, top_reasons (SHAP), latency_ms.

## Storage

API keys and monthly usage live in a local SQLite file
(backend/returnguard_api.db), created automatically on first run.

## Payments (future)

/v1/subscribe is the integration point for a real gateway (e.g. Stripe).
Today it issues keys for free; swap in a checkout + webhook to charge the
price_usd from plans.py before issuing the key.
