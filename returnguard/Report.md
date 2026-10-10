# ReturnGuard — Project Report

> Explainable AI dynamic return-policy engine. Honest shoppers get frictionless
> free returns; wardrobing, policy abuse, and fraud are routed to the right
> intervention — with a reason for every decision.

Built with spec-driven development (Kiro). Event: Islamabad Hack 2026.

---

## 1. Problem (slide 1)

- E-commerce return rates average ~19% of orders; handling a single return costs
  an estimated $21–$46.
- Return fraud and wardrobing are estimated at ~$101B/year industry-wide.
- Blanket return fees punish the ~70% of customers who return honestly and raise
  cart abandonment.
- Returns also add transport emissions and packaging waste.

*Statistics are industry estimates from project research; cite as such.*

## 2. Solution (slide 2)

Score each return **at initiation** with a LightGBM 4-class model, then map the
prediction to one of five graded actions — not a blunt allow/block:

| Class | Action | Customer experience |
|-------|--------|---------------------|
| Legitimate | AUTO_APPROVE | Green — instant free approval |
| Wardrobing | VERIFY_TAGS | Yellow — tag/photo check |
| Policy Abuser | FEE_DEDUCTION (PKR 950) | Orange — fee shown up front |
| Fraudulent Return | FRAUD_HOLD | Red — routed to human review |
| Uncertain | LIGHT_VERIFY | Yellow — light verification |

Severity-first evaluation order: **Fraud → Wardrobing → Abuser → Legitimate → Uncertain.**

## 3. Data

- Dataset: public **E-Commerce Return Abuse** benchmark, **60,000 rows**, 4-class
  target `abuse_type`.
- Class mix (actual): Legitimate ~70%, Policy Abuser ~12%, Fraudulent ~10%,
  Wardrobing ~8%.
- Split: stratified **70% train / 10% validation / 20% test**, seed 42.
- Leakage guard: `category_risk_index` computed from **training rows only** and
  mapped to validation/test.
- Excluded from features: `customer_id`, `order_id`, raw dates, and the target.
  No sensitive attributes used.

## 4. Model & features (slide 3)

- **LightGBM** multiclass, `class_weight="balanced"`, native categorical support
  (no one-hot), early stopping on the validation set.
- 34 input features: 11 raw numeric, 10 boolean, 8 categorical, and **5 engineered**:
  - `order_to_return_days`, `return_ratio`, `risk_exposure`,
    `category_risk_index`, `dispute_frequency`.

### Measured results (test set, report honestly)

| Metric | Value |
|--------|-------|
| Macro F1 | **0.998** |
| F1 Legitimate | 1.000 |
| F1 Policy Abuser | 0.997 |
| F1 Fraudulent Return | 0.998 |
| F1 Wardrobing | 0.998 |
| Inference latency | **~45–70 ms** per request (target < 100 ms) |

**Honesty note for judges:** the near-perfect F1 reflects the dataset, not a
miracle model. The classes in this public synthetic benchmark are highly
separable — two raw columns (`return_rate_pct`, `days_to_return`) already reach
~0.94 macro F1 with a shallow tree. We report measured numbers and state that
production use requires retraining on real merchant data, where scores will be
lower and threshold tuning matters more.

## 5. Explainability (slide 4)

- `shap.TreeExplainer` cached once; one **waterfall plot per class** in the
  notebook, plus **top-3 plain-English reasons** on every live decision.
- Example reasons surfaced in the demo:
  - Honest → `return_ratio`, `customer_support_contacts`, `return_rate_pct`
  - Wardrober → `days_to_return`, `wishlist_to_cart_time_hrs`, `item_returned_opened`
  - Abuser → `return_rate_pct`, `is_high_value_item`, `total_returns_lifetime`
  - Fraudster → `days_to_return`, `wishlist_to_cart_time_hrs`, `return_rate_pct`

## 6. Product — two-sided demo (slide 5)

**Customer Portal:** pick a profile, submit, get a colored banner with the
resolution, the reason, and the latency.

**Merchant Dashboard:**
- KPIs: returns analyzed, flagged count, fees applied, auto-approved %.
- **Business-impact panel** (illustrative): value protected (PKR), handling cost
  saved ($30/return), CO₂ avoided (1.0 kg/trip) — all assumptions on screen.
- Filterable table of analyzed returns.
- SHAP reasons + probability bar per selected return.
- **Manual override** with a required audit note (human-in-the-loop).

Verified: 4 demo profiles produce the expected action every time; on a random
sample of 30 real rows, the policy action agreed with the true label 29/30.

## 7. Architecture

```
datasets/ecommerce_return_abuse_dataset.csv
        │
        ▼
returnguard_prototype.ipynb   (validated prototype: EDA → features → train →
        │                       tune → SHAP → policy → demo)
        ▼
train_model.py  ──►  artifacts/model.joblib, metrics.json, policy_config.json
        │
        ▼
policy_engine.py  (PolicyEngine.predict_return — model-aligned, cached SHAP)
        │
        ▼
app.py  (Streamlit: Customer Portal | Merchant Dashboard)
```

## 8. Responsible AI

- Fraud class goes to **human review**, never auto-banned.
- Customer messages are **neutral** — they state next steps, never accuse.
- **No sensitive attributes** in features.
- Merchant override is logged with an audit note.
- Known limitation: public dataset may not match a real merchant's population;
  monitor false-positive rate on honest users and retrain before deployment.

## 9. Business model

SaaS plugin for Shopify/WooCommerce, or an API on AWS Lambda. Priced per scored
return or as a share of recovered value.

## 10. How to run

```bash
cd returnguard
pip install -r requirements.txt           # or: uv pip install -r requirements.txt
python train_model.py                     # produces artifacts/
pytest tests                              # 21 property + integration tests
streamlit run app.py                      # launch the two-view app
```

Notebook: open `returnguard_prototype.ipynb` → Restart & Run All (uses the
`returnguard-venv` kernel). No internet required at runtime.

## 11. Suggested slide order (3-min pitch)

1. Hook — returns cost money; blanket fees alienate honest buyers.
2. Solution — four graded, explained interventions.
3. Live demo — Fatima (green) then Wardrober (yellow).
4. Merchant dashboard + SHAP "why" + impact panel.
5. Honest results, responsible AI, business model. Close.

## 12. Likely judge questions

- *Is it fair/biased?* No sensitive attributes; explanations on every decision;
  human review for fraud.
- *Why is F1 so high?* The public benchmark is highly separable; we report
  measured numbers and note real data will be harder.
- *Latency?* ~45–70 ms per request, well under 100 ms.
- *Data realism?* Trained on a public benchmark; production needs merchant data
  and retraining.
