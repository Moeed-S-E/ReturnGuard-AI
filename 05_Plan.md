# Plan: ReturnGuard (6-hour build)

## Pre-Hackathon (tonight)
- [ ] Install Python 3.10+, create venv, `pip install -r requirements.txt`
- [ ] Download the Kaggle dataset; check column names
- [ ] Put the spec files in `.kiro/specs/`; test that the synthetic generator runs
- [ ] Charge laptops; save a backup copy of everything on USB

## Event Day Timeline (adjust to your actual window)
| Time | Phase | Output |
|------|-------|--------|
| 09:30-10:15 | Setup | Kiro open, env verified, repo created |
| 10:15-11:30 | Data + model | data_pipeline.py, train_model.py, baseline metrics |
| 11:30-13:30 | Policy + SHAP | policy_engine.py, boundary tests, SHAP working |
| 13:30-14:30 | Lunch + checkpoint | Pipeline stable, notes on gaps |
| 14:30-16:15 | UI | app.py with both views, demo profiles |
| 16:15-17:15 | Hardening + deck | Profiles verified, 5-slide deck, 2 dry runs |
| 17:15-18:30 | Pitch | Live demo, Q&A |

## Roles
| Role | Owns |
|------|------|
| Backend/Kiro Lead | Spec prompts, pipeline, policy_engine |
| ML Specialist | Training, tuning, SHAP, thresholds |
| Frontend Dev | Streamlit UI, banners, dashboard |
| Product/Pitch Lead | Deck, script, timing, Q&A, assumptions slide |

## Milestone Gates
- **G1 (11:30):** Model trained and saved, metrics printed.
- **G2 (13:30):** `predict_return` works for all 4 profiles with SHAP.
- **G3 (16:15):** Full app runs end to end.
- **G4 (17:00):** Feature freeze. Only bug fixes and rehearsal after this.

## Cut List (if behind schedule, drop in this order)
1. CO2 panel detail -> single number
2. Manual override
3. Filters on table
4. Probability bar chart
Never cut: 4-class prediction, policy banners, one SHAP plot.

## Contingency
- Model fails: use the synthetic dataset with a clean generator.
- Kiro slow: write code manually from the TRD.
- SHAP breaks: show feature importance bar + reason strings as a temporary fallback.
- Live app crashes: have a screen recording of the demo ready.

## Pitch Script (3 min)
1. 0:00-0:40 Hook: returns cost, fees alienate honest buyers. 
2. 0:40-1:40 Live demo: Fatima (green) then wardrober (yellow).
3. 1:40-2:20 Merchant dashboard + SHAP "why".
4. 2:20-2:45 Sustainability + impact (state assumptions).
5. 2:45-3:00 Business model: Shopify/WooCommerce plugin or Lambda microservice. Close.

## Likely Judge Questions
- Is it fair/biased? -> no sensitive attributes, explanations, human review for fraud.
- Data realism? -> trained on a public benchmark dataset; production needs merchant data and retraining.
- Accuracy? -> quote measured Macro F1 and confusion matrix, not targets.
