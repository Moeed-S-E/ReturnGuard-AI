# Product Requirements Document (PRD): ReturnGuard

## 1. Summary
ReturnGuard is an explainable AI dynamic return-policy engine. At return initiation it scores the request and picks one of four interventions: instant free approval, verification required, fee deduction, or fraud hold.

## 2. Problem
- E-commerce return rates average roughly 19.3% (figure from project research).
- Handling cost per return is cited at $21-$46; return fraud and wardrobing at about $101B/year.
- Blanket return fees hurt honest customers (about 70% of returns in the dataset) and increase cart abandonment.
- Returned goods add transport emissions and packaging waste.

*Note: statistics come from the research documents supplied; cite sources in the pitch or label them "industry estimates".*

## 3. Personas
1. **Fatima, Honest Shopper**: occasional returns for fit. Wants fast, free, no-questions return.
2. **Bilal, Risk Ops Lead (Merchant)**: wants to cut margin loss without raising churn or support load. Needs to know why a return was flagged.

## 4. User Stories
| ID | Story | Priority |
|----|-------|----------|
| US1 | As Fatima, I submit a return and get instant approval with a QR code | Must |
| US2 | As a customer flagged for wardrobing, I get clear instructions (store drop-off or tag photo) | Must |
| US3 | As a policy abuser, I see the fee and the reason before accepting | Must |
| US4 | As a suspected fraud case, I see a neutral "under review" message | Must |
| US5 | As Bilal, I see all analyzed returns with risk class and action | Must |
| US6 | As Bilal, I open a SHAP waterfall for any return | Must |
| US7 | As Bilal, I override a decision with a note | Should |
| US8 | As Bilal, I see money protected and CO2 avoided | Should |

## 5. Policy Decision Matrix
| Class | Trigger | Action | Customer Experience |
|-------|---------|--------|---------------------|
| Legitimate | P(Legit) >= 0.75 | Auto-approve, free QR/pickup | Green banner |
| Wardrobing | P(Wardrobing) >= 0.60 | Conditional: in-store tag check or tag photo | Yellow banner |
| Policy Abuser | P(Abuser) >= 0.55 | Approve with PKR 950 handling/restocking deduction | Orange banner |
| Fraudulent | P(Fraud) >= 0.50 | Hold label, send to manual review queue | Red banner |
| Uncertain (no rule fires) | none above | Approve with lightweight verification (tag photo) | Yellow-light banner |

**Evaluation order (severity first):** Fraud, Wardrobing, Abuser, Legitimate, Uncertain. Thresholds are starting values and must be tuned on the validation set to meet KPIs.

## 6. KPIs
| KPI | Target |
|-----|--------|
| Recall on Fraud + Wardrobing combined | >= 80% (stretch) |
| Honest shoppers getting zero-fee approval | >= 70% (stretch) |
| Macro F1 | >= 0.82 (stretch) |
| Inference latency | < 100 ms |
| Avoided CO2 / packaging | Computed from prevented consignments, assumptions displayed |

Targets are goals, not guarantees. Report measured results.

## 7. Functional Requirements
- FR1 Load dataset CSV, or synthesize data if missing (see TRD).
- FR2 Engineer features and train LightGBM multiclass model.
- FR3 Serve prediction + probabilities + policy action.
- FR4 Compute SHAP values per prediction.
- FR5 Customer Portal view (order, product, reason, submit, resolution banner).
- FR6 Merchant Dashboard view (KPI banners, transaction table, SHAP plot, override).
- FR7 Four preset demo profiles selectable from dropdown.

## 8. Non-Functional Requirements
- Latency < 100 ms; app starts in < 30 s using a cached model.
- Runs on a laptop CPU, no internet needed during demo.
- Modular code, readable by judges.
- Customer-facing text never accuses; it states next steps.

## 9. Risks
| Risk | Mitigation |
|------|-----------|
| Dataset download fails / venue Wi-Fi | Download tonight; synthetic fallback generator |
| Model misses target F1 | Report honestly; tune thresholds; show confusion matrix |
| Model gives unexpected result on demo profile | Preset profiles verified; hardcoded fallback outputs flagged as "demo mode" |
| SHAP multiclass shape errors | Test early; pin shap version |
| Fairness concerns | Exclude sensitive attributes; human review for fraud class |
