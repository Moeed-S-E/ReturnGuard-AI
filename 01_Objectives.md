# Objectives: ReturnGuard

## Vision
Make returns free for honest shoppers and costly only for abusers, using explainable AI that decides at the moment a return is requested.

## Primary Objectives (must achieve at hackathon)
| # | Objective | Measure |
|---|-----------|---------|
| O1 | Classify each return into 4 classes (Legitimate, Policy Abuser, Fraudulent Return, Wardrobing) | Macro F1 >= 0.82 on held-out test split (stretch; report actual value honestly) |
| O2 | Turn predictions into one of 4 automated policy actions | Policy engine returns action + customer message + merchant note for every input |
| O3 | Explain every decision | SHAP waterfall + plain-English reasons for any selected transaction |
| O4 | Two-sided working demo | Streamlit app with Customer Portal and Merchant Dashboard |
| O5 | Reliable live demo | 4 pre-built profiles (Honest, Wardrober, Abuser, Fraudster) produce expected outcomes every time |

## Secondary Objectives (nice to have)
- O6: Business impact panel (fraud value blocked, handling cost saved, CO2 avoided) with clearly labeled assumptions.
- O7: Manual merchant override with audit note.
- O8: Inference latency under 100 ms per sample, shown in the UI.

## Pitch Objectives
- Judges understand the problem in 30 seconds.
- Judges see one honest approval and one flagged case live.
- Judges see WHY the AI decided (SHAP).
- Sustainability angle is explicit (avoided trips and packaging).

## Non-Objectives (out of scope)
- Real payment/refund processing, real courier integration.
- Computer-vision tag verification (photo upload is simulated).
- Production auth, multi-tenant support.
- Deep learning models.

## Guiding Principles
1. Fairness first: never auto-punish without an explanation.
2. Human in the loop: fraud class routes to a person, never auto-bans.
3. Demo reliability beats model sophistication.
4. State all assumptions (cost figures, CO2 factors) on screen.
