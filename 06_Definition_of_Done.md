# Definition of Done: ReturnGuard

## Feature-level DoD
A task is done when: code runs without errors, is committed, has at least one test or manual check, and meets its acceptance criteria below.

## Acceptance Criteria
### Data and Model
- [ ] Pipeline runs with real CSV and with synthetic fallback
- [ ] Features match TRD; no leakage (category risk from train split only; customer_id excluded)
- [ ] Model trained, `artifacts/model.joblib` and `metrics.json` saved
- [ ] Macro F1, per-class precision/recall, confusion matrix reported (actual numbers, even if below target)

### Policy Engine
- [ ] Returns all fields in TRD section 7
- [ ] Threshold boundary tests pass (0.49/0.50, 0.59/0.60, 0.54/0.55, 0.74/0.75)
- [ ] Latency < 100 ms measured and displayed

### Explainability
- [ ] SHAP waterfall renders for the predicted class
- [ ] Top-3 reasons shown in plain English

### UI
- [ ] Customer Portal shows correct banner for each of 4 profiles
- [ ] Merchant Dashboard shows 4 KPI banners, table, SHAP plot
- [ ] Assumptions panel visible
- [ ] No console errors during a full demo run

### Demo
- [ ] 4 profiles verified: A=Green, B=Yellow, C=Orange, D=Red
- [ ] Full demo rehearsed twice within 3 minutes
- [ ] Backup screen recording saved
- [ ] App runs offline

### Deliverables
- [ ] README with run instructions
- [ ] 5-slide deck
- [ ] Code pushed to repo
- [ ] Spec files (requirements, design, tasks) included to show the Kiro workflow

## Project-level "Done"
All boxes above checked, OR remaining unchecked items are only from the Cut List in the Plan and the team has agreed to drop them.

## Quality Gates
- Nothing hardcoded in the model path that bypasses prediction, except an explicitly labeled "demo mode".
- All stated stats in the deck have a source or are labeled as assumptions.
