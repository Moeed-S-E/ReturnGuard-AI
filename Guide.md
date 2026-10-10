# Guide: Building and Running ReturnGuard

## 1. Setup
```bash
mkdir returnguard && cd returnguard
python -m venv venv
# Windows: venv\Scripts\activate
source venv/bin/activate
pip install streamlit lightgbm shap pandas numpy scikit-learn matplotlib joblib
pip freeze > requirements.txt
```

## 2. Data
Download the Kaggle dataset, save as `data/returns.csv`. Print `df.columns` and update the column map in `data_pipeline.py`. If missing, the pipeline generates synthetic data.

## 3. Using Kiro
1. Open the folder in Kiro, switch to Spec mode.
2. Create `.kiro/specs/returnguard/` with `requirements.md` (from PRD), `design.md` (from TRD + Design), `tasks.md`.
3. Suggested tasks.md:
```
- [ ] 1 data_pipeline.py: load CSV or synthesize 5,000 rows, config column map
- [ ] 2 Feature engineering per TRD section 5
- [ ] 3 train_model.py: LightGBM multiclass, balanced, metrics.json, model.joblib
- [ ] 4 policy_engine.py: predict_return, POLICY_CONFIG, SHAP
- [ ] 5 tests/test_policy.py: threshold boundaries
- [ ] 6 demo_profiles.py: 4 profiles
- [ ] 7 app.py: Customer Portal + Merchant Dashboard
```
4. Prompt: "Read the spec files here and execute tasks 1 to 7 in order. Run tests after each task. If the dataset is missing, synthesize it. Do not skip SHAP. Report metrics."
5. Review generated code yourself; do not demo code nobody on the team understands.

## 4. Run
```bash
python train_model.py
pytest tests
streamlit run app.py
```

## 5. Tuning thresholds
On the validation set, sweep each threshold and pick the value that keeps honest-user false positives low while meeting recall goals. Record the final values in `POLICY_CONFIG`.

## 6. Common Problems
| Problem | Fix |
|---------|-----|
| SHAP returns a list vs 3-D array | Check `type(shap_values)`; index by predicted class accordingly |
| Streamlit re-trains on every click | Use `st.cache_resource` for model/explainer |
| Waterfall not showing | Call `plt.gcf()` and `st.pyplot(fig)`; use `show=False` |
| Demo profile gives wrong banner | Adjust input features, re-test, never fake the model output |
| Unicode/PKR symbol issues | Use "PKR" text |

## 7. Pre-Pitch Checklist
- [ ] 4 profiles verified
- [ ] Metrics screenshot ready
- [ ] Assumptions slide ready
- [ ] Backup recording
- [ ] Wi-Fi not required
- [ ] Each teammate can explain one part of the code

## 8. Honest Claims
Say "targeted" and "measured on a public benchmark". Avoid claiming real-world savings. State that the figures for CO2 and handling cost are assumptions.
