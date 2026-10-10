# Dec.md — ReturnGuard ke Decisions (Roman Urdu)

Is file mein saari badi decisions likhi hain jo project banate waqt li gayin,
aur har decision ke peeche ki wajah bhi di gayi hai. Simple Roman Urdu mein.

---

## 1. Notebook pehle, phir modular .py files
**Decision:** Pehle `returnguard_prototype.ipynb` banaya (poora pipeline ek
notebook mein), jab wo chal gaya to usko modular Python files (`train_model.py`,
`policy_engine.py`, `app.py`) mein convert kiya.
**Wajah:** Aap ne kaha tha ke pehle notebook mein kaam kar ke dekho, agar chal
jaye to behtar OOP/modular code likho. Notebook se jaldi results dikhte hain aur
debugging aasan hoti hai. Baad mein clean code production jaisa ban gaya.

## 2. Dataset ka chunao — konsa CSV use kiya
**Decision:** Teen CSV mein se sirf `ecommerce_return_abuse_dataset.csv` use kiya
(60,000 rows, 4 classes). Baaki do (`Fraudulent_E-Commerce_Transaction_Data`)
istemaal nahi kiye.
**Wajah:** Sirf yehi dataset 4-class `abuse_type` target rakhta hai jo humein
chahiye tha. Baaki do sirf binary fraud (haan/nahi) wale thay, humare masle ke
liye theek nahi.

## 3. Model — LightGBM
**Decision:** LightGBM multiclass model, `class_weight="balanced"`, native
categorical support (one-hot nahi), early stopping ke sath.
**Wajah:** Tree model fast hai, categorical columns seedha handle karta hai,
scaling ki zaroorat nahi, aur SHAP ke sath acchi explainability deta hai. Deep
learning scope se bahar thi (project ke rules mein mana tha).

## 4. Macro F1 bohat zyada (0.998) — isko chupaya nahi, sach bola
**Decision:** F1 0.998 aaya. Humne ye nahi dikhaya ke "wah humara model kamaal
hai". Report mein saaf likha ke dataset hi aasan hai (sirf do column se 0.94 F1
aa jata hai).
**Wajah:** Judges ke saamne honesty zaroori hai. Project ke Objectives doc mein
bhi likha tha ke measured results sach bolo. Real merchant data par score kam
hoga — ye baat clear ki.

## 5. Data leakage se bachao
**Decision:** `category_risk_index` sirf TRAIN split se calculate kiya, phir
validation/test par map kiya. `customer_id`, `order_id` aur dates ko feature
nahi banaya.
**Wajah:** Agar poore data se risk index banate to test par cheating ho jati
(leakage). Isliye sirf train se banaya taake score honest rahe.

## 6. Policy decision — severity order
**Decision:** Faisla is tarteeb mein hota hai: pehle Fraud, phir Wardrobing,
phir Policy Abuser, phir Legitimate, warna LIGHT_VERIFY.
**Wajah:** Sabse khatarnaak case pehle pakadna zaroori hai. Fraud ko kabhi auto
approve nahi karna chahiye, isliye wo sabse upar check hota hai.

## 7. Threshold tuning — fixed values nahi
**Decision:** Thresholds validation set par tune kiye (0.30 se 0.90 sweep), PRD
ki values sirf starting point thin.
**Wajah:** Har class ka best threshold data se nikalna behtar hai banisbat haath
se fixed number dene ke. Recall maximize kiya minority classes ke liye.

## 8. Demo profiles ki values adjust keen (model nahi)
**Decision:** 4 demo profiles (Honest, Wardrober, Abuser, Fraudster) ki feature
values ko dataset ke real patterns ke mutabiq set kiya taake sahi action aaye.
Model ko haath nahi lagaya.
**Wajah:** Design doc mein likha tha ke demo fail ho to profile ki values badlo,
model ya output fake mat karo. Pehle 2/4 fail ho rahe thay, calibrate kar ke
4/4 pass ho gaye.

## 9. SHAP explainability
**Decision:** Har decision ke sath top-3 reasons (plain English) aur notebook
mein har class ka ek waterfall plot. Explainer sirf ek dafa cache kiya.
**Wajah:** O3 objective tha ke har faisla explain ho. Cache isliye taake app
fast rahe (har request par naya explainer na bane).

## 10. Streamlit UI (React nahi)
**Decision:** Frontend ke liye Streamlit chuna, React nahi. Do views: Customer
Portal aur Merchant Dashboard.
**Wajah:** Aap ne kaha tha agar zaroorat ho to React, warna Streamlit. 6-ghante
ke hackathon mein Streamlit se do-tarfa demo bohat jaldi ban jata hai, Python
hi mein, alag frontend setup ki zaroorat nahi.

## 11. Business impact panel + manual override (MVP)
**Decision:** Merchant dashboard mein value protected, handling cost saved, aur
CO2 avoided ka panel, aur manual override jisme audit note zaroori hai.
**Wajah:** Ye secondary objectives (O6, O7) thay. Override mein human-in-the-loop
rakha taake koi bhi faisla insan badal sake, record ke sath.

## 12. Git — har task alag branch, phir dev, phir main
**Decision:** `main` + `dev` branches, kaam `dev` par, phir `main` mein merge
kar ke GitHub push.
**Wajah:** Aap ne branch-per-task aur looping maangi thi. Isse history saaf
rehti hai aur har step track hota hai.

## 13. Datasets aur planning docs ko gitignore kiya
**Decision:** `datasets/` (375 MB wali CSV) aur 01–07 planning docs ko git se
bahar rakha.
**Wajah:** GitHub ki 100 MB limit hai, badi CSV push nahi ho sakti thi (push
hang ho raha tha). Planning docs aap ne khud bola tha ke publish mat karo.

## 14. model.joblib ko commit kiya (deployment ke liye)
**Decision:** Pehle `*.joblib` gitignore mein tha, lekin baad mein
`model.joblib` (3 MB) ko force-add kar ke commit kiya.
**Wajah:** Streamlit Cloud par model file chahiye warna app crash karti ("Model
not found"). 3 MB chhoti hai, GitHub limit ke andar, isliye commit karna safe
tha. cat_risk_map ko `policy_config.json` mein daala taake runtime par dataset
ki zaroorat na rahe.

## 15. Requirements versions pin keen
**Decision:** `lightgbm`, `shap`, `scikit-learn`, `numpy`, `joblib` ki exact
versions pin keen jo model train karne ke waqt thin.
**Wajah:** Agar Cloud alag version install karta to `joblib.load` fail ya warning
de sakta tha. Exact match se model safely load hota hai.

## 16. Tests — property-based + boundary + integration
**Decision:** 31 tests likhe: formula properties (Hypothesis se), threshold
boundary tests (0.49/0.50 waghera), aur 4 demo profile integration tests.
`conftest.py` + `pytest.ini` add kiya taake import kahin se bhi chale.
**Wajah:** DoD mein tests maange thay. Boundary tests isliye ke threshold ke
bilkul kinare par faisla sahi ho. conftest isliye ke `policy_engine` import kisi
bhi folder se chale (pehle error aa raha tha).

---

## Khulasa (Summary)
- Pehle notebook, phir clean modular code.
- Sahi dataset chuna, leakage se bacha, honest results bole.
- Explainability (SHAP) aur fairness (fraud par insan) rakhi.
- Streamlit se do-tarfa demo, impact panel aur override ke sath.
- Git clean rakha, deployment ke liye model aur pinned versions commit keen.
- 31 tests pass, app dono views render karti hai — deployment ready hai.
