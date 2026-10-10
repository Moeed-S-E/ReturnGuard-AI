# ReturnGuard

Explainable AI dynamic return-policy engine. At return initiation it scores a
request into one of four classes (Legitimate, Policy Abuser, Fraudulent Return,
Wardrobing) and maps it to a policy action (auto-approve, verify tags, fee
deduction, fraud hold, or light verify).

This repo starts as a prototype Jupyter notebook. Once validated it is converted
to modular `.py` files (data_pipeline, train_model, policy_engine) and a
Streamlit two-sided app.

## Run the notebook

```bash
cd returnguard
pip install -r requirements.txt
python -m ipykernel install --user --name returnguard   # optional
jupyter lab   # or: jupyter notebook
```

Open `returnguard_prototype.ipynb` and run **Kernel -> Restart & Run All**.
It executes top to bottom with no internet access and saves artifacts to
`artifacts/model.joblib` and `artifacts/metrics.json`.

## Data

Primary dataset: `../datasets/ecommerce_return_abuse_dataset.csv`
(4-class target `abuse_type`). The two `Fraudulent_E-Commerce_Transaction_Data`
CSVs are binary fraud sets and are not used for training.

## Tests (optional property-based tests)

```bash
pytest returnguard/tests
```

## Git workflow

Per-task branching via the helper at the repo root:

```bash
bash ../setup_git.sh init            # one-time
bash ../setup_git.sh start 01-setup  # branch task/01-setup from dev
# ... implement ...
bash ../setup_git.sh finish 01-setup # merge into dev, delete branch
```

`main` holds specs and scaffolding; `dev` is the integration branch.
