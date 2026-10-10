<div align="center">
  <h1><img src="https://fonts.gstatic.com/s/i/short-term/release/materialsymbolsoutlined/security/default/48px.svg" alt="Shield" style="vertical-align: middle; margin-right: 10px;">ReturnGuard</h1>
  <p><strong>Explainable AI Return-Policy Engine</strong></p>
</div>

## Overview
ReturnGuard is an end-to-end Explainable AI (XAI) pipeline and modern web dashboard designed to combat e-commerce return abuse. It classifies return requests into four severity categories (Legitimate, Policy Abuser, Wardrobing, Fraudulent Return) and automatically enforces policy actions based on dynamically tuned risk thresholds.

It provides two distinct views in a beautifully designed Streamlit UI:
1. **Customer Portal**: A front-end simulator where customers request returns and get instant, AI-explained decisions.
2. **Merchant Dashboard**: A risk-analyst workbench for batch-scoring returns, reviewing business impact (KPIs), inspecting AI reasoning via SHAP values, and applying manual overrides with audit logging.

## Features
* **LightGBM Multiclass Model**: Highly optimized model utilizing native categorical features.
* **Auto-Tuned Thresholds**: Decision boundaries are dynamically optimized on a held-out validation set for maximum recall of minority fraud classes.
* **SHAP Explainability**: Fully transparent AI. Every decision comes with a SHAP explanation detailing exactly *why* the AI made its choice.
* **Zero-Knowledge UI**: Completely interactive, HTML-styled interface with omnipresent tooltips and Google Material Symbols, ensuring non-technical users can interpret AI outputs effortlessly.

## Project Structure
* `returnguard_prototype.ipynb`: The canonical Jupyter notebook containing the entire data pipeline, EDA, feature engineering, model training, and threshold tuning.
* `app.py`: The modern, interactive Streamlit frontend application.
* `policy_engine.py`: The core inference module used by the app to score returns.
* `tests/`: Property-based correctness tests (Hypothesis/Pytest) verifying critical invariants of the data pipeline and policy engine.
* `artifacts/`: Contains the compiled model (`model.joblib`) and metrics (`metrics.json`).

## Setup & Installation

1. **Navigate** to the project root directory.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install streamlit
   ```
3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```
   *The application will automatically launch in your default web browser.*

## UI & Design
The Streamlit frontend has been heavily augmented with custom CSS injection to simulate a premium HTML web experience:
- **Streamlit Artifacts Hidden**: Default headers, footers, and hamburger menus are hidden.
- **Edge-to-Edge Layout**: Container padding is minimized for a modern, flush look.
- **Glassmorphism & Gradients**: Metrics and banners use blurred, semi-transparent backgrounds with deep radial gradients.
- **Micro-Animations**: Hover effects, smooth tab transitions, and artificial processing spinners.
