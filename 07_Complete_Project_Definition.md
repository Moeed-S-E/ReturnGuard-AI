# Complete Project Definition: ReturnGuard

## One-liner
An explainable AI policy engine that gives honest shoppers frictionless free returns and routes wardrobing, policy abuse, and fraud to the right intervention.

## Context
- Event: Islamabad Hack 2026, BIIT Rawalpindi (6-hour build)
- Tracks: retail/e-commerce, sustainability, AI for business
- Dev environment: AWS Kiro (spec-driven)

## Problem
Return costs and return fraud erode e-commerce margins, while blanket fees punish the majority who return honestly. Retailers need targeted, explainable policies.

## Solution
Score each return at initiation with a LightGBM 4-class model. Map class probabilities to four actions. Show customers a clear resolution, and show merchants SHAP-based reasons and business impact.

## Users
Honest shopper (Fatima), merchant risk lead (Bilal).

## Scope
**In:** data pipeline, model, policy engine, SHAP, two-view Streamlit app, demo profiles, impact estimates.
**Out:** real payments, courier APIs, image verification, auth, production deployment.

## Data
Kaggle E-Commerce Return Abuse Detection (about 60k rows, 4 classes) with synthetic fallback. Verify schema on download.

## Model
LightGBM multiclass, balanced class weights, stratified split, Macro F1 target 0.82 (stretch).

## Policy
Fraud hold, wardrobing verification, abuser fee (PKR 950), legitimate auto-approval, uncertain light verification. Severity-first evaluation with tunable thresholds.

## Differentiators
1. Four graded interventions, not a block/allow binary
2. Per-decision explainability (SHAP)
3. Two-sided product (customer + merchant)
4. Sustainability metric (avoided trips, packaging)
5. Built with spec-driven development in Kiro

## Responsible AI
- No sensitive attributes in features
- Fraud class goes to human review, never automatic punishment
- Customer messages are neutral
- Merchant override with audit note
- Known limitation: public dataset may not reflect a local merchant's population; retrain on real data before deployment
- Monitor false-positive rate on honest users

## Business Model
SaaS plugin for Shopify/WooCommerce or API on AWS Lambda; priced per scored return or as a share of recovered value.

## Success Criteria
See Objectives and Definition of Done.

## Documents
01_Objectives, 02_PRD, 03_TRD, 04_Design, 05_Plan, 06_Definition_of_Done, 07_Complete_Project_Definition, Guide.
