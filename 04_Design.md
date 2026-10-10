# Design: ReturnGuard

## 1. Design Principles
Frictionless for honest users, transparent for everyone, merchant sees the "why".

## 2. Visual Language
| Banner | Color | Meaning |
|--------|-------|---------|
| Green | success | Auto-approved |
| Yellow | warning | Verification required |
| Orange | notice | Fee applies |
| Red | alert | Under review |
Currency PKR. Mobile-friendly single column for the customer view.

## 3. View 1: Customer Return Portal
```
+------------------------------------------+
| [Logo]  Fast Return & Exchange Center    |
+------------------------------------------+
| Demo profile: [ Fatima - Casual Shirt v ]|
| Order ID:     [ #1001               v ]  |
| Item:         [ Casual Cotton Tee 2,200 ]|
| Reason:       [ Doesn't fit         v ]  |
|           [ Submit Return Request ]      |
+------------------------------------------+
| [ BANNER: resolution + next steps ]      |
| (Green: QR code image) (Yellow: upload   |
|  tag photo / store finder) (Orange: fee  |
|  breakdown) (Red: review timeline)       |
+------------------------------------------+
```
Copy:
- Green: "Your return is approved! Show this QR code to the courier. Full refund on handoff."
- Yellow: "Quick check needed: drop the item at a store with tags attached, or upload a photo of the tag."
- Orange: "Return authorized. A PKR 950 handling and restocking fee will be deducted from your refund."
- Red: "Your request is under review. Our team will contact you within 24 hours."

## 4. View 2: Merchant Dashboard
```
+------------------------------------------------------------+
| [Returns analyzed] [Value protected] [CO2 avoided] [Honest |
|                                                  approval%]|
+------------------------------------------------------------+
| Filters: class [all v]  action [all v]                     |
| Table: order | customer | category | price | class | action |
+------------------------------------------------------------+
| Selected: order #2042                                      |
| [ SHAP waterfall ]   | Plain-English reasons (top 3)       |
| Probabilities bar    | [ Manual Override ] note: [______ ] |
+------------------------------------------------------------+
| Assumptions (expandable): $30 handling, CO2 factor, thresholds
+------------------------------------------------------------+
```

## 5. Demo Profiles
| Profile | Order | Category | Delta days | Return rate | Other | Expected |
|---------|-------|----------|-----------|-------------|-------|----------|
| A Honest | #1001 | Men's Casual Shirt | 7 | 4% | 0 disputes | Green |
| B Wardrober | #2042 | Luxury Evening Gown | 1 | 72% | high price | Yellow |
| C Abuser | #3089 | Shoes | 28 | 91% | many returns | Orange (PKR 950) |
| D Fraud | #4401 | Smart Watch | 0 | moderate | 4 disputes, address mismatch | Red |
Profiles must be tuned against the trained model before the pitch; if a profile misses, adjust feature values (not the model) and re-verify.

## 6. Accessibility
Banners carry text and an icon, not color alone. Minimum 16 px body text.

## 7. Error States
Missing model: app shows "Training model..." and triggers training. Invalid input: inline message.
