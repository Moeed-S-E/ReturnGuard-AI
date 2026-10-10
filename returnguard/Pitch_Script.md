# ReturnGuard — Slide-by-Slide Pitch Script

12 slides, ~3-minute pitch. Each slide: what to put on it, what to say
(speaker notes), and a timing cue. Keep stats labeled as estimates.

Total target: 3:00. Spend the most time on slides 3–7 (problem → live demo).

---

## Slide 1 — Title (0:00–0:15)

**On slide**
- 🛡️ **ReturnGuard**
- Tagline: *Free returns for honest shoppers. Costly only for abusers.*
- Team name, Islamabad Hack 2026, "Built with Kiro (spec-driven)"

**Say**
> "Hi, we're [team]. ReturnGuard is an explainable AI that decides — the moment
> a customer asks to return something — whether to approve it instantly, verify
> it, charge a fee, or flag it for review. Let me show you why that matters."

---

## Slide 2 — The Problem (0:15–0:40)

**On slide** (3 big numbers, labeled "industry estimates")
- ~19% of e-commerce orders get returned
- $21–$46 to handle a single return
- ~$101B/year lost to return fraud + wardrobing

**Say**
> "Returns are expensive. Handling one costs up to $46, and fraud and
> 'wardrobing' — wearing something once then returning it — cost the industry
> around a hundred billion a year. So retailers add blanket return fees. But that
> punishes the ~70% of people who return honestly, and drives them to competitors."

---

## Slide 3 — The Insight (0:40–0:55)

**On slide**
- A split image: "Blanket fee = honest customers leave" vs "No policy = fraud wins"
- Big text: **Returns aren't all the same. The policy shouldn't be either.**

**Say**
> "The core problem is that everyone gets treated the same. A loyal customer and
> a serial fraudster hit the exact same return button. We thought: what if the
> policy adapted to the person — instantly, and with a reason?"

---

## Slide 4 — Our Solution (0:55–1:15)

**On slide** — the 4-class → action table with color chips
| Class | Action | Banner |
|-------|--------|--------|
| Legitimate | Instant free approval | 🟢 Green |
| Wardrobing | Tag / photo check | 🟡 Yellow |
| Policy Abuser | Fee applied (shown up front) | 🟠 Orange |
| Fraudulent | Human review | 🔴 Red |

**Say**
> "ReturnGuard scores every return into four classes and maps it to a graded
> action — not just allow or block. Honest returns are approved free and instantly.
> Risky ones get the right friction. And fraud always goes to a human — never an
> automatic ban. Evaluation is severity-first: fraud is checked before anything else."

---

## Slide 5 — How It Works (1:15–1:35)

**On slide** — simple architecture arrow diagram
`Return request → LightGBM (4-class) → SHAP reasons → Policy action`
- 34 features, 5 engineered (return speed, lifetime return ratio, risk exposure…)
- No sensitive attributes used

**Say**
> "Under the hood: a LightGBM model trained on 60,000 returns, using 34 signals
> like how fast the item came back, the customer's lifetime return ratio, and
> dispute history. We deliberately exclude anything sensitive. Every prediction
> comes with a SHAP explanation — so we can always answer 'why'."

---

## Slide 6 — Live Demo: Honest Shopper (1:35–2:00)

**On slide** — screenshot of Customer Portal, green banner
**Do live:** pick "Fatima — Honest Shopper", click Submit → 🟢 green.

**Say**
> "Here's Fatima, a loyal customer returning a shirt that didn't fit. She submits…
> instant green approval, full refund, no questions — decided in about 50
> milliseconds. That's the experience that keeps honest customers loyal."

---

## Slide 7 — Live Demo: Wardrober / Fraudster (2:00–2:25)

**On slide** — yellow and red banners side by side
**Do live:** pick "Wardrober" → 🟡 yellow (verify tags). Then "Fraudster" → 🔴 red.

**Say**
> "Now a wardrobing case — high-value dress, worn, returned after 40 days. We
> don't accuse; we ask for a quick tag check — yellow. And a fraud case —
> 'item not received', fast high-value refund, prior disputes — goes to human
> review, red. Same button, three very different, fair outcomes."

---

## Slide 8 — Explainability (2:25–2:45)

**On slide** — a SHAP waterfall plot + the top-3 plain-English reasons
**Do live (optional):** open the Merchant Dashboard, show SHAP reasons for a return.

**Say**
> "This is the part judges and regulators care about. For every decision, the
> merchant sees the top reasons — here, 'days to return' and 'return rate' drove
> the flag. No black box. If a customer disputes it, there's an auditable reason."

---

## Slide 9 — Merchant Dashboard & Impact (2:45–3:00 … or extend)

**On slide** — dashboard screenshot: KPIs + impact panel
- Value protected, handling cost saved, CO₂ avoided (labeled estimates)
- Manual override with audit note

**Say**
> "Merchants get a dashboard: what's flagged, fees applied, and an illustrative
> impact panel — value protected, handling cost saved, even CO₂ avoided from
> fewer shipping trips. Every assumption is on screen. And a human can override
> any decision, with a logged note."

---

## Slide 10 — Results (honest)

**On slide**
- Macro F1 **0.998** on held-out test (⚠️ dataset is highly separable — see note)
- Latency **~45–70 ms** (< 100 ms target)
- 4/4 demo profiles correct; 29/30 on random real rows

**Say**
> "On the test set we measured a macro F1 of 0.998. We want to be honest: this
> public benchmark is unusually easy — two columns already get you to 0.94. The
> real value is the explainable, graded-policy framework. On real merchant data
> we'd expect lower scores and tuning would matter more."

---

## Slide 11 — Responsible AI

**On slide** (4 checkmarks)
- ✅ No sensitive attributes in features
- ✅ Fraud → human review, never auto-ban
- ✅ Neutral customer messaging (states next steps, never accuses)
- ✅ Auditable override + monitor false positives on honest users

**Say**
> "Fairness is built in: no sensitive attributes, humans in the loop for fraud,
> neutral language, and auditing. We'd monitor false positives on honest users
> before any real deployment."

---

## Slide 12 — Business Model & Close

**On slide**
- SaaS plugin for Shopify / WooCommerce, or API on AWS Lambda
- Priced per scored return or share of recovered value
- "Built spec-first with Kiro" + GitHub link + QR to repo

**Say**
> "ReturnGuard ships as a Shopify plugin or a Lambda API, priced per scored return.
> Built entirely spec-first in Kiro in one day. Free returns for honest shoppers,
> real protection for merchants, with an explanation every time. Thank you — happy
> to take questions."

---

## Q&A — ready answers (keep in your pocket)

- **Is it biased?** No sensitive attributes; SHAP explanation on every decision;
  human review for fraud; we'd monitor honest-user false positives.
- **Why is accuracy so high?** The public benchmark is highly separable; we report
  measured numbers and expect real data to be harder.
- **Latency?** ~45–70 ms per request, well under 100 ms.
- **What if the model is wrong on stage?** The 4 demo profiles are verified; the
  merchant can override live with an audit note.
- **Data source?** Public 60k-row benchmark; production needs merchant data + retraining.

## Timing cheat-sheet
- 0:00 Title → 0:15 Problem → 0:40 Insight → 0:55 Solution → 1:15 How →
  1:35 Demo honest → 2:00 Demo flagged → 2:25 SHAP → 2:45 Dashboard → close.
- If running long, cut slide 3 (Insight) and merge into slide 2.
