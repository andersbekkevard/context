---
concept: aic-bic-conceptual
module: 05-resample
lectures: [L10, L12, L13]
isl-ref: 6.1.3
exercises:
  - Exercise6.3a — pick best subset using Cp, BIC, adjusted R² on Credit (cross-check against CV)
related: [cross-validation, k-fold-cv, subset-selection, r-squared]
tags:
  - concept
  - module/05-resample
aliases:
  - AIC
  - BIC
  - Mallows Cp
  - information criteria
  - penalty criteria
---

# Penalty criteria (AIC / BIC / Cp) — conceptual only

The prof distrusts these and **explicitly excluded the formulas and derivations from the exam.** Know they exist, know what they're for, know why he prefers cross-validation. That's it.

## Definition (prof's framing)

A family of training-error-plus-complexity-penalty criteria for model selection:

- **Mallows' Cp** — adjusts training RSS upward to estimate test MSE.
- **AIC** (Akaike Information Criterion) — derived from information theory; for Gaussian models reduces to a multiple of Cp.
- **BIC** (Bayesian Information Criterion) — derived from Bayesian arguments; penalizes complexity more aggressively than AIC.
- **Adjusted $R^2$** — adjusts the fraction of variance explained by predictor count.

All four share the same structural idea: training error always falls as you add parameters, so penalize the parameter count to estimate test performance.

> "If all your assumptions are true… you can use these criteria, which are essentially penalties, for adding additional parameters to your model. And then those alone can give you a nice way to evaluate your model." — [[L10-resample-1]]

## What's in scope vs out

**In scope (conceptual):**
- They exist as ways to penalize training error by model complexity.
- The conceptual claim "penalize complexity to estimate test error from training error."
- BIC penalizes more aggressively than AIC ⇒ favors smaller models.
- They are an **alternative to CV** that works when the underlying assumptions are met.
- The prof's verdict: distrusted, prefers CV.

**Out of scope (prof's verbatim exclusion):**

> "I really don't think I'm going to ask any questions about this." — [[L12-modelsel-1]] / [[L13-modelsel-2]]

The slide deck for module 5 also explicitly defers AIC / BIC to Module 6 (it lists them under "alternative strategy for model selection… see Module 6"). And the Module 6 lectures L12 / L13 then exclude the formulas and derivations from the exam.

> "I really don't think I'm going to ask any questions about this. I'm not going to ask you to use these. I'm not going to ask you to derive them." — [[L12-modelsel-1]]

The prof acknowledged the pitch on penalty criteria *"didn't land cleanly: 'Maybe that wasn't so helpful in understanding. I thought it was interesting.'"* Takeaway: he's *not* testing the formulas for $C_p$, AIC, BIC, adjusted $R^2$ — only the conceptual claim that an information criterion penalises model complexity. They don't show up on the exam.

So this atom is **deliberately a stub**: the conceptual existence statement is in scope; the algebra is not.

## Why the prof prefers CV

> "Your assumptions have to be right. And they're not. They're not typically right." — [[L10-resample-1]]

AIC / BIC / Cp depend on:
- Correct distributional assumptions (typically Gaussian errors, or specific GLM family).
- Correctly specified model.
- IID samples.
- Variance estimate $\hat\sigma^2$ that is itself reasonable.

In real data, *"distribution wasn't what you thought, samples are correlated in time/space/relationships, people you check are not independent because they're all related or they're all white or they're all whatever."* — [[L10-resample-1]]. Resampling makes **fewer assumptions** and is more robust to violations.

> "These things are used a lot and they're nice because they don't make so many assumptions." — [[L10-resample-1]]

## Exam signals

> "I really don't think I'm going to ask any questions about this." — [[L12-modelsel-1]]

> "I'm not going to ask you to use these. I'm not going to ask you to derive them." — [[L12-modelsel-1]]

The conceptual one-liner — "they exist, they penalize complexity, prof distrusts them, prefers CV" — is what to remember. Don't memorize the formulas.

## Pitfalls

- **Memorizing the formulas.** Wasted effort given the prof's explicit exclusion.
- **Forgetting BIC penalizes more than AIC.** This is the one comparison that might appear conceptually (BIC favors smaller models).
- **Confusing Cp with the OLS test-error formula.** Cp is a training-set quantity adjusted to estimate test error; not the same as the actual test error.

## Scope vs ISLR

- **In scope:** they exist, they penalize complexity, prof distrusts them, prefers CV.
- **Look up in ISLR:** §6.1.3 (pp. 244–246) for definitions and the formulas — only if you're curious, **not for the exam**.
- **Skip in ISLR (book-only, prof excluded):** all derivations of AIC (information theory), BIC (Bayesian), Cp (unbiased estimator of test MSE under Gaussian errors), and adjusted $R^2$. The prof was emphatic.

## Exercise instances

- **Exercise6.3a** — pick the best subset of predictors on the Credit data using Cp, BIC, and adjusted $R^2$. Then **cross-check against k-fold CV** (the prof's preferred method) in 6.3b. The exercise exists mostly to show that CV and the penalty criteria *agree most of the time* but diverge in edge cases — and to give Anders a reason to trust CV more.

## How it might appear on the exam

- **Conceptual / multiple choice:** "AIC and BIC are alternatives to CV that estimate test error by penalizing the training error for model complexity" → true.
- **Compare BIC vs AIC:** BIC penalizes complexity more than AIC ⇒ prefers smaller models (the standard one-line comparison).
- **"Why does the prof prefer CV over AIC/BIC?"** → AIC/BIC depend on assumptions (correct distribution, IID samples, well-specified model) that often fail in practice; CV makes fewer assumptions.
- **Definitely not asked:** plug into a formula, derive, or compare numerical values.

## Related

- [[cross-validation]] — the prof's preferred alternative
- [[k-fold-cv]] — the operational workhorse
- [[subset-selection]] — where these criteria show up as a model-selection tool (in module 6)
- [[r-squared]] — adjusted-$R^2$ is the related distrusted statistic
