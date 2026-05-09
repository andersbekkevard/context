---
concept: one-standard-error-rule
module: 05-resample
lectures: [L10]
isl-ref: 6.1.3
exercises: []
related: [k-fold-cv, cross-validation, cost-complexity-pruning, subset-selection]
tags:
  - concept
  - module/05-resample
aliases:
  - 1-SE rule
  - one-SE rule
---

# One-standard-error rule

The prof's preferred selection criterion when the CV curve is flat near the minimum: **pick the simplest model whose CV error is within one standard error of the minimum.** Bias toward simpler models. The SE estimate itself is "not quite valid", and that footnote is a thought question the prof flagged.

## Definition (prof's framing)

Given a hyperparameter $\theta$ (e.g. polynomial degree, $K$ in KNN, $\lambda$ in ridge, tree size in pruning) and a k-fold CV curve $\text{CV}(\theta)$ with associated SE $\widehat{\text{SE}}(\text{CV}(\theta))$:

1. Find $\hat\theta = \arg\min_\theta \text{CV}(\theta)$.
2. Walk $\theta$ in the direction of "simpler" until the bound

   $$\text{CV}(\theta) \leq \text{CV}(\hat\theta) + \widehat{\text{SE}}\big(\text{CV}(\hat\theta)\big)$$

   stops holding.
3. The simplest $\theta$ still satisfying the bound wins.

> "The one standard error rule is to choose the simplest model… within one standard error of the minimal error.", slide deck, [[L10-resample-1]]

## Notation & setup

- $\hat\theta$ = hyperparameter at the CV minimum.
- $\widehat{\text{SE}}(\text{CV}(\hat\theta))$ = sample standard deviation of the per-fold MSEs at $\hat\theta$, divided by $\sqrt{k}$ in some textbook formulations; the slide deck uses the unscaled sample SD form (see [[k-fold-cv]] for the formula).
- "Simpler" depends on the model, lower polynomial degree, larger $K$ in KNN (smoother), larger $\lambda$ in ridge/lasso, fewer terminal nodes in trees, fewer predictors in subset selection.

## Formula(s) to know cold

The selection rule:

$$\boxed{\theta^* = \min\Big\{\theta \text{ simpler than } \hat\theta \;:\; \text{CV}(\theta) \leq \text{CV}(\hat\theta) + \widehat{\text{SE}}(\text{CV}(\hat\theta))\Big\}}$$

(more precisely: among models within one SE of the minimum, take the simplest).

The SE estimate (per slide deck):

$$\widehat{\text{SE}}\big(\text{CV}_{(k)}(\theta)\big) = \sqrt{\frac{1}{k - 1} \sum_{j=1}^{k} \big(\text{MSE}_j(\theta) - \overline{\text{MSE}}(\theta)\big)^2}$$

## Insights & mental models

- **The motivation** is Anders's question in L10 that the prof acknowledged on the spot: picking the model at the test-set minimum is itself a kind of fit-to-test, the minimum is noisy, and a slightly worse but much simpler model is usually a better choice. *"Instead of using the lowest point, you move over a bit."* - [[L10-resample-1]]
- **Why "simpler"** is the tiebreaker: simpler models generalize better (Occam's razor), are more interpretable, and you've paid for them with hard-won understanding rather than CV-noise exploitation. The 1-SE rule says "the data can't actually distinguish these models, so prefer the simpler one."
- **The "not quite valid" footnote** the prof flagged: the SE you compute on per-fold MSEs is *not* a clean independent SE because the same held-out folds were used both for selecting $\hat\theta$ and for estimating the SE. *"You're already using the validation data to select, so the SE you compute on it isn't a clean independent SE."* - [[L10-resample-1]]. The slide footnote asks the question explicitly: *"Strictly speaking, this estimate is not quite valid. Why?"*
- **Where it shows up later in the course:**
  - Module 6: best-subset selection, pick the simplest model within 1 SE of the CV minimum.
  - Module 6: ridge / lasso $\lambda$ tuning, `cv.glmnet` returns both `lambda.min` and `lambda.1se`.
  - Module 8: tree pruning, pick the smallest tree within 1 SE of the minimum CV deviance.
  - Same logic everywhere: the CV curve is often flat near the minimum, and there's a meaningful "simpler" direction.

## Exam signals

> "There are corrections, instead of using the lowest point, you move over a bit." - [[L10-resample-1]] (foreshadowing the 1-SE rule)

> [!note] Footnote on the slide
> "Strictly speaking, this estimate is not quite valid. Why?", slide deck

The prof said *"think about why"* and let it sit as a thought question. Be ready to answer it: the SE is computed from the same held-out folds used to pick the model, so it's not an independent estimate.

## Pitfalls

- **Forgetting the "simpler" direction**. The rule isn't symmetric, among models within 1 SE, you want the *simplest*, not just any one of them.
- **Confusing the SE formula** with the bootstrap SE or the cross-validation MSE itself. The 1-SE band is built from the per-fold MSE *standard deviation*, not from bootstrapped resamples.
- **Treating it as a hard rule** rather than a default. In some applications you genuinely want the minimum (e.g. last-mile prediction accuracy contests). The 1-SE rule's bias toward simplicity is a feature for interpretability and generalization, not a free lunch.

## Scope vs ISLR

- **In scope:** the rule itself, the SE formula, the "not quite valid" footnote, the application to k-fold CV, the bias-toward-simplicity rationale.
- **Look up in ISLR:** §6.1.3 (pp. 246), appears in the model-selection chapter as a tiebreaker for subset selection. Also discussed in §6.2.3 for ridge / lasso $\lambda$ choice. The rule is referenced (less explicitly) for tree pruning in §8.1.
- **Skip in ISLR (book-only, prof excluded):** detailed theoretical justification of the SE estimate (it's not derived rigorously; the prof's footnote gestures at why it's heuristic).

## Exercise instances

No dedicated recommended exercise on the 1-SE rule itself; it's a tool that shows up implicitly in:

- **Exercise6.3b**: best-subset selection scored by 10-fold CV (the natural place to apply 1-SE).
- **Exercise8.2c**: `cv.tree()` for choosing pruning size (1-SE often used to pick a smaller tree than the bare minimum).

## How it might appear on the exam

- **Output interpretation:** given a CV-vs-$\theta$ plot with error bars, pick the model the 1-SE rule recommends. Read off CV($\hat\theta$) + SE, find the leftmost / smoothest / sparsest model still under that bound.
- **Conceptual:** "Why prefer the simplest model within 1 SE of the CV minimum?" → CV curves are noisy near the minimum; the data can't reliably distinguish the slight differences in CV error among nearby candidates; simpler generalizes better and is more interpretable.
- **The footnote question:** "Why is the SE estimate not quite valid?" → because the same held-out folds were used to select the optimal hyperparameter and to compute the SE, so the SE is not an independent estimate.

## Related

- [[k-fold-cv]]: supplies the SE formula and the CV curve
- [[cross-validation]]: global picture of the prof's preferred tuner
- [[cost-complexity-pruning]]: natural application in module 8 (tree size selection)
- [[subset-selection]]: natural application in module 6
