---
concept: partial-dependence-plots
module: 09-boosting
lectures: [L21, L27]
isl-ref: 8.2.5
exercises: []
related: [boosting, gradient-boosting, xgboost, random-forest, variable-importance, regression-tree, classification-tree]
tags:
  - concept
  - module/09-boosting
aliases:
  - PDP
  - partial dependence
  - PDPs
---

# Partial dependence plots

The standard interpretability patch for tree ensembles. Boosted forests / XGBoost models are individually opaque ("we have a model that we can't understand" — [[L21-unsupervised-1]]); PDPs claw qualitative interpretability back by **collapsing the model onto one predictor at a time**, marginalizing the rest. The book's empirical estimator is just "average the prediction over all observations with $X_j$ held fixed." Fair-game on the exam to ask what they show.

## Definition (prof's framing)

> "The partial dependence function represents the effects of $X_j$ on $f(X)$ after accounting for the effects of all the other variables. And that's of course different than… what's the effect of $X$ if everything else was not there. That's a totally different question." — [[L21-unsupervised-1]]

The mathematical ideal:
$$f_j(x_j) \;=\; \mathbb{E}_{X_{-j}}\!\left[\, f(x_j, X_{-j}) \,\right]$$

— marginalize over the joint of the other predictors $X_{-j}$ at $X_j = x_j$.

The book's empirical estimator (the one you actually compute and the one you'd write on the exam):
$$\bar f_j(x_j) \;=\; \frac{1}{N} \sum_{i=1}^N f\!\left( x_j,\; x_{i, -j} \right).$$

For a chosen sweep of $X_j$ values, evaluate the trained model with $X_j = x_j$ but the other coordinates set to each observation's own values, then average.

## Notation & setup

- $X_j$ — the one predictor you're interested in.
- $X_{-j}$ — all other predictors, treated as a vector.
- $f(\cdot)$ — the trained black-box (boosted ensemble, RF, etc.).
- $\bar f_j(x_j)$ — the empirical partial dependence at $X_j = x_j$.

## How you'd compute one (pseudocode-ish, exam-friendly)

For a chosen variable $X_j$:

1. Pick a grid of values $x_j^{(1)}, \dots, x_j^{(K)}$ over the range of $X_j$ (e.g. quantiles of the training data).
2. For each grid value $x_j^{(k)}$:
   - For each training observation $i = 1, \dots, N$, replace the $j$-th coordinate with $x_j^{(k)}$ but keep $x_{i,-j}$ as is. Call the modified row $\tilde x_i^{(k)}$.
   - Compute $\hat y_i^{(k)} = \hat f(\tilde x_i^{(k)})$.
   - Average: $\bar f_j(x_j^{(k)}) = \frac{1}{N} \sum_i \hat y_i^{(k)}$.
3. Plot $\bar f_j(x_j^{(k)})$ versus $x_j^{(k)}$.

## Insights & mental models

- **Marginal effect, not causal effect.** PDPs show how the *prediction* changes with $X_j$ "after accounting for" the other variables — averaging out their joint distribution. They are *not* "what would happen if $X_j$ changed and nothing else did" (that's a counterfactual / causal question).
- **Why use the data and not the population.**
  > "You don't have the underlying distribution, so you use the data to approximate that distribution." — [[L21-unsupervised-1]]
  Same bootstrapping logic — empirical distribution stands in for the unknown joint of $X_{-j}$.
- **Local interpretability, not global understanding.** PDPs give you a one-variable curve. They don't tell you about interactions (for that, you'd need 2-D PDPs or ICE / Shapley methods, all out of scope here).
- **Fragility under correlation.** Averaging over the data with $X_j$ pinned can create implausible synthetic observations:
  > "You might be looking at a tiny, tiny house with like 50 bedrooms, which doesn't make any sense." — [[L21-unsupervised-1]]
  When predictors are highly correlated, the PDP averages over combinations that essentially never occur in reality.
- **Stability check.** PDPs are computed from the data; they wobble across resamples / bootstraps. If the overall structure is stable that's a sign of believability; a noisy PDP across reruns is a sign the marginal effect is weak or the model is unstable on that dimension.
- **Pairs naturally with [[variable-importance]].** Importance plots tell you *which* variables matter; PDPs tell you *how* the top variables matter. The two together give you a usable picture of an opaque ensemble.

## Worked example — Boston housing (slides + lecture)

After fitting `boost.boston <- gbm(medv~., ..., n.trees=5000, interaction.depth=4)`, the slide deck shows PDPs for `rm` (number of rooms — the most important predictor) and `lstat` (% lower-status population). Both are in the slide refs:

```r
plot(boost.boston, i = "rm",    ylab = "medv")
plot(boost.boston, i = "lstat", ylab = "medv")
```

PDP for `rm` is monotone increasing (more rooms → higher predicted price). PDP for `lstat` is monotone decreasing (higher %low-SES → lower predicted price). Magnitude and shape (kinks, plateaus) are the qualitative info you'd quote on the exam.

## Exam signals

> "Partial dependence plots… know what they mean and where they come from, but you won't compute them." — [[L27-summary]] (paraphrased — see the [[variable-importance|influence/importance plot]] callout for the full Q6c context: "[Variable] importance plots are fair game — know what they mean and where they come from, but you won't compute them.")

The same logic the prof applied to importance plots applies to PDPs: in scope (slides + lecture), explicitly fair game for "what does this show" interpretation, not for hand-computation.

> "If it was covered either in the slides or in the exercises, then I would say fair game." — [[L27-summary]]

## Pitfalls

- **Reading PDPs as causal effects.** They aren't. They are averages of model predictions, conditional on a fixed $X_j$ — useful for understanding what the model learned, not for "what would happen if I changed this in the world."
- **Forgetting they assume independence between $X_j$ and $X_{-j}$.** When predictors are heavily correlated, the PDP averages over implausible combinations and can be misleading.
- **Treating one PDP as a global model summary.** A PDP shows one variable; the model's behavior also depends on interactions and on combinations of other variables. PDPs are point summaries, not the whole picture.
- **Confusing PDP with the marginal distribution of $\hat y$ at $X_j = x_j$.** PDP averages over training $X_{-j}$; the marginal averages over the joint $(X_j, X_{-j})$ at the data's natural distribution. Different objects.

## Scope vs ISLR

- **In scope:** the empirical estimator $\bar f_j(x_j) = \tfrac1N \sum_i f(x_j, x_{i, -j})$; the marginalization-over-$X_{-j}$ idea; what a PDP shows (qualitative effect of $X_j$ on $\hat y$, after accounting for others); the contrast with "$X$ if everything else were not there" (causal); pairing with [[variable-importance]] for tree-ensemble interpretation.
- **Look up in ISLR:** PDPs aren't deeply covered in ISLR's chapter 8; the slide deck refers to **Elements of Statistical Learning §10.13.2** for the full treatment. Anders does not need this depth — the empirical formula and the marginalization framing are enough.
- **Skip in ISLR (book-only / out of scope):**
  - **ICE plots (individual conditional expectation)** — not lectured.
  - **Shapley / SHAP values** — [[L21-unsupervised-1]] mentions interpretability machinery in passing, never derives.
  - **Two-variable / interaction PDPs** — not lectured.

## Exercise instances

None. No Exercise 9 problem touches PDPs explicitly — they're pure lecture / slide content.

## How it might appear on the exam

- **Output interpretation** — given a PDP for one variable from a boosted-tree fit, describe the qualitative effect (e.g. "predicted house price increases monotonically with number of rooms; the effect plateaus past 7 rooms").
- **Multiple choice / true-false** — "a PDP shows the causal effect of $X_j$ on $Y$" (false — it's a marginal effect on the model's prediction), "a PDP averages the model's prediction across the data with $X_j$ fixed" (true).
- **Conceptual short-answer** — "you have a black-box gradient-boosted model and you want to understand how `lstat` affects predicted price. What do you compute?" Expected: the partial dependence of $\hat f$ on `lstat` — sweep `lstat`, hold the rest at the data values, average the predictions, plot.
- **Pseudocode** — given the empirical PDP formula, write the algorithm in pseudocode (per the open-book / no-language rule, pseudocode is acceptable).
- **Pitfall awareness** — "when can a PDP mislead you?" Expected: highly correlated predictors → averaging over implausible $(x_j, x_{-j})$ combinations.

## Related

- [[boosting]] — the parent context; PDPs are one of the two interpretability tools the prof covers for boosted ensembles.
- [[gradient-boosting]] / [[xgboost]] — the typical opaque-ensemble PDPs are computed from.
- [[random-forest]] — same interpretability gap, same PDP fix; PDPs work for any black-box $\hat f$.
- [[variable-importance]] — the other interpretability tool from module 8 / 9; importance answers "which variables matter," PDPs answer "how do the important ones matter."
- [[regression-tree]] / [[classification-tree]] — the components of the ensembles that PDPs interpret; single trees don't need PDPs because they're already interpretable.
