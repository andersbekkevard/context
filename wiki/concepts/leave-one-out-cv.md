---
concept: leave-one-out-cv
module: 05-resample
lectures: [L10, L11]
isl-ref: 5.1.2
exercises:
  - CE1 problem 4b - true/false on bias and variance of LOOCV vs k-fold
related: [k-fold-cv, validation-set-approach, design-matrix-and-hat-matrix, cross-validation, bias-variance-tradeoff]
tags:
  - concept
  - module/05-resample
aliases:
  - LOOCV
  - leave one out
---

# Leave-one-out cross-validation (LOOCV)

The other extreme from the validation set: $n$-fold CV with one observation per fold. Low bias (train on $n-1$ each time), but **high variance** because the $n$ training sets are nearly identical → fold errors highly correlated → averaging them doesn't help much. For OLS there's a beautiful hat-matrix shortcut that needs only one fit.

## Definition (prof's framing)

For $i = 1, \dots, n$:

1. Hold out observation $i$.
2. Fit the model on the remaining $n - 1$.
3. Predict $\hat y_i$ and compute $\text{MSE}_i = (y_i - \hat y_i)^2$ (regression) or $\text{Err}_i = \mathbb{1}(y_i \neq \hat y_i)$ (classification).

Average across all $n$ folds:

$$\text{CV}_{(n)} = \frac{1}{n} \sum_{i=1}^{n} \text{MSE}_i$$

## Notation & setup

- $n$ = sample size; LOOCV requires $n$ refits.
- No randomness: the procedure is fully deterministic given the data.
- For OLS, the closed-form shortcut below replaces the $n$ refits with a single fit.

## Formula(s) to know cold

The general LOOCV estimator:

$$\boxed{\text{CV}_{(n)} = \frac{1}{n} \sum_{i=1}^{n} \big(y_i - \hat y_i^{(-i)}\big)^2}$$

where $\hat y_i^{(-i)}$ is the prediction at $x_i$ from the model trained without observation $i$.

**The hat-matrix shortcut (OLS only):**

$$\boxed{\text{CV}_{(n)} = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{y_i - \hat y_i}{1 - h_{ii}} \right)^2}$$

where $\hat y_i$ is the **full-data** fitted value (no leave-out) and $h_{ii}$ is the $i$-th diagonal element of the hat matrix $\mathbf H = \mathbf X(\mathbf X^\top \mathbf X)^{-1} \mathbf X^\top$. **Only one fit needed.**

> "We have a nice shortcut for linear regression and in some other settings, but not all." - [[L10-resample-1]]

This is **derived in compulsory exercise 1** (per the slide deck pointer). The leverage $h_{ii}$ measures how much obs $i$ pulls the fitted surface toward itself; $1/(1-h_{ii})$ inflates the residual to account for the fact that obs $i$ helped fit the surface.

## Insights & mental models

- **No randomness**: fully deterministic. The validation set approach varies wildly across splits; LOOCV doesn't.
- **The "highly-correlated folds" intuition** is the core variance argument:
  - Two LOOCV training sets share $n - 2$ of $n - 1$ observations → essentially identical → the per-fold errors $\text{MSE}_i$ are highly correlated.
  - Variance of an average of correlated variables: $\text{Var}(\sum a_i X_i) = \sum a_i^2 \text{Var}(X_i) + 2 \sum_{i < j} a_i a_j \text{Cov}(X_i, X_j)$. The cross-covariance terms blow up when the $X_i$ are highly correlated, so the average has high variance, even though we averaged $n$ estimates.
- **Outlier sensitivity** is a property to know: an extreme point hits one LOOCV fold with a huge held-out error, can dominate the average. The prof framed this as a kind of feature (it tells you which points are problematic) but mostly as a thing to be aware of.
- **Independence trap (extra-bad version):** if your data is dependent in time/space, LOOCV is *terrible*: the model can predict the held-out point from its neighbor, fold error is artificially tiny, CV will recommend the most complex model. *"It will be basically identical to just using the likelihood without any penalization."* - [[L10-resample-1]]
- **LOOCV = $k$-fold with $k = n$.** The two ends of the same spectrum.

## Pros and cons (prof's slide form)

**Pros:**
- No randomness in splits.
- Low bias, train on $n - 1$ each time, almost the full data.

**Cons:**
- Expensive, $n$ refits (unless you're in OLS-land with the shortcut).
- **High variance**: correlated folds inflate the variance of the average.
- Extra-bad outlier and dependence sensitivity.

## Exam signals

> "Two training sets only differ by one observation, thus estimates from each fold highly correlated, which can lead to high variance in their average.", slide deck (paraphrased in [[L10-resample-1]])

> "Leave one out will have a better bias. But the K-fold will likely have a much better variance. And typically in this setting, you're winning by having less variance." - [[L11-resample-2]]

CE1 problem 4b runs the comparison as true/false:

- *"5-fold CV will generally lead to an estimate of the prediction error with less bias, but more variance, than LOOCV"* → **false.** LOOCV has lower bias (trains on more data); k-fold has lower variance (less correlated folds). The statement reverses both.
- *"LOOCV is a form of bootstrapping"* → **false.** LOOCV partitions; bootstrap resamples with replacement.

## Pitfalls

- **Mixing up the bias and variance directions.** LOOCV: low bias, high variance. The wrong-direction trap is on CE1 problem 4b, easy to flip.
- **Using LOOCV under temporal/spatial correlation.** Worse than k-fold here, see independence trap above.
- **Forgetting the OLS-only restriction on the shortcut.** Eq. (5.2) in ISLR works only for linear models fit by least squares (or with the appropriate generalization for other linear-projection methods). For trees, KNN, GAMs, neural nets, etc., you have to actually do the $n$ refits.

## Scope vs ISLR

- **In scope:** the algorithm, the bias/variance comparison with k-fold (CE1 4b), the hat-matrix shortcut (CE1 derives it), the independence trap.
- **Look up in ISLR:** §5.1.2 (pp. 200–202), equation 5.2 is the shortcut. The footnote on the multiple-regression generalization of leverage is also there.

## Exercise instances

- **CE1 problem 4b**: true/false on bias and variance of LOOCV vs 5-fold CV (and "LOOCV is a form of bootstrapping"). The flipped-direction trap.
- The OLS shortcut derivation is referenced in the slides as "see Compulsory exercise 1", historical, but the shortcut formula itself is exam-worthy.

## How it might appear on the exam

- **True/false** on bias / variance / compute properties, direct CE1 problem 4b style. Easy to flip directions; write reasoning even for T/F.
- **"Compute LOOCV by hand"** for a small dataset (say $n = 5$, simple linear regression), feasible because the shortcut works.
- **"Why is k-fold preferred to LOOCV"** → variance argument: nearly-identical training sets → highly-correlated fold errors → large variance of the average. K-fold de-correlates by making the training sets actually different.
- **Conceptual on the OLS shortcut**: what does $h_{ii}$ measure, why does $1/(1-h_{ii})$ appear? Connects to leverage from module 3 ([[design-matrix-and-hat-matrix]]).
- **Independence-trap question**: given temporally correlated data, would you prefer LOOCV or k-fold? Why?

## Related

- [[k-fold-cv]]: the practical compromise; LOOCV = $k$-fold with $k = n$
- [[validation-set-approach]]: the other extreme
- [[design-matrix-and-hat-matrix]]: where $h_{ii}$ comes from; the shortcut formula uses it
- [[bias-variance-tradeoff]]: the lens for comparing CV variants
- [[cross-validation]]: global picture
