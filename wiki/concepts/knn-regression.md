---
concept: knn-regression
module: 02-statlearn
lectures: [L10]
isl-ref: 3.5
exercises:
  - CE1 problem 4a - write 10-fold CV pseudocode for KNN regression with MSE
  - CE1 problem 1e - read a KNN bias-variance plot over K and decide whether overfitting or underfitting would be preferable
related: [knn-classification, parametric-vs-nonparametric, flexibility-overfitting-underfitting, bias-variance-tradeoff, cross-validation, k-fold-cv, curse-of-dimensionality, standardization]
tags:
  - concept
  - module/02-statlearn
aliases:
  - K-nearest neighbors regression
  - KNN regression
---

# K-nearest neighbors (regression)

The regression sibling of [[knn-classification]]: at a test point $x_0$, find its $K$ nearest training points by Euclidean distance and predict the **average** of their $y$-values. Same K-as-flexibility story, same bias-variance picture, same curse-of-dimensionality death. The prof uses it as the standing example when explaining how to pick a hyperparameter via [[cross-validation]] in M5.

## Definition (prof's framing)

> "Recall KNN regression: $\hat f(x_0) = \frac{1}{K} \sum_{i \in \mathcal N_0} y_i$ where $\mathcal N_0$ is the K nearest training points. K small → high complexity (jagged, K=1 hits every training point); K large → low complexity (smooth, K = number-of-points = horizontal mean)." - [[L10-resample-1]]

The prof literally uses KNN regression as the running model-selection example in M5, "K in KNN, polynomial degree" are the two prototype hyperparameters CV is meant to tune.

## Notation & setup

- Training data $(x_1, y_1), \dots, (x_n, y_n)$ with $x_i \in \mathbb R^p$, $y_i \in \mathbb R$ (continuous response).
- $\mathcal N_0$ = index set of the $K$ training points closest to $x_0$ in Euclidean distance.
- Loss for tuning: MSE = $\frac{1}{n_{\text{val}}} \sum (y_j - \hat f(x_j))^2$ on a held-out / CV set.

## Formula(s) to know cold

$$\boxed{\;\hat f(x_0) = \frac{1}{K} \sum_{i \in \mathcal N_0} y_i\;}$$

Validation MSE (e.g. inside a CV loop):
$$\mathrm{MSE}_{\text{val}} = \frac{1}{|V|} \sum_{j \in V} \left(y_j - \hat f^{(-V)}(x_j)\right)^2$$

where $\hat f^{(-V)}$ is the KNN fit using all data *except* the validation fold $V$. CE1 problem 4a wants this written as 10-fold CV pseudocode.

## Insights & mental models

**KNN regression is the same mechanism as KNN classification, with `mean` swapping in for `majority vote`.** Same flexibility knob ($K$), same trade-off, same fix (CV for K).

**The K-flexibility picture, regression flavor** (slide in [[L10-resample-1]] with $f(x) = -x + x^2 + x^3$, $n = 61$, $K = 1, \dots, 25$, repeated $M = 1000$ times):

> "1 is going to suck, it's going to jump to every single point, and then 25 is going to be a lot, because there's 61 points … you're using like almost more than a third of the data, so that's going to be way too smooth." - [[L10-resample-1]]

- $K = 1$: $\hat f(x_i) = y_i$ exactly on training points → training MSE = 0. Test MSE high (variance).
- $K = n$: $\hat f(x_0) = \bar y$ everywhere, horizontal line at the global mean. Pure underfit.
- Optimal $K$ is intermediate, recovered by CV (Exercise CE1.4a is the canonical procedural drill).

**The bias-variance plot for KNN regression over $K$** (Figure 1 of CE1, used in problem 1e): squared bias rises with $K$ (smoother fit can't track wiggles → bias), variance falls with $K$ (averaging more points lowers Var$(\hat f(x_0))$), irreducible error is flat, total error is U-shaped, minimum somewhere in the middle. Reading this plot is the exam-relevant skill: identify the U, identify the minimum, name what each curve is.

**One subtlety the prof flagged** ([[L10-resample-1]]): in the slide's plot, "bias" is computed on the **fit data**, not from the true model. *"In other parts, we talk about the bias as how biased it is from the true model, not from the fit data."* In real settings you don't know the truth, so you have to **estimate** these from data, that's why we need resampling.

**Why KNN regression is the canonical CV target**: it has exactly one tuning parameter ($K$), the loss (MSE) is unambiguous, it works for any continuous $Y$, and it's the simplest nonparametric regressor. So in M5 the worked CV example is "tune $K$ in KNN regression by 10-fold CV", write the loop, compute MSE per fold, average, pick the $K$ at the minimum. ([[L10-resample-1]] / Exercise 5.1.)

### The pseudocode template (CE1 problem 4a)

```
Input: training data (x_i, y_i), candidate K values
For each K in candidate set:
  Randomly partition training data into 10 folds C_1, ..., C_10
  For j = 1, ..., 10:
    Train: hold out fold C_j; the rest is the training subset.
    For each x_i in C_j: predict ŷ_i = (1/K) * sum of y over the K nearest x's in the training subset.
    Compute MSE_j = (1/|C_j|) * sum_{i in C_j} (y_i - ŷ_i)^2
  CV(K) = (1/10) * sum_j MSE_j
Return the K minimizing CV(K)
```

Math notation, English, or pseudocode all acceptable per [[scope]]. The 2026 form would not require R syntax.

### Standardization

Same warning as KNN classification: Euclidean distance is scale-sensitive. **Standardize predictors first** (z-score) before fitting KNN. See [[standardization]], KNN regression is on the list of methods that are *not* scale-invariant.

## Pitfalls

- **Same K-direction trap as classification.** Small $K$ = more flexible. Easy T/F mistake.
- **No standardization.** Distances dominated by the largest-scale predictor; results garbage.
- **No extrapolation.** $\hat f(x_0)$ for $x_0$ outside the training range is the average of the nearest training points, pulled toward whatever boundary points exist, not extrapolated along any trend.
- **Curse of dimensionality.** Same as classification: in high $p$, no point is meaningfully "near" any other. KNN's mechanism breaks.
- **K must be an integer.** Doesn't matter for the math, but a candidate sweep is over $K = 1, 2, 3, \dots$.
- **Tie-breaking for "the K nearest"**: if multiple points are equidistant, libraries vary; doesn't affect exam-style hand calculations.

## Scope vs ISLP

- **In scope:** the formula, $K$ as flexibility, bias-variance interpretation, picking $K$ by CV, the use of KNN regression as the running model-selection example in M5.
- **Look up in ISLP:** §3.5 ("Comparison of Linear Regression with K-Nearest Neighbors"), the KNN regression formula, comparison with linear regression, the curse-of-dimensionality discussion. Equation (3.39) is the formula above; Figures 3.16–3.20 show the contrast with linear models in 1D and as $p$ grows.
- **Skip in ISLP:** weighted KNN ([[scope|out of scope]]). Distance metrics other than Euclidean (M10 [[distance-metrics]] discusses options for clustering, but KNN here is Euclidean only).

## Exercise instances

- **CE1 problem 4a**: write 10-fold CV pseudocode for KNN regression with MSE, including the formula for the validation error. Procedural template.
- **CE1 problem 1e**: given a bias-variance plot of KNN regression over $K$ (Figure 1 of CE1), evaluate four T/F statements: (i) decreasing $K$ increases flexibility; (ii) squared bias always contributes the most; (iii) you have enough info to pick $K$; (iv) for the plotted range, would overfitting or underfitting be preferable. The classic plot-reading + concept-application combo.

## How it might appear on the exam

- **Pseudocode question** (the prof's flagged 2026 format from [[scope]] §3): "Write out how you'd choose $K$ for KNN regression by 10-fold cross-validation." Math, English, or pseudocode all OK. CE1 problem 4a is the template.
- **Bias-variance plot reading.** Given a curve of bias², variance, irreducible, total over $K$, identify the U-shape minimum, name each curve, decide which side of the U is overfit and which is underfit.
- **Method comparison.** "When would KNN regression beat linear regression?" → strongly nonlinear $f$, low $p$, lots of data. (When would linear win? → low $n$, high $p$, additive structure.)
- **Direction-of-effect T/F.** Same family as KNN classification: small $K$ = high flexibility, low bias, high variance.
- **Hand calculation** (less common, but plausible): given a small dataset and a test point, compute $\hat f(x_0)$ by averaging the $K$ nearest $y$-values. Direct analog of Exercise 4.1.

## Related

- [[knn-classification]]: same algorithm, vote instead of average; the more heavily exam-tested sibling
- [[parametric-vs-nonparametric]]: KNN regression sits squarely on the nonparametric side
- [[flexibility-overfitting-underfitting]]: $K$ is the flexibility knob; the U-shape applies
- [[bias-variance-tradeoff]]: the formal decomposition behind the U
- [[cross-validation]] / [[k-fold-cv]]: the standard way to pick $K$
- [[curse-of-dimensionality]]: KNN regression's headline failure mode in high $p$
- [[standardization]]: mandatory preprocessing
