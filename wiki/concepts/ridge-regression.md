---
concept: ridge-regression
module: 06-modelsel
lectures: [L04, L12, L13, L14, L15, L24]
isl-ref: 6.2.1
exercises:
  - Exercise6.5  -  apply ridge to Credit, compare with OLS
related: [lasso, ridge-vs-lasso-geometry, elastic-net, principal-component-regression, bias-variance-tradeoff, cross-validation, regularization, standardization, high-dimensional-regression, collinearity, double-descent]
tags:
  - concept
  - module/06-modelsel
aliases:
  - L2 regularization
  - L2 penalty
  - Tikhonov regularization
---

# Ridge regression

The prof's preferred lens on regularization: an L2 penalty on the coefficients that smoothly shrinks them toward zero, trading a tiny bit of bias for a lot of variance reduction. He frames [[regularization]] as *"the most important variant of model selection that we talk about throughout"* - [[L14-modelsel-3]], and ridge as the workhorse instance: closed-form, easy to fit, works when $p > n$, but never sets a coefficient exactly to zero (so does **not** do variable selection, that's [[lasso]]'s job).

## Definition (prof's framing)

> "The ridge regression coefs $\beta^R$ are the ones that minimize $\text{RSS} + \lambda \sum_{j=1}^p \beta_j^2$, with $\lambda > 0$ being a tuning parameter.", slide deck (selection_regularization_presentation_lecture1.md)

> "This is also known as L2. It's entirely possible that I will just start calling it L2 one day. So it would be nice if you can remember that L2 and ridge mean exactly the same thing." - [[L12-modelsel-1]]

The prof also occasionally calls it "Tikhonov" (he can't pronounce it), the engineering / inverse-problems name for the same idea - [[L13-modelsel-2]].

## Notation & setup

- $\beta_0$ = intercept (**not penalized**); $\beta_1, \dots, \beta_p$ = slope coefficients (penalized).
- $\lambda \geq 0$ = tuning parameter, chosen by [[cross-validation]].
- $X$ = $n \times p$ design matrix; **must be standardized** (mean 0, sd 1 per column) before fitting.
- Sometimes written as the constraint form: minimize RSS subject to $\sum_j \beta_j^2 \le s$. Equivalent, with $s$ inversely related to $\lambda$.

## Formula(s) to know cold

**Penalized objective:**
$$\hat\beta^R_\lambda = \arg\min_\beta \left\{ \sum_{i=1}^n \left(y_i - \beta_0 - \sum_{j=1}^p \beta_j x_{ij}\right)^2 + \lambda \sum_{j=1}^p \beta_j^2 \right\}$$

**Constraint form (equivalent):**
$$\min_\beta \text{RSS} \quad \text{subject to} \quad \sum_{j=1}^p \beta_j^2 \le s$$

**Standardization** (slide deck verbatim):
$$\tilde{x}_{ij} = \frac{x_{ij}}{\sqrt{\frac{1}{n}\sum_i (x_{ij} - \bar{x}_j)^2}}$$

Two extremes:
- $\lambda = 0$ → recover OLS.
- $\lambda \to \infty$ → all $\hat\beta_j \to 0$ (intercept stays).

The prof gestures at the closed form $\hat\beta^R = (X^\top X + \lambda I)^{-1} X^\top y$ (the $\lambda I$ regularizes the inverse, works even when $X^\top X$ is singular) but doesn't push the derivation. ISLP §6.2.1 gives it.

## Insights & mental models

### Tug of war (the prof's central image)

> "RSS… it wants to make those parameters beta whatever it can to fit the data. So it's pulling them away from zero, either to big numbers, positive or negative, whatever. It's pulling them away from zero. And then this thing [the penalty] pulls them back to zero. So one is pushing away, one is pulling back. So you have this tug of war on the betas." - [[L12-modelsel-1]]

The optimizer settles on a unique compromise: a few large betas (the most useful ones) and the rest small. Letting one big $\beta$ in costs the same as several small ones, but if the data really needs that one $\beta$, the RSS gain outweighs the penalty.

### Smooth shrinkage, never exactly zero

The penalty $\beta^2$ is **flat at zero** ($\frac{d}{d\beta}\beta^2 = 0$ at $\beta = 0$), there's no gradient pulling coefficients *to* zero, only *toward* zero.

> "This is exactly why this ridge regression term doesn't ever get you to zero, because the squared function just doesn't [have a kink] there." - [[L13-modelsel-2]]

Consequence: ridge **does not perform variable selection**. All $p$ predictors stay in the final model, just with shrunken coefficients.

> "Often it's zero enough that you can just throw them away if you want, but it won't guarantee it." - [[L13-modelsel-2]]

### Works when $p \geq n$ (ridge's headline benefit)

OLS blows up when $p \geq n$, $X^\top X$ singular, infinite least-squares solutions. Ridge stays unique because $\lambda I$ regularizes the inverse.

> "If you have too many parameters, you can add a regularizer to keep the model finding a unique solution. It maintains the model as convex, meaning it will still have a unique solution for a given value of $\lambda$." - [[L13-modelsel-2]]

This is the same trick that makes giant modern models trainable. Connects to [[double-descent]], in [[L04-statlearn-3]] the prof showed an over-parameterized polynomial fit and called the implicit minimum-norm solution *"ridge regression… even though it's not actually put in there explicitly."*

### "Socialist" personality

> "[Lasso] encourages winners and losers, right? It's the capitalist regularization method. Whereas the ridge one… encourages ties, encourages that everyone has a vote and no one has zeros." - [[L14-modelsel-3]]

> "If we're in this ridge case, when we use ridge regression, we're actually averaging over the two. So we're saying: don't let any one of these dominate." - [[L13-modelsel-2]]

When two predictors are correlated, ridge averages over them (keeps both at moderate values). Lasso would pick one and zero the other. Robust because in the next dataset, the "noisy" one might flip.

### Squaring the penalty hurts useful big coefficients

Two regions of $\beta^2$ are bad:
1. **Far from zero**, penalty grows fast, over-aggressive on big coefficients you actually need.
2. **Near zero**, flat, no gradient pushing to zero (see above).

> "If a $\beta$ should genuinely be 10, then $\beta^2 = 100$ is a huge contribution to the total objective, ridge will pull it down, introducing bias precisely when one variable should *dominate* the model." - [[L13-modelsel-2]]

### Bias-variance reading of the lambda sweep

- $\lambda \approx 0$: high variance, low bias (= OLS).
- $\lambda \to \infty$: lowest variance (zero, perfect repeatability!), worst bias.
- Sweet spot in the middle, found by [[cross-validation]]. *"Variance can shrink to about half of the OLS variance for the cost of a small bias bump"* - [[L13-modelsel-2]].

> "We don't typically know what the bias and the variance are. These are just things we think about. If we knew what the bias was, we wouldn't be fitting a model, we would just use the true model and we wouldn't do any statistics, because that would be stupid." - [[L13-modelsel-2]]

### Reading the trace plot

Slide showed standardized $\hat\beta_j(\lambda)$ vs $\lambda$ (or vs $\|\hat\beta(\lambda)\|/\|\hat\beta(0)\|$):
- **Far left** ($\lambda \approx 0$): $\hat\beta$'s sit at OLS values, large in magnitude, mixed signs.
- **Far right** ($\lambda$ very large): all $\hat\beta$'s squashed near zero.
- **In between**: smooth shrinkage paths, **never crossing zero** (this distinguishes a ridge trace from a lasso trace).

### Choosing $\lambda$

> "Pick $\lambda$ for which the cross-validation error is smallest. Re-fit using all of the available observations and the selected value of $\lambda$.", slide deck

Standard recipe (per [[L14-modelsel-3]]): grid-search $\lambda$ via $k$-fold CV → pick the $\lambda$ minimizing CV error → refit on full data with that $\lambda$. Some prefer the [[one-standard-error-rule|one-SE rule]] (largest $\lambda$ within 1 SE of the CV minimum) for extra parsimony, both 2024 and 2025 past exams used 1-SE.

### PCR ≈ a discretized ridge

A deeper conceptual link the prof drew explicitly in [[L15-modelsel-4]]:

> "PCR can be seen as a discretized version of ridge regression. Ridge regression encourages ties, if two things explain the same thing, then have them share the weight. … PCR is doing it more abruptly because it simply says okay, direction, direction, direction, new axes, new data, everything that's shared go here. Ridge regression does it more continuously."

Both shrink the small-eigenvalue directions; ridge does it smoothly via $\lambda$, [[principal-component-regression|PCR]] does it abruptly by truncating components. Slide gave the per-PC ridge shrinkage factor as $\lambda_j^2/(\lambda_j^2 + \lambda)$, heavier shrinkage on smaller eigenvalues.

## Exam signals

> "I would argue this is the most important one that we talk about throughout, the most important form of this type of parameter selection." - [[L14-modelsel-3]] (regularization in general)

> "Worth learning how to interpret this geometric interpretation of ridge and lasso." - [[L14-modelsel-3]]

> "Importantly, ridge regression is **not scale-invariant**." - [[L12-modelsel-1]]

> "Ridge regression are not scale-invariant. The standard least square are scale-invariant.", slide deck

The prof returned to ridge in lecture after lecture (L04 implicit, L12–L15 explicit, L24 weight decay), high-recurrence content. The 2024 and 2025 exams asked directly: "Carry out Ridge regression… choose the largest $\lambda$ within 1 SE of the CV minimum… report the MSE… compare coefficients with OLS / lasso." 2026 will be the open-book interpretation analogue.

## Pitfalls

- **Forgetting to standardize.** Ridge is not scale-invariant. *"If one had a standard deviation of like a million, then that will be your strongest variable."* - [[L14-modelsel-3]] / [[L15-modelsel-4]]. Diagnostic: weird coefficient magnitudes, one variable dominates → check standardization.
- **Penalizing the intercept.** Slide-flagged: don't. *"If we included the intercept, $\beta^R$ would depend on the average of the response."*, slide deck.
- **Expecting variable selection.** Ridge never zeros a coefficient. If interpretability ("which 3 of 6 matter?") is the goal, use [[lasso]] or [[subset-selection|subset selection]].
- **Confusing direction of $\lambda$.** $\lambda \uparrow$ → more shrinkage → simpler model → less variance, more bias. $\lambda \downarrow 0$ → recover OLS. (Easy MC trap: which direction does $\lambda$ shrink?)
- **Reading "ridge requires $p < n$"**: false. Ridge works fine when $p > n$. *"Lasso requires that $p < n$"* was a wrong-answer option on the 2024 exam; ridge being possible at $p > n$ was the right one.

## Scope vs ISLP

- **In scope:** the L2 objective formula; the tug-of-war intuition; standardization requirement; intercept not penalized; "doesn't go to zero, no variable selection"; works when $p \geq n$; $\lambda$ chosen by CV; bias-variance behaviour across $\lambda$; the geometric picture and the socialist/capitalist framing (see [[ridge-vs-lasso-geometry]]); the PCR-as-discretized-ridge analogy.
- **Look up in ISLP:** §6.2.1 (pp. 252–256) for the closed-form derivation $\hat\beta^R = (X^\top X + \lambda I)^{-1} X^\top y$ and the special-case orthonormal design where ridge's per-coefficient shrinkage factor is $\hat\beta_j / (1 + \lambda)$ (§6.2.2 "Simple Special Case", pp. 269–270). Use ISLP for full derivations and the simulated bias-variance figure.
- **Skip in ISLP (prof-excluded):** the **Bayesian interpretation** of ridge as the posterior mode under a Gaussian prior (§6.2.2 pp. 271–273). *"I really don't think I'd put this on the test, just because it kind of assumes a lot of knowledge that maybe you don't have."* - [[L14-modelsel-3]]. The conceptual analogy ("ridge ↔ Gaussian prior, lasso ↔ Laplace prior") is OK to know; the algebra is not.

## Exercise instances

- Exercise6.5, apply ridge to the Credit dataset, compare coefficients and prediction with OLS. Standard `cv.glmnet(alpha=0)` workflow: standardize, CV-select $\lambda$, refit, compare coefficient table.

(Note: the slides label this "Recommended exercise 5"; CE1 doesn't use ridge.)

## How it might appear on the exam

- **Coefficient table interpretation.** Given a printout of OLS vs ridge vs lasso coefficients on the same dataset, explain the patterns: ridge shrinks *all* coefficients toward zero but none to exactly zero; lasso zeros some out (this exact comparison appeared in the 2024 exam answer key).
- **Multiple choice / true-false**: "Ridge regression can be fit when $p > n$." → True. "Ridge sets some coefficients to exactly zero." → False.
- **Trace plot reading**: given a coefficient-vs-$\log(\lambda)$ plot, identify the OLS end ($\lambda \to 0$, coefficients large) and the all-shrunken end ($\lambda \to \infty$, coefficients near zero); note that no path crosses zero (vs lasso, where they do).
- **CV plot reading**: given a CV-MSE-vs-$\lambda$ curve with `lambda.min` and `lambda.1se` marked, explain the [[one-standard-error-rule]] choice.
- **Conceptual / pseudocode**: "How would you tune $\lambda$?" → standardize $X$ → grid of $\lambda$ values → $k$-fold CV on each → pick min-CV (or 1-SE) → refit on full data.
- **Bias-variance question**: "What happens to bias and variance as $\lambda$ increases?" → bias up, variance down; test MSE U-shaped.
- **Method comparison**: ridge vs lasso vs PCR, when would you prefer each? Ridge when many small contributors and you don't need interpretability; lasso when you want sparsity; PCR when there's strong multicollinearity and the response lives in the high-variance directions.

## Related

- [[lasso]]: the L1 cousin; does variable selection; "capitalist" to ridge's "socialist."
- [[ridge-vs-lasso-geometry]]: the side-by-side ellipse-meets-ball-vs-diamond picture explaining why ridge doesn't sparsify.
- [[elastic-net]]: combines ridge + lasso; ridge's correlated-variable averaging plus lasso's sparsity.
- [[principal-component-regression]]: "discretized ridge"; both shrink small-eigenvalue directions.
- [[bias-variance-tradeoff]]: the lens through which the prof motivates ridge.
- [[cross-validation]]: how $\lambda$ is chosen.
- [[regularization]]: the cross-cutting Special; ridge is the canonical instance.
- [[standardization]]: required preprocessing.
- [[high-dimensional-regression]]: ridge is one of the few OLS substitutes that survives $p > n$.
- [[collinearity]]: ridge handles correlated predictors by averaging.
- [[double-descent]]: implicit ridge in the over-parameterized regime ([[L04-statlearn-3]]).
