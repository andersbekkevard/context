---
concept: collinearity
module: 03-linreg
lectures: [L06, L08, L14, L15]
isl-ref: 3.3.3
exercises: []
related: [linear-regression, design-matrix-and-hat-matrix, sampling-distribution-of-beta, t-test-and-significance, ridge-regression, principal-component-regression, high-dimensional-regression]
tags:
  - concept
  - module/03-linreg
aliases:
  - multicollinearity
---

# Collinearity / multicollinearity

When two or more predictors are correlated, $\mathbf{X}^\top\mathbf{X}$ becomes near-singular, $(\mathbf{X}^\top\mathbf{X})^{-1}$ blows up, and the variance of $\hat{\boldsymbol\beta}$ explodes. Coefficients **trade off** against each other; SEs go up; significance disappears even when the joint relationship is strong. The cleanest fix is to drop a variable, or use [[ridge-regression]] / [[principal-component-regression|PCR]] to escape the singularity.

## Definition (prof's framing)

> "Some of the predictors $x_1, x_2, \ldots$ are themselves correlated. … We could trade between $x_1$ and $x_2$ — e.g., make $\beta_1$ bigger and $\beta_2$ smaller, while fit is similar." — [[L08-classif-2]]

Perfect collinearity → infinitely many least-squares solutions ($\mathbf{X}^\top\mathbf{X}$ exactly singular). Mild collinearity → highly sensitive solution that swings around with tiny data perturbations.

> "Going to go wah, wah." — [[L08-classif-2]]

> "Predictions blow up out of sample because you end up with 'a million minus a million.'" — paraphrase from [[L08-classif-2]]

## Notation & setup

- Collinearity is a property of $\mathbf{X}$ — independent of $\mathbf{y}$.
- Source: $\mathrm{Cov}(\hat{\boldsymbol\beta}) = \sigma^2 (\mathbf{X}^\top\mathbf{X})^{-1}$. As columns of $\mathbf{X}$ become correlated, eigenvalues of $\mathbf{X}^\top\mathbf{X}$ approach zero; corresponding eigenvalues of the inverse $\to \infty$.
- The **variance inflation factor** (VIF) measures the explosion per coefficient — flagged as self-study by the prof, **not on the exam**.

## Insights & mental models

### Why this is the inevitable problem

The prof keeps coming back to it because it's where the algebra of OLS first cracks:

> "I think this is really why statisticians love these distributions, because you can read out what's going to happen when you look at them. You can be like — ah, that X transpose X is going to screw us later." — [[L06-linreg-2]]

He flagged it in [[L06-linreg-2]] before ever getting to a worked example, then returned to it in [[L08-classif-2]] (one slide), then again in [[L14-modelsel-3]] / [[L15-modelsel-4]] when introducing PCR — collinearity is the *reason* PCR exists.

### Symptoms in the regression output

- Estimated coefficients have **huge SEs** even when the joint contribution of the variables is highly significant (large F, individually insignificant t's).
- Coefficient signs and magnitudes are unstable across resamples — refit on a different subset of data and they swing wildly.
- Adding or removing one variable dramatically changes the others' coefficients.
- Predictions on new data are wildly off because the trade-off "a million minus a million" evaporates with any small shift.

### Connection to the t-test

This is *why* the F-test exists:

> "The variables can actually be correlated, and then none of them actually look significant, but overall the test is very significant." — [[L06-linreg-2]]

So skipping Q1 (F-test on all coefficients) and going straight to per-coefficient t-tests can hide a real signal that's spread across correlated predictors. See [[f-test]] and [[t-test-and-significance]].

### Pathological in $p > n$

When $p > n$, $\mathbf{X}^\top\mathbf{X}$ is **always** singular regardless of correlation — collinearity becomes total. This is the regime that motivates module 6 (regularization / dimensionality reduction). See [[high-dimensional-regression]].

### Fixes

The prof's menu:

- **Drop a variable.** Cheapest, often the right answer.
- **Combine** the collinear ones (e.g., average them, take a difference). Domain-knowledge driven.
- **PCA / [[principal-component-regression|PCR]].** The "compress your variables into fewer variables with some loss" route — replaces the correlated columns with orthogonal principal components, killing the collinearity directly.

> "PCA is a way of compressing your variables into fewer variables with some loss. … in this case $X_1$ and $X_2$ would turn into… one would be this trend, basically, and then the other one would be the one that's moving around — the shit around it. … because then they all become orthogonal." — [[L08-classif-2]]

- **[[ridge-regression]] (L2).** Adds $\lambda\mathbf{I}$ to $\mathbf{X}^\top\mathbf{X}$ before inverting → guaranteed invertible, finite-variance coefficients. The standard fix.
- **LDA as dimensionality reduction.** Brought up in passing as the second answer to the collinearity question — "another route" — see [[L08-classif-2]].

## Exam signals

> "This factor X transpose X comes into play in particular when two variables are basically the same — because then they can trade off each other and then this variance explodes. That's a thing we discuss at the very end of today. It's called collinearity." — [[L06-linreg-2]]

> "Only checking individual p-values is dangerous. … The variables can actually be correlated, and then none of them actually look significant, but overall the test is very significant." — [[L06-linreg-2]]

> "The collinearity problem we talked about a minute ago, that can happen here [logistic regression], and then this thing's no longer having a single maximum and it gets weird." — [[L08-classif-2]]

> The 2025 Q4 polynomial trap: model B with $\beta_2 x^2 + \beta_3 x^2$ (collinear quadratic terms) — *"if you try to fit this model, your optimizer goes, 'hey, no, this sucks.' Adding L2 makes $\beta_2 = \beta_3$."* — [[L27-summary]]

## Pitfalls

- **High individual p-values, low joint p-value.** Classic collinearity signature. *Always* run the F-test before drilling into individual t-tests.
- **Stable predictions, unstable coefficients.** A collinear regression often predicts $\hat y$ well — because the *sum* $\beta_1 x_1 + \beta_2 x_2$ is well-determined even when each coefficient isn't. So if you only care about prediction, you may not feel the symptoms. If you care about *interpretation* of which predictor matters, you're in trouble.
- **K dummies for K-level factor.** Perfect collinearity. Identifiability fails — $\mathbf{X}^\top\mathbf{X}$ singular. Always use $K - 1$ dummies plus a reference category (see [[categorical-encoding-and-interactions]]).
- **High correlation ≠ collinearity.** Two predictors can be highly correlated without breaking $\mathbf{X}^\top\mathbf{X}$ irreparably. The threshold for "trouble" depends on $n$ and $\sigma^2$; rule of thumb VIF > 5 or 10 (but VIF is *self-study* per the prof).
- **Standardize before diagnosing.** Numerical near-singularity can come from scale differences across columns. Standardize first when investigating.

## Scope vs ISLR

- **In scope:** the qualitative story — what collinearity is, how it shows up in the SE, why it makes coefficients unstable, the connection to $\mathbf{X}^\top\mathbf{X}$ inverse, and the menu of fixes (drop, combine, PCR, ridge).
- **Look up in ISLR:** §3.3.3 (pp. 99–102, *Collinearity*) — Credit-card example with `limit` and `rating`, the VIF formula. The book's level of detail matches the prof's coverage.
- **Skip in ISLR (book-only / prof excluded):** the VIF formula and computation — [[L08-classif-2]] explicit "read it as self-study"; not exam material. Condition number, eigen-decomposition diagnostics — [[L04-statlearn-3]] deferred. Bayesian interpretation of ridge-as-prior — [[L14-modelsel-3]] explicit "I really don't think I'd put this on the test."

## Exercise instances

None directly tagged for collinearity in module 3. The concept reappears in module 6 — Exercise 6.5 (ridge on Credit) and Exercise 6.6 (lasso on Credit) — but those are owned by the [[ridge-regression]] / [[lasso]] atoms.

## How it might appear on the exam

- **Identify symptoms in regression output.** Given a table with two correlated predictors (e.g., `limit` and `rating` in Credit), spot the high SEs and low individual significance, then explain why they happen.
- **F-test vs t-test under collinearity.** Explain why the F-test can be highly significant while individual t's are not.
- **What's the fix?** Multiple-choice or short-answer: drop a variable / use ridge / use PCR. State the principle (orthogonalize, regularize, or eliminate).
- **The 2025 Q4 polynomial trap.** Two collinear $x^2$ terms; what does L2 do? (Forces $\beta_2 = \beta_3$; the optimizer otherwise has infinite valley.)
- **$p > n$.** Conceptual: why does OLS break? Because $\mathbf{X}^\top\mathbf{X}$ is necessarily singular. Need regularization to make the problem well-posed.

## Related

- [[linear-regression]] — the model where collinearity bites
- [[design-matrix-and-hat-matrix]] — $(\mathbf{X}^\top\mathbf{X})^{-1}$ is what blows up
- [[sampling-distribution-of-beta]] — variance covariance inflated by collinearity
- [[t-test-and-significance]] — individual t's lose power
- [[ridge-regression]] — the canonical regularization fix
- [[principal-component-regression]] — orthogonalize the predictors directly
- [[high-dimensional-regression]] — the $p > n$ regime where collinearity is total
