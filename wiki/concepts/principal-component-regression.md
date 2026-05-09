---
concept: principal-component-regression
module: 06-modelsel
lectures: [L08, L14, L15]
isl-ref: 6.3.1
exercises:
  - Exercise6.7 — how many PCs to use for Credit (justify cutoff)
  - Exercise6.8 — apply PCR on Credit, compare to other module 6 methods
related: [principal-component-analysis, ridge-regression, partial-least-squares, lasso, dimensionality-reduction, standardization, collinearity, explained-variance-and-scree-plot, cross-validation]
tags:
  - concept
  - module/06-modelsel
aliases:
  - PCR
---

# Principal components regression (PCR)

The third of module 6's three families (after [[subset-selection]] and [[regularization|shrinkage]]): **transform** the $p$ correlated predictors into $M < p$ orthogonal composite predictors (the principal components) via [[principal-component-analysis|PCA]], then fit standard OLS on those. The prof's headline framing: **PCR is a discretized [[ridge-regression]]** — both shrink the small-eigenvalue directions, but PCR does it abruptly (truncate) while ridge does it smoothly. *"PCR can be seen as a discretized version of ridge regression."* — [[L15-modelsel-4]].

## Definition (prof's framing)

> "Principal Components Regression involves: constructing the first $M$ principal components $Z_1, \dots, Z_M$, [and] using these components as the predictors in a standard linear regression model." — slide deck (selection_regularization_presentation_lecture2.md)

The pipeline:

> "Standardize $X$, run PCA, get $Z_1, \dots, Z_p$. Fit a standard linear regression of $Y$ on $Z_1, \dots, Z_M$, where $M$ is a tuning parameter. Sweep $M$ from 1 up to $p$. Pick the $M$ that minimizes CV-MSE." — [[L15-modelsel-4]]

> "You're taking the X, you're squishing it down to fit your model, and then you go backwards to the original model again." — [[L14-modelsel-3]]

## Notation & setup

- $X$: $n \times p$ design matrix (standardized — see below).
- $Z_m = \sum_j \phi_{jm} X_j$: $m$-th principal component (linear combination of original $X$'s, with $\sum_j \phi_{jm}^2 = 1$, ordered by decreasing variance, mutually orthogonal). See [[principal-component-analysis]].
- $M < p$: number of PCs retained — the **PCR tuning parameter**, chosen by [[cross-validation]].
- $\theta_m$: regression coefficients of $Y$ on the $Z_m$'s.
- $\beta_j$ (back-transformed): coefficient on the original $X_j$, recovered as $\beta_j = \sum_{m=1}^M \theta_m \phi_{jm}$.

## Formula(s) to know cold

**Component construction:**
$$Z_m = \sum_{j=1}^p \phi_{jm} X_j, \quad \text{subject to} \quad \sum_{j=1}^p \phi_{jm}^2 = 1$$

**Reduced regression:**
$$y_i = \theta_0 + \sum_{m=1}^M \theta_m z_{im} + \varepsilon_i$$

**Back-transform to original-X coefficients (slide-flagged):**
$$\beta_j = \sum_{m=1}^M \theta_m \phi_{jm}$$

This last formula is what the slides labeled the "constrained interpretation": dimension reduction *constrains* the coefficients of a standard linear regression to live in the $M$-dimensional subspace spanned by $\{\phi_{\cdot 1}, \dots, \phi_{\cdot M}\}$.

**Fraction of $X$-variance captured by first $M$ PCs (slide-flagged):**
$$R^2_\text{PCA} = \frac{\sum_{i=1}^M \lambda_i}{\sum_{j=1}^p \lambda_j}$$
where $\lambda_i$ are eigenvalues of $\hat\Sigma_X$ (= variances of the corresponding PCs).

## The pipeline (canonical procedure)

1. **Standardize** $X$ (mean 0, sd 1 per column). PCA is not scale-invariant.
2. **Run PCA** on the standardized $X$ → get $Z_1, \dots, Z_p$ in decreasing-variance order.
3. **Sweep $M$** from 1 to $p$. For each $M$:
   - Fit OLS of $Y$ on $Z_1, \dots, Z_M$ → get $\hat\theta$'s.
   - Compute CV-MSE.
4. **Pick $M$** at the CV-MSE minimum (or by the [[one-standard-error-rule|1-SE rule]] / explained-variance threshold).
5. **(Optional) back-transform** $\hat\theta$'s to $\hat\beta$'s in original-$X$ space via $\hat\beta_j = \sum_m \hat\theta_m \phi_{jm}$.

## Insights & mental models

### Why bother — the multicollinearity angle

PCR's main reason to exist: handling [[collinearity]].

> "If two $X_j$'s are nearly identical (or strongly correlated more generally — multicollinearity), the OLS fit can't tell them apart. Parameters can trade off of each other, which is bad… Squishing to orthogonal $Z$'s removes the redundancy. Each $Z_m$ is independent of the others by construction. Cleaner fit, lower variance." — [[L14-modelsel-3]]

Three ways the course offers to handle multicollinearity (per [[L14-modelsel-3]]):
- **L1 (lasso)**: pick one of the correlated variables, zero the others.
- **L2 (ridge)**: hold both back, share the load.
- **PCA / PCR**: rotate to an orthogonal basis where the correlation is gone by construction.

### PCR ≈ a discretized ridge — the load-bearing analogy

The single most important conceptual point in this segment, from [[L15-modelsel-4]]:

> "PCR can be seen as a discretized version of ridge regression. Ridge regression encourages ties — if two things explain the same thing, then have them share the weight. The point where they share is actually very similar to getting a principal component that captures the part that both share. So PCR is doing it more abruptly because it simply says okay, direction, direction, direction, new axes, new data, everything that's shared go here. Ridge regression does it more continuously."

Both methods place pressure on the **least important** principal directions:
- **PCR**: drop them outright (those past the cutoff $M$). Hard threshold.
- **Ridge**: shrink them most. Soft threshold. Slide gives the per-PC shrinkage factor as $\lambda_j^2 / (\lambda_j^2 + \lambda)$ — heavier shrinkage on smaller eigenvalues.

> "Higher pressure on less important PCs. PCR discards the $p - M$ smallest eigenvalue components." — slide deck

### The two big PCR assumptions (both can fail)

Slide-flagged:

> "Key assumptions: A small number of principal components suffice to explain (1) most of the variability in the data, [and] (2) the relationship with the response. … The assumptions above are not guaranteed to hold in every case. This is true specially for assumption 2 above. Since the PCs are selected via unsupervised learning." — slide deck

The critical drawback verbatim from [[L15-modelsel-4]]:

> "There's no guarantee that the directions that best explain the predictors will also be the best directions to use for predicting the response."

PCR's bet: the high-variance directions in $X$-space are also the directions $Y$ depends on. When that holds, PCR is great. When the response is driven by *low-variance* directions (e.g., a single rare-but-predictive feature), PCR misses it because it threw that direction away. [[partial-least-squares|PLS]] is the supervised fix — uses $Y$ to choose directions.

### PCR is **not** a variable-selection method

> "Just like Ridge, it doesn't actually select the parameters — it gives you components, and each component is a combination of the original axes." — [[L15-modelsel-4]]

> "Use PCR/ridge when you want predictive power and don't care which raw variables drive it. Use lasso or subset selection when you need 'this or that one.'" — [[L15-modelsel-4]]

After back-transforming, all $p$ original-$X$ coefficients $\hat\beta_j$ are typically nonzero (since each PC is a combination of all $X$'s).

### Bias / variance / test-error reading along $M$

The CV-MSE-vs-$M$ curve is U-shaped (similar in shape to ridge's CV-vs-$\log\lambda$):
- **Bias** drops fast at small $M$, then plateaus (more PCs → more parameters → already low bias).
- **Variance** rises with $M$ (more parameters, more noise fitting).
- **Test MSE** dips through a minimum where bias and variance cross-trade.

> "Of the many solutions that have a low bias, the mean squared error on the held out data is in a similar location to where the variance is minimized." — [[L15-modelsel-4]]

### Credit-data PCR result

> "With $M=1$ the original-variable betas are tiny — PC1 didn't load heavily on the variables that drive $Y$. A big jump at $M=9$: that's where income finally enters strongly. Income just happened not to vary as much or wasn't as correlated as other things. … CV-MSE was decreasing slowly, then dropped massively from $M=9$ to $M=10$. Settle on $M = 10$." — [[L15-modelsel-4]]

Slide line: *"The lowest cross-validation error occurs when there are $M = 10$ components — almost no dimension reduction at all."* On the Credit dataset PCR did **not** outperform other methods because the response-relevant direction (income) had middling $X$-variance.

> "Even though the things that you care about were just ended up being four variables, primarily — it took you 10 PCs to get there." — [[L15-modelsel-4]]

This is a real-world cautionary tale for assumption (2) above.

### PCR generalizes beyond linear PCA

> "The PCR pattern — compress $X$ down to fewer features with some method, then regress — generalizes far beyond PCA. Example: video frames as $X$. They're huge. Run them through a learned feature extractor (a neural net) to get a small vector per frame, then regress $Y$ on that compressed representation." — [[L15-modelsel-4]]

Same skeleton, nonlinear compressor. (Conceptual bridge to [[dimensionality-reduction]] more broadly.)

## Exam signals

> "There's no guarantee that the directions that best explain the predictors will also be the best directions to use for predicting the response." — [[L15-modelsel-4]]

> "PCR can be seen as a discretized version of ridge regression." — [[L15-modelsel-4]] (echoed verbatim on slide)

> "Just like Ridge, it doesn't actually select the parameters." — [[L15-modelsel-4]] (slide also: *"Similar to Ridge, PCR does not perform feature selection — PCs are linear combination of all predictors."*)

> "PCA is not scale invariant. … Standardize all the $p$ variables before applying PCA." — slide deck

The PCR ↔ ridge analogy is the headline conceptual point. The PCR-as-not-variable-selection contrast with [[lasso]] is exam-flavoured. PCR was on the 2024 exam (called via `library(pls)`).

## Pitfalls

- **Forgetting to standardize.** PCA / PCR are not scale-invariant. *"PCA is not scale invariant. So if you don't standardize them so that their mean is zero and their variance is one, then if one had a standard deviation of like a million, then that will be your strongest variable."* — [[L14-modelsel-3]].
- **Thinking PCR selects variables.** It doesn't — every back-transformed $\hat\beta_j$ is typically nonzero.
- **Trusting PCR when assumption (2) fails.** If $Y$ depends on a low-$X$-variance direction, PCR throws it away and underperforms. The Credit example illustrates: needed $M = 10$ of $\sim 11$ to capture income — almost no dimension reduction achieved.
- **Confusing $M$ with $p$.** $M$ is the number of PCs *kept*; $p$ is the total number of original predictors. PCR's appeal is when $M \ll p$.
- **Reading PCR coefficients as variable importance.** They're back-projected from the $\theta$'s — they describe the regression but are not interpretable as "predictor X matters this much."

## Scope vs ISLR

- **In scope:** the PCR pipeline (standardize → PCA → fit → back-transform); the back-transform formula $\beta_j = \sum_m \theta_m \phi_{jm}$; choosing $M$ by CV; the two key assumptions and their failure modes; the PCR-as-discretized-ridge analogy; the Credit-data result; the contrast with [[lasso]] (PCR does not select variables); the $R^2_\text{PCA}$ formula via eigenvalues.
- **Look up in ISLR:** §6.3.1 (pp. 281–286) for the PCR algorithm and the simulated PCR-vs-ridge-vs-lasso comparison figure. §10 for full PCA treatment.
- **Skip in ISLR:** the SVD derivation of PCA (the prof noted full PCA mechanics live in chapter 10; he didn't redo them in module 6); detailed ridge-shrinkage-per-PC algebra (slide gave the $\lambda_j^2/(\lambda_j^2 + \lambda)$ factor, the prof noted *"this is a confusing figure, I'll try to make another one for next time"* — [[L15-modelsel-4]]).

## Exercise instances

- Exercise6.7 — How many principal components should we use for the Credit dataset? Justify. (Read the explained-variance / CV-MSE curve, defend a cutoff.)
- Exercise6.8 — Apply PCR on the Credit dataset and compare with the methods covered in Lecture 1 (OLS / [[subset-selection]] / [[ridge-regression]] / [[lasso]]).

(Slide labels these "Recommended exercise 7" and "Recommended exercise 11" respectively. Note: the slide-deck-labeled "Recommended exercise 11" maps to RecEx6 problem 8 in the actual exercise file.)

## How it might appear on the exam

- **Output interpretation**: given a CV-MSE-vs-$M$ curve and / or a PVE / cumulative-variance plot, identify the optimal $M$ and explain reasoning (CV minimum or 90/95% variance threshold).
- **Method comparison** (recurring across past exams): PCR vs ridge vs lasso vs OLS — given a results table, explain which performed best and why. PCR ≈ ridge is the canonical answer; lasso wins if the truth is sparse.
- **True / false**: "PCR performs variable selection." → False. "PCR can be seen as a discretized version of ridge regression." → True. "PCR uses the response $Y$ to choose components." → False (PCA is unsupervised — that's the contrast with [[partial-least-squares|PLS]]).
- **Conceptual short answer**: "What is PCR's main weakness compared with [[partial-least-squares|PLS]]?" → PCA picks high-variance directions in $X$-space without consulting $Y$, so if $Y$ depends on a low-variance $X$-direction PCR misses it; PLS uses $\text{Cov}(X, Y)$ to choose directions instead.
- **Pseudocode / equation-writing**: "Write out the PCR procedure." → standardize → PCA → CV over $M$ → fit OLS on first $M$ PCs → back-transform.
- **Hand calculation (light)**: given eigenvalues, compute fraction of variance explained by first $M$ PCs via $\sum_{i=1}^M \lambda_i / \sum_j \lambda_j$.

## Related

- [[principal-component-analysis]] — the unsupervised compression PCR rides on (deep treatment in module 10).
- [[ridge-regression]] — PCR's smooth-shrinkage analogue; "PCR ≈ discretized ridge."
- [[partial-least-squares]] — supervised cousin of PCR; uses $Y$ to choose directions.
- [[lasso]] — the variable-selecting alternative; opposite philosophy from PCR.
- [[dimensionality-reduction]] — the broader family PCR belongs to; the same skeleton ("compress, then regress") generalizes to nonlinear compressors.
- [[standardization]] — required preprocessing.
- [[collinearity]] — PCR's main reason to exist.
- [[explained-variance-and-scree-plot]] — how PCA picks the "right" $M$ (variance threshold vs CV-MSE).
- [[cross-validation]] — for choosing $M$ honestly.
