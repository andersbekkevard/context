---
concept: gaussian-error-assumptions
module: 03-linreg
lectures: [L05, L06, L08]
isl-ref: 3.1.2, 3.3.3
exercises:
  - Exercise3.1e - autoplot diagnostic plots, comment on outliers and leverage
  - Exercise3.1h - try X transformations to fix residual issues
  - Exercise3.2e - error vs residual definitions and properties
related: [linear-regression, residual-diagnostics, least-squares-and-mle, sampling-distribution-of-beta, collinearity]
tags:
  - concept
  - module/03-linreg
aliases:
  - linear regression assumptions
  - independence assumption
---

# Gaussian-error assumptions

The five assumptions OLS / inference / CIs / tests all rest on. The prof's repeated drumbeat: **assumptions (4) and (5), independence, are the dangerous ones.** "Violations ruin everything." Spatial / temporal correlation is the canonical failure mode.

## Definition (prof's framing)

For each pair $(x_i, y_i)$ the error term $\varepsilon_i$ satisfies:

1. **Normally distributed.** $\varepsilon_i \sim N(\cdot, \cdot)$.
2. **Mean zero.** $\mathsf{E}(\varepsilon_i) = 0$.
3. **Common variance** (homoscedastic). $\mathsf{Var}(\varepsilon_i) = \sigma^2$, not depending on $x$.
4. **Independent of any other variable** ("independent of other shit").
5. **Independent of each other** ($\varepsilon_i \perp \varepsilon_j$).

In matrix form: $\boldsymbol\varepsilon \sim N_n(\mathbf{0}, \sigma^2 \mathbf{I})$, diagonal $\sigma^2$, zeros elsewhere.

> "These two are the two ones that are so easily violated that they're independent of other things and they're independent of each other… violating these is super common and ruins everything." - [[L05-linreg-1]]

> "I think this is the assumption that more often we accidentally screw up and it screws up a lot of shit and people are like 'that's fine.' It's not." - [[L08-classif-2]]

## Notation & setup

- $\boldsymbol\varepsilon$ = error vector, $n$-dim multivariate normal under the classical setup.
- $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}$ = residual vector, the *prediction* of $\boldsymbol\varepsilon$ from data. Distinct from the unobservable error.
- The classical setup adds: $\mathrm{rank}(\mathbf{X}) = p+1$ and $n \gg p+1$ (so the inverse exists and we have data to spare).

## Insights & mental models

### Why these matter

The whole inferential apparatus, sampling distribution of $\hat{\boldsymbol\beta}$ ([[sampling-distribution-of-beta]]), t-tests, F-tests, CIs, PIs, derives from these. Break them and the *point estimates* are still computed (the algebra works), but the *uncertainty* claims you report are wrong.

### Tier list of dangerousness (prof's own)

> "I don't think those [Gaussian / zero mean / common variance] are so bad. They're so common… I don't think that screws up so much." - [[L05-linreg-1]]

So in the prof's ranking:
- **Most dangerous:** independence (4) + (5). Concrete failure mode he flagged: temperature samples over time → neighboring time bins are correlated → sampling more finely makes the relationship *look* stronger because effective sample size is inflated. He confessed his own third paper had wrong numbers because he assumed independence when he shouldn't have.
- **Less dangerous, often robust:** Gaussian, zero-mean, homoscedasticity. The CLT cushions Gaussian; centering handles zero-mean; mild heteroscedasticity rarely changes inference qualitatively.

### Independence trap recurs in module 5

> "If you just randomly sort them into the test and fit and validation without any concern of these things… you've just partitioned the same data twice, basically." - [[L10-resample-1]]

Same independence assumption, now broken at the cross-validation level. The fix: chunk by the dependency dimension before splitting.

### Generative picture

Imagine drawing the line, then sampling points around it with $N(0, \sigma^2)$ noise. Most points fall near the line; the histogram of vertical deviations approximates a Gaussian. You "never" get points far out, only with probability set by the tail area. When you violate this (an outlier appears), the model contorts to keep the outlier inside its assumed distribution, and the whole fit goes bad.

## Exam signals

> "These two are the two ones that are so easily violated that they're independent of other things and they're independent of each other… violating these is super common and ruins everything." - [[L05-linreg-1]]

> "I think this is the assumption that more often we accidentally screw up and it screws up a lot of shit and people are like 'that's fine.' It's not." - [[L08-classif-2]]

> "These issues will permeate any model you want to do." - [[L06-linreg-2]]

## Pitfalls

- **Independence is silently broken.** Spatial data, temporal data, repeated measures from the same subject, related individuals, all violate (5). Effective sample size is smaller than $n$, so reported significance is "horseshit" (prof's term).
- **Outliers.** A single far point pulls the regression line because LS is quadratic-cost; the fit "contorts" to accommodate it. See [[residual-diagnostics]] and the leverage discussion.
- **Heteroscedasticity (variance scaling with $x$).** Spot via residuals-vs-fitted plot; the cloud fans out.
- **Non-normal errors.** Symptom on the QQ plot, S-shape or fat tails. Inferential statements (t-tests, CIs) are the most affected; point estimates still unbiased.
- **Errors vs residuals confusion.** Errors $\varepsilon_i$ are random and unobservable; residuals $e_i$ are observed predictions of them. Raw residuals have $\mathrm{Cov}(\mathbf{e}) = \sigma^2(\mathbf{I} - \mathbf{H})$, slightly correlated and unequal variance. Standardize / studentize them to make diagnostics behave (see [[residual-diagnostics]]).

## Scope vs ISLP

- **In scope:** all five assumptions, the prof's tier list (independence ≫ rest), the recurring "you can't escape it" theme, what each diagnostic checks.
- **Look up in ISLP:** §3.1.2 for the Gaussian assumption; §3.3.3 for the "potential problems" enumeration (pp. 92-104): the book lists the same six symptoms (non-linearity, error correlation, non-constant variance, outliers, high-leverage, collinearity).
- **Skip in ISLP (book-only / prof excluded):** formal hypothesis tests for normality (Shapiro–Wilk etc.) - [[L08-classif-2]]: "we're not going to talk about it." VIF formula (self-study).

## Exercise instances

- Exercise3.1e: autoplot diagnostic plots on the Auto fit; comment on outliers and leverage (which assumptions are flagged?)
- Exercise3.1h: try $\log X$, $\sqrt X$, $X^2$ to fix residual issues, concretely what the violations look like and how to repair
- Exercise3.2e: explain difference between error and residual; properties of raw residuals; why don't we use raw residuals; what's our solution

## How it might appear on the exam

- **Identify the assumption from a diagnostic plot.** Show a residuals-vs-fitted with a clear pattern (curvature → linearity broken; fanning → heteroscedasticity; clusters → independence broken). Ask: which assumption is violated?
- **Why we standardize residuals.** Raw residuals have unequal variance and weak correlation under the classical setup, *standardizing* divides by $\hat\sigma\sqrt{1-h_{ii}}$ to make them resemble the assumed $\varepsilon_i$. See [[residual-diagnostics]].
- **What breaks if independence fails?** SE estimates are wrong → CIs are wrong → p-values are wrong. Point estimates still unbiased; inference is the casualty.
- **Error vs residual definition.** Common short-answer or true/false trap: residuals are observed predictions, errors are unobservable random variables.

## Related

- [[linear-regression]]: the model the assumptions belong to
- [[residual-diagnostics]]: how to check them
- [[least-squares-and-mle]]: uses (1)-(3) to make MLE = LS work
- [[sampling-distribution-of-beta]]: derives from these assumptions
- [[collinearity]]: the multivariate "rank" assumption
- [[cross-validation]]: where the independence trap reappears at the validation level
