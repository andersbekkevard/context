---
concept: residual-diagnostics
module: 03-linreg
lectures: [L06, L08]
isl-ref: 3.3.3
exercises:
  - Exercise3.1e — autoplot diagnostic plots, comment on residuals and leverage
  - Exercise3.1f — simulate normal data, see how "wonky" QQ plots can look even under correct assumptions
  - CE1 problem 2e — autoplot residual analysis with vs without transformations
  - CE1 problem 2f — why residual analysis matters, what to do under violations
related: [linear-regression, gaussian-error-assumptions, design-matrix-and-hat-matrix, collinearity, least-squares-and-mle]
tags:
  - concept
  - module/03-linreg
aliases:
  - QQ plot
  - leverage
  - studentized residuals
  - Tukey-Anscombe plot
  - diagnostic plots
---

# Residual diagnostics

How to check whether the linear-regression assumptions hold. **The QQ plot is the one the prof said he might put on the test.** Plus: leverage = diagonal of the hat matrix, the "fat kid on a seesaw" intuition, why we standardize residuals, the dangerous combination of high leverage + large residual.

## Definition (prof's framing)

Diagnostics check the assumed properties of the unobservable errors $\varepsilon_i$ via the observable residuals $e_i = y_i - \hat y_i$. The prof's question, restated:

> "We'd expect the residual to be centered at zero, kind of mean a zero and some nice distribution. If they don't, then we're like, uh-oh, shit broken." — [[L08-classif-2]]

Four canonical plots from `autoplot()`:

1. **Residuals vs fitted** (Tukey–Anscombe) — checks (1) zero mean, (2) constant variance, hints at (4) independence.
2. **QQ plot** — checks (3) normality.
3. **Scale-location plot** — checks (2) homoscedasticity in particular.
4. **Residuals vs leverage** — flags influential outliers.

## Notation & setup

- Residual: $e_i = y_i - \hat y_i$.
- Leverage: $h_{ii}$ = $i$-th diagonal entry of $\mathbf{H} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$ (see [[design-matrix-and-hat-matrix]]).
- Residual covariance: $\mathrm{Cov}(\mathbf{e}) = \sigma^2 (\mathbf{I} - \mathbf{H})$ — raw residuals have unequal variance and weak correlation.
- Standardized residual: $\tilde r_i = e_i / (\hat\sigma \sqrt{1 - h_{ii}})$.
- Studentized residual: $r^*_i = e_i / (\hat\sigma_{(i)} \sqrt{1 - h_{ii}})$ where $\hat\sigma_{(i)}$ omits point $i$ when estimating variance.

## Formula(s) to know cold

Leverage in **simple linear regression** (the formula the prof flagged for the exercise class):

$$h_{ii} = \frac{1}{n} + \frac{(x_i - \bar x)^2}{\sum_j (x_j - \bar x)^2}.$$

In multiple regression: $h_{ii} = \mathbf{x}_i^\top (\mathbf{X}^\top\mathbf{X})^{-1} \mathbf{x}_i$. Sum-to-trace: $\sum_i h_{ii} = p + 1$.

Standardized residual:

$$\tilde r_i = \frac{e_i}{\hat\sigma \sqrt{1 - h_{ii}}}.$$

Cook's distance (rule of thumb: examine if $D_i > 0.5$, definitely if $> 1$):

$$D_i \propto \tilde r_i^2 \cdot \frac{h_{ii}}{1 - h_{ii}}.$$

## Insights & mental models

### The QQ plot — the one the prof said is "kind of the thing I would put on a test"

> "If everything works out great, then these QQ plots should look like a straight line. And if they're not — like if it looks kind of like this — then you're like, okay, shit's broken." — [[L06-linreg-2]]

> "This one I do recommend learning. This is the kind of thing I would put on a test. This one [residuals-vs-fitted] probably not — I don't want to say no, but I don't know. This [QQ plot] is super useful." — [[L06-linreg-2]]

**Construction:** rank the residuals smallest-to-largest, give each a quantile rank, look up where a Gaussian would put that quantile, plot empirical vs theoretical.

**Interpretation:** straight line = Gaussian; S-shape = heavy/light tails; banana = skew; sharp kinks at the extremes = outliers.

**Useful because:** visual, no test needed; works for any assumed distribution (Poisson, etc.) by changing the theoretical quantile axis.

### "How do I know if a QQ plot looks good?"

There's no quantitative rule — it's an experience-based eye test. Exercise 3.1f makes you simulate $n = 100$ Gaussian draws six times to see how "wonky" QQ plots can look even when the residuals truly are normal.

### Leverage — fat kid on a seesaw

> "The further away the fat kid was from the center the more leverage he had to shoot the other kid off the seesaw and hurt them right and then you know they jump out and and then everyone cried." — [[L08-classif-2]]

> "They should have called this the fat kid seesaw effect. That would have been way better. … Yeah, that won't be on the test." — [[L08-classif-2]]

A point with $x$-value far from the bulk pulls the regression line by a lot. Note: leverage is an $x$-only quantity. A $y$-outlier near the middle of $x$ is much less harmful — there's no lever arm.

### The dangerous corner

The leverage-vs-residual plot has four corners. The dangerous ones are **top-right** and **bottom-right**: high leverage *and* large residual.

> "A high-leverage point that lies on the line is fine ('it actually lines up in the same direction'); the problematic case is a high-leverage point the model can't fit despite being pulled toward it." — [[L08-classif-2]]

Cook's distance summarizes: how much would the fitted model change if you deleted point $i$?

### Why standardize residuals

Raw residuals don't have variance $\sigma^2$ and aren't independent — diagnostics on raw $e_i$ can mislead. Dividing by $\hat\sigma\sqrt{1 - h_{ii}}$ rescales each one to ~unit variance. **Studentized** additionally drops point $i$ from the variance estimate (removes the circularity of using $y_i$ to fit and to evaluate).

> "My guess is, in practice, that these two numbers are going to look exactly the same if you have more than like 50 points or something. But still, it's still nice to be exact or correct." — [[L08-classif-2]]

Standardized/studentized residuals are **diagnostics, not estimators** — your $\hat{\boldsymbol\beta}$ doesn't change.

> "Your betas stay the same. It's just a way to say, is my model any good?" — [[L08-classif-2]]

### What to do when assumptions fail

CE1 problem 2f directly asks this. Toolkit:

- Curvature in residuals → transform $X$ (try $\log X$, $\sqrt X$, $X^2$) or move to [[polynomial-regression]] / GAMs.
- Fanning variance → transform $Y$ (often $\log Y$); use weighted LS; or use a GLM with the right variance structure.
- Non-Gaussian residuals → transform $Y$; use a GLM (e.g., logistic for binary).
- Correlated residuals (independence violated) → mixed-effects model, GLS, time-series model, or chunk-based CV (module 5).
- Outliers → don't just delete. Investigate. *"Whenever you see an outlier just figure out like did you screw something up. Why is it there? Don't just throw it away. Use it as a way to understand your data better."* — [[L08-classif-2]]
- High-leverage point → check if it's a data-entry error; consider robust regression.

## Exam signals

> "This [QQ plot] I do recommend learning. This is the kind of thing I would put on a test." — [[L06-linreg-2]]

> "One exercise you can do for your exercise class is figuring out / verify this formula [for $h_{ii}$ in simple regression] comes from linear regression." — [[L08-classif-2]]

> "Whenever you see an outlier just figure out like did you screw something up. Why is it there? Don't just throw it away. Use it as a way to understand your data better." — [[L08-classif-2]]

## Pitfalls

- **Over-interpreting QQ plots.** Tail wobble is normal even for true Gaussian — Exercise 3.1f makes the point explicitly. Don't reject a model on a slight deviation.
- **Leverage ≠ outlier.** Leverage is a property of $\mathbf{X}$ alone; a high-leverage point can sit perfectly on the line. Outlier means large residual. The dangerous combination is *both*.
- **Raw vs standardized.** If you read residuals off `summary(lm)` directly, they're raw — not unit variance. Use `rstandard()` / `rstudent()` for diagnostic plots.
- **Deleting points to "fix" it.** Robust statistics, not blind deletion. Almost never the right answer in coursework.
- **QQ plot direction.** The convention is theoretical quantiles on x-axis, empirical on y-axis. Some packages flip it; check.
- **Cook's distance threshold.** 0.5 = "give attention," > 1 = "examine." Not a hard rule.

## Scope vs ISLR

- **In scope:** all four `autoplot` panels and what each checks; the QQ plot interpretation; leverage formula in simple LR; the leverage-residual plot's dangerous corner; standardized vs studentized residuals; what to do when assumptions fail.
- **Look up in ISLR:** §3.3.3 (pp. 92–104, *Potential Problems*) — covers the same six diagnostic patterns. Figure 3.9 (residuals-vs-fitted), figure 3.13 (leverage), figure 3.14 (Cook's distance).
- **Skip in ISLR (book-only / prof excluded):** Shapiro–Wilk and other formal normality tests — [[L08-classif-2]]: "we're not going to talk about it." Variance inflation factor (VIF) details — [[L08-classif-2]]: marked self-study, not exam material.

## Exercise instances

- Exercise3.1e — `autoplot()` the Auto fit; comment on outliers and leverage from the four diagnostic panels
- Exercise3.1f — repeatedly simulate $n = 100$ Gaussian draws; see how "wonky" QQ plots can look even under correct assumptions
- CE1 problem 2e — perform residual analysis with `autoplot` on the worm data, with and without the transformations from problem 2b. Are the assumptions fulfilled?
- CE1 problem 2f — why residual analysis matters; what happens if assumptions are violated; mention at least one remedy

## How it might appear on the exam

- **Read a QQ plot.** Show a QQ plot with an S-shape or banana; ask which assumption is violated and what the residuals look like (heavy tails, skew, etc.). Highest-probability question per the prof.
- **Identify the violation in residuals-vs-fitted.** Curvature → linearity broken; fanning → heteroscedasticity; clusters → independence broken.
- **Compute leverage in simple LR.** Given small $(x_i)$ values, plug into $h_{ii} = 1/n + (x_i - \bar x)^2 / \sum_j (x_j - \bar x)^2$.
- **Identify the dangerous corner of a leverage-residual plot.** Top-right or bottom-right; explain why.
- **What to do when…?** Conceptual: what action follows from each diagnostic violation?
- **Studentized vs raw residuals — why?** Raw have unequal variance via $\sigma^2(\mathbf{I} - \mathbf{H})$; standardize to get back the assumed unit-variance behavior.

## Related

- [[linear-regression]] — the model being checked
- [[gaussian-error-assumptions]] — what's being checked
- [[design-matrix-and-hat-matrix]] — leverage = diagonal of $\mathbf{H}$
- [[collinearity]] — adjacent diagnostic concern (VIF would slot here, but excluded)
- [[least-squares-and-mle]] — the fit whose residuals we examine
