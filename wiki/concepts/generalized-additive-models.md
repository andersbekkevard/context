---
concept: generalized-additive-models
module: 07-beyondlinear
lectures: [L16, L17, L27]
isl-ref: 7.7
exercises:
  - Exercise7.5 - fit gam() with cubic-spline displacement, polynomial horsepower, linear weight, smoothing-spline acceleration, factor origin (five different f_j types in one model)
related: [basis-functions, regression-splines, smoothing-splines, local-regression, polynomial-regression, step-functions, logistic-regression, linear-regression]
tags:
  - concept
  - module/07-beyondlinear
aliases:
  - GAM
  - additive model
  - logistic GAM
---

# Generalized additive models (GAMs)

The closing trick of module 7: each predictor gets its own non-linear shape ($f_j$, polynomial, spline, smoothing spline, LOESS, indicator), then **add them up**. No interactions across predictors. The natural multivariate generalisation of every method from this module, with the canonical wage / Auto / Boston worked examples.

## Definition (prof's framing)

The additive model:
$$y_i = \beta_0 + f_1(x_{i1}) + f_2(x_{i2}) + \ldots + f_p(x_{ip}) + \varepsilon_i.$$

Each $f_j$ can be **anything from this module**: polynomial, step-function indicator, cubic regression spline, natural spline, smoothing spline, local regression. Different $f_j$ can use different methods.

> "You're assuming that they don't interact. Like it's not like you have to be educated and old, but rather the component that has to do with how old you are can be considered separate from education... they combine in how they predict, right? But they combine additively." - [[L16-beyondlinear-1]]

The **logistic GAM** does the same thing on the log-odds for binary $Y$:
$$\log\frac{P(X)}{1 - P(X)} = \beta_0 + f_1(X_1) + f_2(X_2) + \ldots + f_p(X_p).$$

> "Polynomial logistic regression extends the polynomial regression to logistic data. Just the exact same way." - [[L17-trees-1]]

## Notation & setup

- $f_j$: the (non-linear) component for predictor $X_j$. Free to choose per predictor.
- For binary $Y$: same model, log-odds on the LHS, fit via `family = "binomial"`. Same logic as standard [[logistic-regression]].
- The "**generalized**" part: any GLM link function can be plugged in (Gaussian for regression, logit for binary, etc.). In this course only Gaussian and logit are used.

> "Sounds better like GAMs, right? Because it sounds like you're, I don't know, like jelly or something." - [[L16-beyondlinear-1]]

## Fitting

Two regimes:

**1. All basis-function components** (polynomial, step, regression spline, natural spline). Stack each component's design matrix into one big $\mathbf X = (\mathbf 1, \mathbf X_1, \mathbf X_2, \ldots, \mathbf X_p)$, then plain OLS. Exercise 7.4 walks through this construction explicitly.

The slide deck shows the construction:
$$\mathbf X = \begin{pmatrix} \mathbf 1 & \mathbf X_1 & \mathbf X_2 & \mathbf X_3 \end{pmatrix}$$
with $\mathbf X_1$ a cubic-spline block on age, $\mathbf X_2$ a natural-spline block on year, $\mathbf X_3$ a dummy-coded block on education.

**2. Includes a smoothing-spline `s()` or local-regression `lo()` component**. Plain OLS no longer applies; the **backfitting algorithm** is used. Iterative: hold every other $f_k$ fixed, fit $f_j$ on the partial residuals $r_i = y_i - \sum_{k \ne j} f_k(x_{ik})$, repeat until convergence.

> "Backfitting is used to fit the AM. It is an iterative algorithm where we fit one component at a time, holding the others fixed. Details on this is beyond the scope of what we do here.", slide deck

So **backfitting algorithm internals are out of scope**. The fact that backfitting handles `s()` and `lo()` components is in scope.

## Worked examples (these are exam-ready)

**Wage GAM** (slides + lecture):
$$\texttt{wage} = \beta_0 + f_1(\texttt{age}) + f_2(\texttt{year}) + f_3(\texttt{education}) + \varepsilon$$
- $f_1$: cubic spline on age with knots at 40, 60.
- $f_2$: natural spline on year with knot at 2006.
- $f_3$: step function (dummy-coded education levels).

**Logistic wage GAM**:
$$\log\frac{P(\texttt{wage} > 250)}{1 - P} = \beta_0 + f_1(\texttt{age}) + \beta_2 \cdot \texttt{year}$$
- $f_1$: local linear on age (`lo(age, span = 0.6)`).
- year: linear (`f_2(x) = \beta_2 x`).

**Auto GAM** (Exercise 7.5):
- displacement: cubic spline (`bs`) with one knot at 290.
- horsepower: polynomial degree 2 (`poly`).
- weight: linear.
- acceleration: smoothing spline df = 3 (`s`).
- origin: factor / step function.

**Five different $f_j$ types in one model**, demonstrating the central GAM virtue: per-predictor flexibility.

## Insights & mental models

- **Each panel is a contribution, not a prediction.** The standard plot of a fitted GAM shows one panel per predictor $X_j$, plotting $f_j(X_j)$ against $X_j$ (with the other predictors held at their means).

  > "It's not that you look at this and like, oh well, he's 47, so he makes eight whatever units... this is just the contribution. You also have to consider all these other ones." - [[L17-trees-1]]

  This visualisation is essentially [[partial-dependence-plots|partial dependence]] for the additive case, exact rather than averaged because additivity makes the marginal *exact*.

- **Additive ≠ no flexibility.** Each $f_j$ can be richly non-linear; what's missing is *cross-predictor interactions*. If the data has strong interactions (e.g. salary ↑ with years only when hits is high, the canonical regression-tree example from [[L17-trees-1]]), an additive model will fail.

  > "The assumption is often not a terrible one, either because the assumption is maybe a good one or because adding more terms could be bad, they could get unwieldy." - [[L16-beyondlinear-1]]

- **"Explaining away" via additive structure.** When a previously-significant predictor is paired with a better-suited one, its $f_j$ contribution can flatten. The prof's neuroscience anecdote (his postdoc work with Whitlock):

  > "What mattered is how the animals were postured, how their heads were relative to their bodies, those were the variables that mattered. And when we included those, then the increase that they saw, this trend that they would see, just flattened out... we explained away the correlations that they saw by getting a better set of variables." - [[L17-trees-1]]

  This is the mechanism behind GAM-based variable importance: a term that doesn't contribute when better predictors are present is genuinely redundant.

- **GAMs vs trees.** GAMs handle each variable flexibly but additively; trees handle interactions naturally but split each variable in chunks. The prof closes module 7 (and opens module 8) on this contrast, see [[L17-trees-1]] for the ozone / hitters comparison.

## Pros and cons (ISLP §7.7.1)

| Pros | Cons |
|---|---|
| Flexible non-linear $f_j$ per predictor | **No interactions** unless added manually |
| Interpretable per-predictor contribution plots | Smoothness summarised via dof, slightly opaque |
| Predictions usually more accurate than linear | Less flexible than fully nonparametric (random forest, boosting) |
| Standard inference machinery (CIs, dof) | Backfitting convergence not guaranteed in pathological cases |

For full flexibility without the additivity restriction, use random forests or boosting (modules 8/9). GAMs are the **interpretable middle ground**.

## Exam signals

> "Polynomial logistic regression extends the polynomial regression to logistic data. Just the exact same way." - [[L17-trees-1]]

> "These models will actually center all of the data and then when you predict for each value you're essentially having seeing the other variables set to their mean values." - [[L17-trees-1]]

GAMs appear on **all three past exams** (2023, 2024, 2025) as a method-comparison or interpretation question, see "How it might appear on the exam" below. The 2025 Q4e is the canonical pattern: given a `gam()` call, count dof for each spline term, fit and compare test MSE to other methods.

## Pitfalls

- **Forgetting the additivity assumption.** A GAM cannot capture "$y$ depends on $x_1$ AND $x_2$ together", it captures only the marginal contributions. If interactions matter, you'd need to add explicit interaction terms (which destroys some interpretability) or move to trees / boosting.
- **Confusion about backfitting scope.** Know it exists and that it handles `s()` / `lo()` components; do **not** try to memorise the iterative update formula.
- **Per-panel plot interpretation.** Each $f_j$ panel shows the contribution holding others at their means, not a marginal in the data, not a joint prediction. Easy to misread as "if I'm 47, I make X."
- **Logistic GAM panels are on the log-odds scale.** Plotting shows the $f_j$ contribution to $\log[p/(1-p)]$, not to $p$ directly.
- **Counting dof for a GAM** = sum of dof for each component, and remember to **not** double-count the intercept across components. The 2025 exam Q4e(i) explicitly tests this.
- **`<HS` education category in logistic wage GAM gave huge CIs**: because there were zero positives in that group. Real-data lesson: empty cells inflate CIs. Re-fit excluding the empty category.

## Scope vs ISLP

- **In scope:** the additive form; per-predictor $f_j$ flexibility; logistic-GAM extension; the `gam()` interface conceptually; partial-effect interpretation of fitted-component plots; backfitting at the conceptual level (it exists, handles `s()` and `lo()`); the additive-vs-interactive trade-off.
- **Look up in ISLP:** §7.7.1 (regression GAMs, the wage example, Figures 7.11–7.12), §7.7.2 (logistic GAMs, Figures 7.13–7.14), and the pros/cons summary.
- **Skip in ISLP:** the **backfitting algorithm details** (§7.7.1 footnote on partial residuals, the punchline is in scope, the algebra isn't) and **two-dimensional smoothers** for interactions (mentioned in the cons-of-GAMs paragraph as an extension, not lectured).

## Exercise instances

- **Exercise 7.5**: fit a GAM on Auto with five different $f_j$ types in one model: `bs(displace, knots=290) + poly(horsepower, 2) + weight + s(acceleration, df=3) + factor(origin)`. The point of the exercise is mixing component types, basis-function (cubic spline, polynomial), linear, smoothing spline, factor, all in one `gam()` call. Plot each $f_j$ panel and interpret. **The canonical GAM exercise for this course.**

(Implicit GAM exercise: Exercise 7.4 builds the basis-function-only GAM design matrix by hand, but is filed under [[regression-splines]] / [[basis-functions]] since the fitting is plain OLS.)

## How it might appear on the exam

- **Output interpretation**: given a fitted `gam()` call (e.g. 2025 Q4e: `gam(medv ~ bs(rm,df=5) + crim + dis + bs(age,knots=quantile(age,c(0.2,0.4,0.6,0.8))) + black + chas)`), identify the $f_j$ for each predictor, count total dof, predict / compute test MSE, compare to lasso / boosting outputs.
- **Method choice in a free-form data analysis**: 2023 exam Problem 4 (data analysis): "students will probably use a GAM or a regression tree." If choosing GAM, must explain each term and the choices made.
- **Per-panel interpretation**: given a four-panel plot of $f_1, f_2, f_3, f_4$, describe what each says about the relationship. Standard pattern from the wage and Boston demos.
- **Logistic GAM**: same machinery on log-odds. "What does `f_j(X_j)` represent in `gam(I(wage>250) ~ ..., family='binomial')`?" → contribution to log-odds of wage > 250k holding other predictors at their means.
- **Compare GAM to simpler / fancier alternatives**: GAM > linear (handles non-linearity) but < random forest/boosting (no interactions). 2024/2025 method-comparison patterns: GAM beats linear / lasso, loses to boosted trees.
- **Direction T/F on additivity**: "A GAM can capture interactions between predictors." → False (without explicit cross-terms).

## Related

- [[basis-functions]]: the underlying fit-by-OLS machinery for basis-function GAM components.
- [[regression-splines]]: the most common $f_j$ choice; cubic and natural splines are the workhorses.
- [[smoothing-splines]]: smoothing-spline component via `s(...)`; needs backfitting to fit.
- [[local-regression]]: LOESS component via `lo(...)`; same backfitting story.
- [[polynomial-regression]]: polynomial component via `poly(x, d)` or just `x + I(x^2) + ...`.
- [[step-functions]]: automatically used when a predictor is a factor in the formula.
- [[logistic-regression]]: the host model for the logistic GAM; same `family = "binomial"` machinery.
- [[linear-regression]]: the host model for the regression GAM; the GAM design matrix slots into plain OLS when all components are basis-function.
- [[partial-dependence-plots]]: for non-additive models, the analogue of GAM per-predictor panels.
