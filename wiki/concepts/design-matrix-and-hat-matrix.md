---
concept: design-matrix-and-hat-matrix
module: 03-linreg
lectures: [L06, L08, L10]
isl-ref: 3.2.1
exercises:
  - Exercise6.1a  -  derive normal equations / closed form
related: [linear-regression, least-squares-and-mle, sampling-distribution-of-beta, residual-diagnostics, collinearity, leave-one-out-cv]
tags:
  - concept
  - module/03-linreg
aliases:
  - normal equations
  - hat matrix
  - X transpose X
  - leverage
---

# Design matrix, normal equations, and the hat matrix

The algebra under multiple regression. **$\mathbf{X}$ is $n \times (p+1)$ with a column of ones** (the intercept hides in there). $\mathbf{H} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$ is the **hat matrix** , it puts the hats on $\mathbf{y}$, its diagonal is leverage, it appears in the LOOCV shortcut, and the prof's annoyance with "design matrix" terminology is part of the lecture's flavor.

## Definition (prof's framing)

> "It's often called the design matrix. The data. Never understood why. It's not really a design of any kind. But it's what people call it." - [[L06-linreg-2]]

Design matrix:

$$\mathbf{X} = \begin{bmatrix} 1 & x_{11} & x_{12} & \cdots & x_{1p} \\ 1 & x_{21} & x_{22} & \cdots & x_{2p} \\ \vdots & & & & \vdots \\ 1 & x_{n1} & x_{n2} & \cdots & x_{np}\end{bmatrix}.$$

The leading column of ones lets you absorb the intercept $\beta_0$ into the same $\boldsymbol\beta$ vector. "Behind this beta is actually an X. It's just all the values of X are one. So you don't need to write it." - [[L06-linreg-2]]

Hat matrix:

$$\mathbf{H} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top.$$

> "This matrix H is also known as the hat matrix. The reason we call it a hat matrix is that it's where all the hats come from." - [[L08-classif-2]]

## Notation & setup

- $n$ samples, $p$ predictors, $\mathbf{X}$ is $n \times (p{+}1)$.
- Some books include the intercept in $p$ , gotcha he flagged.
- $\mathbf{H}$ is $n \times n$, **symmetric** ($\mathbf{H}^\top = \mathbf{H}$) and **idempotent** ($\mathbf{H}^2 = \mathbf{H}$). It's the orthogonal projection onto the column space of $\mathbf{X}$.
- $\mathbf{I} - \mathbf{H}$ is the orthogonal projection onto the residual space.
- Leverage of observation $i$: $h_{ii} = \mathbf{H}[i,i]$ , diagonal entry of $\mathbf{H}$, depends only on $\mathbf{X}$, not on $\mathbf{y}$.

## Formula(s) to know cold

Normal equations (from differentiating RSS):

$$\boxed{\mathbf{X}^\top\mathbf{X}\,\hat{\boldsymbol\beta} = \mathbf{X}^\top \mathbf{y}}$$

$$\implies \hat{\boldsymbol\beta} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}.$$

Predictions:

$$\hat{\mathbf{y}} = \mathbf{X}\hat{\boldsymbol\beta} = \underbrace{\mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top}_{\mathbf{H}}\mathbf{y} = \mathbf{H}\mathbf{y}.$$

Leverage in **simple linear regression** (closed form, also asked in the exercise class):

$$h_{ii} = \frac{1}{n} + \frac{(x_i - \bar x)^2}{\sum_j (x_j - \bar x)^2}.$$

Sum-to-trace identity: $\sum_i h_{ii} = \mathrm{tr}(\mathbf{H}) = p+1$.

Residual covariance:

$$\mathrm{Cov}(\mathbf{e}) = \sigma^2 (\mathbf{I} - \mathbf{H}).$$

LOOCV shortcut for OLS , only **one fit needed**, because $\mathbf{H}$ already encodes how each point is fitted:

$$\mathrm{CV}_n = \frac{1}{n}\sum_{i=1}^n \left(\frac{y_i - \hat y_i}{1 - h_{ii}}\right)^2.$$

## Insights & mental models

### Three roles of $\mathbf{H}$ in this course

The hat matrix is the binding object across modules:

1. **Predictor:** $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$ , turns observed $\mathbf{y}$ into fitted values.
2. **Leverage:** the diagonal $h_{ii}$ measures how much each $\mathbf{x}_i$ pulls the fit. A point with high $h_{ii}$ AND large residual is the dangerous combination , see [[residual-diagnostics]] and the "fat kid on a seesaw" image from L08.
3. **LOOCV shortcut for OLS:** the hat matrix lets you compute leave-one-out CV without re-fitting $n$ times. The prof showed this in [[L10-resample-1]].

### Tug-of-war with $\mathbf{X}^\top\mathbf{X}$

The factor $(\mathbf{X}^\top\mathbf{X})^{-1}$ is the load-bearing element:

> "I think this is really why statisticians love these distributions, because you can read out what's going to happen when you look at them. You can be like , ah, that X transpose X is going to screw us later." - [[L06-linreg-2]]

When two predictors are nearly identical, $\mathbf{X}^\top\mathbf{X}$ is near-singular → its inverse blows up → the variance of $\hat{\boldsymbol\beta}$ explodes (the [[collinearity]] story). See also [[sampling-distribution-of-beta]] for the $\sigma^2(\mathbf{X}^\top\mathbf{X})^{-1}$ covariance.

### Reduction to univariate

Recommended exercise: show that the matrix formula reduces to the simple-LR formula for $\hat\beta_1$ when $p=1$. Same RSS, same derivative, same answer , just less notation.

## Exam signals

> "This matrix H is also known as the hat matrix. The reason we call it a hat matrix is that it's where all the hats come from." - [[L08-classif-2]]

> "One exercise you can do for your exercise class is figuring out / verify this formula [for $h_{ii}$ in simple regression] comes from linear regression." - [[L08-classif-2]]

> "I think this is really why statisticians love these distributions, because you can read out what's going to happen when you look at them. You can be like , ah, that X transpose X is going to screw us later." - [[L06-linreg-2]]

## Pitfalls

- **Forgetting the intercept column.** Without the leading column of ones, $\hat\beta_0$ is fixed at zero. If you write $\hat\beta = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ without the ones column, you're forcing the fit through the origin.
- **Identifiability with categorical.** $K$ dummies for $K$ levels makes columns linearly dependent , $\mathbf{X}^\top\mathbf{X}$ singular, no inverse. Use $K{-}1$ dummies plus a reference level. See [[categorical-encoding-and-interactions]].
- **High-leverage point.** A point with $h_{ii}$ near 1 effectively determines its own fitted value ($\hat y_i \approx y_i$); the rest of the data has little say. Doesn't break the algebra but distorts conclusions.
- **$\mathbf{H}$ depends only on $\mathbf{X}$.** Useful: leverage is a property of the design, *not* of the response. Detect high-leverage points before you even look at $\mathbf{y}$.
- **The inverse may not exist.** Rank deficiency (perfect [[collinearity]] or $n < p+1$) breaks the closed form. Without "tricks" (regularization, pseudoinverse), OLS literally has no unique solution.

## Scope vs ISLR

- **In scope:** the design-matrix structure, the normal-equations derivation, definition of the hat matrix, leverage as $h_{ii}$, and the LOOCV shortcut formula.
- **Look up in ISLR:** §3.2.1 (estimating coefficients) , the matrix formulation is mostly in equation form, light on derivation. §3.3.3 covers leverage briefly (pp. 97–98). §5.1.2 covers LOOCV with the shortcut formula.
- **Skip in ISLR (book-only / prof excluded):** Moore–Penrose pseudoinverse details , [[L08-classif-2]]: "explicitly bracketed off." Spectral / eigen-decomposition theory of $\mathbf{X}^\top\mathbf{X}$ , [[L04-statlearn-3]] deferred to Linear Statistical Models.

## Exercise instances

- Exercise6.1a: derive $\hat{\boldsymbol\beta} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ from RSS. (Listed under module 6 because it's the LS estimator derivation; conceptually it's module 3 material , and per L12, [[L12-modelsel-1]], the prof considered this an already-done module-3 exercise.)

## How it might appear on the exam

- **Derive the normal equations.** Take RSS, expand, differentiate, solve. The 6-line derivation in [[L06-linreg-2]] is the template.
- **Verify the simple-LR leverage formula.** $h_{ii} = 1/n + (x_i - \bar x)^2 / \sum_j (x_j - \bar x)^2$ is *the* exercise the prof flagged in L08.
- **Why does the LOOCV shortcut work for OLS?** Because $\mathbf{H}$ is independent of $\mathbf{y}$, you can compute $\hat y_i$ from the full-data fit, then "remove" point $i$'s effect via the $1/(1-h_{ii})$ correction.
- **Read leverage off a plot.** Given a residuals-vs-leverage plot ([[residual-diagnostics]]), identify the dangerous corner: high $h_{ii}$ AND large residual.

## Related

- [[linear-regression]]: the model whose algebra this is
- [[least-squares-and-mle]]: the derivation of $\hat{\boldsymbol\beta}$
- [[sampling-distribution-of-beta]]: uses $(\mathbf{X}^\top\mathbf{X})^{-1}$ in the covariance
- [[residual-diagnostics]]: leverage and the leverage-vs-residual plot
- [[collinearity]]: what happens when $\mathbf{X}^\top\mathbf{X}$ is near-singular
- [[leave-one-out-cv]]: the $1/(1-h_{ii})$ shortcut lives here
