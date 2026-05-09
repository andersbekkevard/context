---
concept: polynomial-regression
module: 03-linreg
lectures: [L06, L16]
isl-ref: 3.3.2, 7.1
exercises:
  - Exercise2.5 — full polynomial-regression simulation for bias-variance
  - Exercise7.1 — fit polynomials of degree 1..4 to mpg ~ horsepower, plot test error vs degree
related: [linear-regression, basis-functions, regression-splines, bias-variance-tradeoff, double-descent, cross-validation]
tags:
  - concept
  - module/03-linreg
aliases:
  - polynomial fit
  - linear in parameters
---

# Polynomial regression

The conceptual point the prof hammered: **still linear regression** — linear in the parameters $\boldsymbol\beta$, even though it's quadratic / cubic in $x$. The standard simulation playground for [[bias-variance-tradeoff]] and the natural lead-in to [[basis-functions]] in module 7.

## Definition (prof's framing)

$$y_i = \beta_0 + \beta_1 x_i + \beta_2 x_i^2 + \cdots + \beta_d x_i^d + \varepsilon_i.$$

Despite the powers of $x$, this is fit with the **same OLS machinery** as plain linear regression — just stack $(x, x^2, \ldots, x^d)$ as columns of $\mathbf{X}$.

> [!important] Linear in parameters — verbatim
> "It's still called linear regression even though you're fitting $y = \beta_0 + \beta_1 x + \beta_2 x^2$. It's a quadratic function but it's still a linear model. And it's linear because it's linear in the coefficients. Linear in the parameters — the things that we actually find out, the things that we're fitting. Those all look like slope terms. So it's weird to think of it that way because the x is changing quadratically. But you have to remember it's still a line. It's just now it's a line in terms of x squared. Even though you might think about it in terms of x, in which case it's curvy — but you're doing everything as linear regression." — [[L06-linreg-2]]

## Notation & setup

- Degree $d$ → $d + 1$ parameters (counting intercept).
- Design matrix has columns $(\mathbf{1}, \mathbf{x}, \mathbf{x}^2, \ldots, \mathbf{x}^d)$ — see [[design-matrix-and-hat-matrix]].
- In R: `lm(y ~ poly(x, d, raw = TRUE))` or `lm(y ~ x + I(x^2) + ...)`.
- The prof keeps the polynomial degree as the canonical "complexity knob" for bias-variance demonstrations.
- Common transformations beyond polynomials are the same idea — $\log x$, $\sqrt x$, $\sin x$, $\cos x$ — all "linear regression" provided the model is linear in $\boldsymbol\beta$.

## Insights & mental models

### When you'd reach for it

The prof's heuristic:

> "Sometimes the world is not linear. In particular, if there is a theoretical/biological/medical reason to believe in a non-linear relationship, or the residual analysis indicates that there are non-linear associations in the data." — module 3 slides

So either *theoretically motivated* (e.g., physics tells you a square law), or *empirically motivated* (residuals-vs-fitted shows curvature → fix by adding $X^2$ or transforming).

### As the bias-variance simulation playground

This is *the* setup for the bias-variance tradeoff demos. Exercise 2.5 simulates polynomials of degrees 1–20 fit to noisy data; tracks training MSE (always falls), test MSE (U-shape), squared bias (falls), variance (rises), and irreducible error (flat). The polynomial degree *is* the complexity knob.

> "The polynomial-fits demo with degrees 1, 5, 10, 25 — those were all linear-regression fits in the parameters." — [[L06-linreg-2]]

This connection to [[bias-variance-tradeoff]] is why polynomial regression keeps showing up in the course even after you've moved on to splines / GAMs / NNs.

### Beyond the U: double descent

The prof's hobbyhorse. As you push past the interpolation point ($\#\text{params} \approx \#\text{samples}$), test error comes back down — the "second descent." Polynomial regression with degree → $\infty$ in finite samples is one of the simplest models exhibiting this.

> "It still captures phenomena that more complicated models exhibit. His example: the second descent in the bias-variance tradeoff when you scale up parameters — credited for much of deep learning's success — *can be seen and understood from the simple regression lens*. Complicated deep models are unreachable theoretically; regression is." — [[L05-linreg-1]]

See [[double-descent]].

### Choosing the degree

Use [[cross-validation]] (typically 10-fold). Test-set MSE bottoms out at the bias-variance optimum; pick that degree, or use the [[one-standard-error-rule]] to push toward a simpler model. **Don't pick by training $R^2$** — it monotonically increases with degree.

### Bridge to module 7 (splines / GAMs)

Polynomials are the simplest [[basis-functions]]. Issues:

- **Global behavior:** a high-degree polynomial wiggles everywhere, even in regions you don't care about. (Splines fix this by being piecewise.)
- **Boundary blow-up:** polynomials extrapolate aggressively past the data range. (Natural splines force linear extrapolation.)

This is exactly the motivation for [[regression-splines]] — same "basis function" framework, better-behaved bases.

## Exam signals

> [!important] Linear in parameters — verbatim
> "It's still called linear regression even though you're fitting $y = \beta_0 + \beta_1 x + \beta_2 x^2$. … It's linear in the parameters." — [[L06-linreg-2]]

> "That's why it's so powerful — you can expand it to all these different kinds of variables. Very convenient." — [[L06-linreg-2]]

> 2025 Q4 polynomial trap: assuming the truth is linear, will training RSS be lower for the bigger (overfit) model? — [[L27-summary]]

## Pitfalls

- **"Quadratic = nonlinear" trap.** False if "linear" refers to the parameters. The model $y = \beta_0 + \beta_1 x + \beta_2 x^2$ is still linear regression. Fooling people on this is a common T/F.
- **Train vs test trap.** A high-degree polynomial always has lower *training* RSS than a low-degree one (more parameters → can't fit worse). Test RSS bottoms out and then rises (U-shape) — see [[bias-variance-tradeoff]]. Watch the keyword.
- **Collinearity from raw polynomial terms.** $x$, $x^2$, $x^3$ are highly correlated, especially for $x$ around the mean. Use `poly(..., raw = FALSE)` for orthogonal polynomials, or center $x$ first. R's `poly()` does this by default.
- **Over-extrapolating.** Polynomial fits explode outside the data range. Don't use a degree-10 fit to predict at $x = 100$ if your data ranges over $0 \le x \le 10$.
- **Choosing the degree by training error.** Always picks the highest degree → overfit. Use CV.
- **Stacking too many predictors.** A degree-$d$ fit on $p$ predictors needs $\binom{p+d}{d}$ terms — explodes fast in multi-predictor regression. Module 7 (GAMs) handles this additively.

## Scope vs ISLR

- **In scope:** the model, that it's linear in parameters, the train-vs-test U-shape via degree, choosing degree by CV, the bridge to basis functions / splines.
- **Look up in ISLR:** §3.3.2 (pp. 90–92, *Non-linear Relationships*) — short subsection; §7.1 (pp. 290–291, *Polynomial Regression*) — module 7's deeper treatment.
- **Skip in ISLR:** orthogonal polynomial construction details — used by `poly()` but not exam material. Local polynomial regression (LOESS) — owned by the [[local-regression]] atom (module 7).

## Exercise instances

- Exercise2.5 — full polynomial-regression simulation: trainMSE, testMSE, decompose into bias², variance, irreducible. The canonical bias-variance simulation. (Owned by [[bias-variance-tradeoff]] but also relevant here as the polynomial-regression playground.)
- Exercise7.1 — `Auto` data: fit polynomials of degree 1..4 to `mpg ~ horsepower`, add fitted lines to a plot, then plot test MSE vs degree. The pure CV-on-degree exercise.

## How it might appear on the exam

- **Linear-in-parameters T/F.** "Adding $X^2$ to a regression makes it nonlinear regression" — false, it's still linear regression.
- **Pick the degree from a CV plot.** Given test MSE vs degree, pick the bottom of the U. Or apply [[one-standard-error-rule]] to pick a simpler model.
- **Train vs test contrast.** "Model with degree 5 vs degree 2 — which has lower training RSS? Test RSS?" Training: always degree 5. Test: depends; usually whichever balances bias and variance.
- **Identify the basis.** "How would you write a degree-3 polynomial as a linear regression?" → expand $\mathbf{X}$ to $(\mathbf{1}, \mathbf{x}, \mathbf{x}^2, \mathbf{x}^3)$ and run OLS.
- **Connection to splines.** Why use a piecewise cubic instead of a global polynomial? Local fit, better extrapolation, less wiggle far from data — bridges to module 7.

## Related

- [[linear-regression]] — the underlying machine
- [[basis-functions]] — polynomial regression is the simplest example
- [[regression-splines]] — same idea, better-behaved bases
- [[bias-variance-tradeoff]] — polynomial degree is the canonical complexity knob
- [[double-descent]] — what happens past the U
- [[cross-validation]] — how to pick the degree
