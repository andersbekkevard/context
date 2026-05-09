---
concept: basis-functions
module: 07-beyondlinear
lectures: [L16, L17]
isl-ref: 7.3
exercises:
  - Exercise7.3  -  derive the natural-cubic-spline basis design matrix for `year` with one knot at 2006
  - Exercise7.4  -  build the AM design matrix by stacking polynomial / truncated-cubic / dummy-coded blocks, verify gam() gives the same fitted values
related: [polynomial-regression, step-functions, regression-splines, generalized-additive-models, linear-regression, design-matrix-and-hat-matrix]
tags:
  - concept
  - module/07-beyondlinear
aliases:
  - basis function approach
---

# Basis functions

The unifying through-line of module 7: **replace $X$ with $b_j(X)$ and you are still doing linear regression**. Polynomial regression, step functions, and regression splines are all the same idea , a different choice of $b_j$ stuffed into the design matrix; OLS does the rest.

## Definition (prof's framing)

> "Instead of looking at $x$ directly we're going through basis functions, and that's a very general term, a very powerful term that has many many versions." - [[L16-beyondlinear-1]]

The model
$$y_i = \beta_0 + \beta_1 x_i + \ldots + \beta_k x_{ik} + \varepsilon_i$$
becomes
$$y_i = \beta_0 + \beta_1 b_1(x_i) + \beta_2 b_2(x_i) + \ldots + \beta_k b_k(x_i) + \varepsilon_i.$$

The $b_j$ are **fixed, known** transformations chosen ahead of time. The design matrix is built from $b_j(x_i)$ values; everything else (closed form $\hat\beta = (X^T X)^{-1}X^T y$, sampling distribution, CIs, F-tests) carries over unchanged because the model is **linear in $\beta$**.

## Notation & setup

- $X$: scalar predictor (the slide deck does the one-predictor case for clarity; multivariate version is just the GAM).
- $b_1, \ldots, b_k$: the $k$ chosen basis functions. The intercept is the implicit "$b_0(x) = 1$".
- Design matrix dimensions $n \times (k+1)$ , same shape as MLR.

$$
\mathbf X = \begin{pmatrix}
1 & b_1(x_1) & \cdots & b_k(x_1) \\
1 & b_1(x_2) & \cdots & b_k(x_2) \\
\vdots & \vdots & & \vdots \\
1 & b_1(x_n) & \cdots & b_k(x_n)
\end{pmatrix}
$$

Each column is one basis function evaluated across the data; each row is one observation.

## The four module-7 instances

| Method | Basis functions $b_j(x)$ |
|---|---|
| Polynomial regression (degree $d$) | $x, x^2, \ldots, x^d$ |
| Step functions ($K$ cutpoints) | $\mathbb{1}(c_{j-1} \le x < c_j)$ |
| Cubic regression spline ($K$ knots) | $x, x^2, x^3, (x - c_1)^3_+, \ldots, (x - c_K)^3_+$ |
| Natural cubic spline ($K$ interior knots) | re-parametrised version with linearity enforced past the boundary knots |

Smoothing splines and local regression **drop** the basis-function frame , they minimize over a function $g$ directly rather than over $\beta$. They are deliberately separated by the prof from the basis-function methods.

## Insights & mental models

> "It's nonlinear, but linear. It's linear in the parameters $\beta$, but it's nonlinear in what you get." - [[L16-beyondlinear-1]]

This slogan covers **all** of polynomial regression, step functions, and regression splines. The whole module is one trick repeated with richer columns in $X$.

The book also collects this idea into one section (ISLR §7.3): polynomials, indicators, splines, wavelets, Fourier , all just choices of $b_j$. Once the design matrix is built, *every* linear-model tool from module 3 (least squares, $t$/$F$-tests, CIs, residual diagnostics) is in scope.

## Exam signals

> "There's a commonality, right? We're talking about basis functions, but all you have to do is fit it with regression." - [[L16-beyondlinear-1]]

> "We're now going to do something more than just finding betas." - [[L16-beyondlinear-1]] (introducing smoothing splines as the *break* with the basis-function frame)

The contrast is itself a fair-game exam point: which methods in module 7 fit by OLS on a basis (poly, step, regression spline), which need a different objective (smoothing spline, local regression).

## Pitfalls

- The design matrix gets wide fast , $K$ knots in a cubic spline already gives $K + 4$ columns (intercept included). High-degree polynomial × many knots = collinearity, instability.
- "Linear in $\beta$" does **not** mean the fitted curve is linear in $x$. Don't confuse the two.
- The truncated-power basis $(x - c_j)^3_+$ is the textbook basis but **R uses `bs()` (B-spline)** which gives the same fit with different columns. The prof: "I don't know why they call it BS." Cosmetic , same predictions.
- For the natural-spline basis (Exercise 7.3), the textbook formula is asymmetric: $b_1(x) = x$, then $b_{k+2}(x) = d_k(x) - d_K(x)$ with $d_k(x) = [(x-c_k)^3_+ - (x-c_{K+1})^3_+]/(c_{K+1} - c_k)$. Easy to mis-write the indexing.

## Scope vs ISLR

- **In scope:** the basis-function frame as the unifier of module 7; the design-matrix construction; that OLS / its inference toolbox carries over.
- **Look up in ISLR:** §7.3 (the explicit "polynomial and step are special cases" framing); §7.4 for the spline-basis derivation.
- **Skip in ISLR:** wavelets and Fourier-basis examples , name-checked in §7.3, not in lecture.

## Exercise instances

- **Exercise 7.3**: derive $\mathbf X_2$ for a natural cubic spline in `year` with one interior knot at 2006 from the textbook basis formula. Pure design-matrix construction, no fitting.
- **Exercise 7.4**: write `mybs()`, `myns()`, `myfactor()` to build the full additive design $\mathbf X = (\mathbf 1, \mathbf X_1, \mathbf X_2, \mathbf X_3)$ by hand and verify $\hat y = \mathbf X(\mathbf X^T \mathbf X)^{-1} \mathbf X^T \mathbf y$ matches what `gam()` returns. The "two different design matrices, same prediction" punchline = different bases, same column space.

## How it might appear on the exam

- "Given a basis $b_1, \ldots, b_k$, write down the design matrix" , pure construction, like Exercise 7.3.
- "Why is fitting a polynomial / cubic spline / step function still 'linear regression'?" , recite the linear-in-$\beta$ slogan.
- "How many parameters does a degree-$d$ polynomial / step function with $K$ cuts / cubic spline with $K$ knots have?" , degree-of-freedom counting (1 + $d$ for poly, $K + 1$ for step including intercept, $K + 4$ for cubic spline).
- Method-comparison: given two different bases that give the same fitted values, explain why (same column space).

## Related

- [[polynomial-regression]]: basis functions $b_j(x) = x^j$; the simplest instance.
- [[step-functions]]: basis functions = indicators of intervals.
- [[regression-splines]]: truncated-power basis, the headline module-7 application.
- [[generalized-additive-models]]: the multivariate generalisation: stack basis-function blocks for each predictor.
- [[linear-regression]]: the host model that the basis-function trick rides on.
- [[design-matrix-and-hat-matrix]]: what's actually being constructed.
