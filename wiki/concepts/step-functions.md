---
concept: step-functions
module: 07-beyondlinear
lectures: [L16]
isl-ref: 7.2
exercises: []
related: [basis-functions, regression-splines, generalized-additive-models, categorical-encoding-and-interactions]
tags:
  - concept
  - module/07-beyondlinear
aliases:
  - piecewise-constant regression
  - binned regression
---

# Step functions

The "stupid but actually pretty common" basis-function instance: cut the predictor range into $K+1$ bins, fit one constant per bin. Discontinuous, no derivatives, but cheap, robust, and the right tool when you genuinely don't have many constraints to push around.

## Definition (prof's framing)

> "I mean, this one is very stupid, but it's actually quite common, simply because you don't need that much information... so you don't have too many constraints that push things around. Even the step functions are actually pretty nice, even if they are a bit stupid looking." - [[L16-beyondlinear-1]]

Pick cutpoints $c_1 < c_2 < \ldots < c_K$ in the range of $X$. Define indicator basis functions
$$b_j(x) = \mathbb{1}(c_{j-1} \le x < c_j), \qquad j = 1, \ldots, K-1, \qquad b_K(x) = \mathbb{1}(c_K \le x).$$

Fit by ordinary least squares with these as columns of $\mathbf X$, same OLS machinery as linear regression, just with a binned design matrix. The result is a piecewise-constant fit: one mean per bin.

## Notation & setup

- $K$ cutpoints $\Rightarrow K + 1$ bins $\Rightarrow K$ dummy basis columns + intercept = $K + 1$ parameters total.
- The first bin is absorbed into the intercept (reference category, just like dummy coding for factors in module 3).
- In R: `cut(age, K)` produces the binned design matrix; pass it to `lm()`.

## Mathematical structure

Design matrix:
$$
\mathbf X = \begin{pmatrix}
1 & \mathbb{1}(c_1 \le x_1 < c_2) & \cdots & \mathbb{1}(c_K \le x_1) \\
\vdots & \vdots & & \vdots \\
1 & \mathbb{1}(c_1 \le x_n < c_2) & \cdots & \mathbb{1}(c_K \le x_n)
\end{pmatrix}.
$$

Each row has a 1 in the intercept column and a 1 in exactly one bin column.

Fitted prediction: $\hat\beta_0$ is the mean response in the first bin (where $X < c_1$); $\hat\beta_0 + \hat\beta_j$ is the mean in bin $j$ relative to the reference. Confidence intervals come for free, it's just OLS:

> "Confidence intervals come the same as you would get the confidence intervals for the prediction in the linear model, because it's just a linear model only now on these basis functions, which in that case were just fixed intervals." - [[L16-beyondlinear-1]]

## Insights & mental models

- **Step functions = "regression on a factor variable."** When the original $X$ is already a factor (e.g. `education` with levels `<HS, HS, Some College, College, Advanced Degree`), `lm(wage ~ education)` is a step function for free, no `cut()` needed. The wage-vs-education demo in lecture is exactly this case.
- **No derivatives.** The fit is piecewise constant, which means jumps at every cutpoint:

  > "You don't have derivatives here. They're not even... it's piecewise constant, but it's not connected, it can jump." - [[L16-beyondlinear-1]]

- **Cutpoint choice is manual.** R defaults to equally spaced bins (or even quantile-based via `breaks=`), but you can hand-pick the breaks. The book notes that 5-year age groups are routine in biostatistics / epidemiology.
- **Why "a bit stupid":** between the cutpoints the model is forced to ignore variation. The first bin in the wage-vs-age demo "clearly misses the increasing trend" (book §7.2 / Figure 7.2 commentary).

## Exam signals

> "Even the step functions are actually pretty nice, even if they are a bit stupid looking." - [[L16-beyondlinear-1]]

Step functions are introduced as a pedagogical bridge from polynomial regression to splines. No prof quote flagging them as exam-likely on their own, but they are an explicit instance of [[basis-functions]], and "what is the design matrix for a step function with cutpoints at..." is a fair-game design-matrix-construction question.

## Pitfalls

- **Choice of cutpoints matters.** Too few and you smear over real structure; too many and the step boundaries dominate. Equal-width bins are not equal-count bins.
- **The first bin is the intercept.** Don't double-count: with $K$ cutpoints you have $K + 1$ bins and $K$ dummy columns, not $K + 1$.
- **Step function ≠ ordered factor.** When $X$ is genuinely ordinal, a step function throws away the ordering (each level gets its own coefficient, no monotonicity). Sometimes that's what you want, sometimes not.
- The discontinuities are a feature, not a bug, but if you need a smooth fit you should be using a [[regression-splines|spline]] instead.

## Scope vs ISLR

- **In scope:** definition, the indicator basis, design matrix construction, the wage-vs-education and wage-vs-age examples, the link to dummy-coding factors.
- **Look up in ISLR:** §7.2 (Figure 7.2, wage vs age step-function fit and its logistic counterpart).
- **Skip in ISLR:** nothing specific; the book's treatment is short and matches the slides.

## Exercise instances

None in the recommended-exercise sheet for module 7. Step functions appear *implicitly* inside Exercise 7.5 (the GAM uses `factor(origin)`, which is a step-function basis), and inside Exercise 7.4 via `myfactor()` for `education`.

## How it might appear on the exam

- "Write the design matrix for a step-function regression of $Y$ on $X$ with cutpoints at $c_1, c_2$": pure basis-function construction.
- "Interpret the coefficient $\beta_2$ in a piecewise-constant fit": it's the difference in mean response between bin 2 and the reference bin.
- "How many parameters does a step function with $K$ cutpoints have?": $K + 1$ (including intercept).
- T/F: "A step-function fit is continuous." (False, it has jumps at the cutpoints.)
- Method-comparison: when is a step function preferable to a polynomial / spline? When the data has natural breakpoints, when interpretability per bin matters, or when there's not enough data to constrain a smoother fit.

## Related

- [[basis-functions]]: step functions are the indicator instance.
- [[regression-splines]]: the smooth alternative when discontinuities at the cutpoints are unacceptable.
- [[categorical-encoding-and-interactions]]: step functions on a factor variable *are* dummy-coded categorical regression; the machinery is identical.
- [[generalized-additive-models]]: step functions slot in as one of the $f_j$ choices, especially for qualitative predictors.
