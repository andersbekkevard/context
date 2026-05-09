---
concept: local-regression
module: 07-beyondlinear
lectures: [L16, L17]
isl-ref: 7.6
exercises:
  - Exercise7.5 - fit GAM where age uses a local-linear component lo(age, span=0.6)
related: [smoothing-splines, knn-regression, knn-classification, generalized-additive-models, regression-splines]
tags:
  - concept
  - module/07-beyondlinear
aliases:
  - LOESS
  - LOWESS
  - kernel regression
  - local linear regression
---

# Local regression (LOESS)

The "smoothed K-nearest-neighbours" of module 7. At every query point $x_0$, fit a local linear (or quadratic) regression weighted by a kernel around $x_0$. The kernel width plays the role of $K$ in KNN: wide → smooth, narrow → choppy. Drops the basis-function frame entirely, there is no global function, just one local fit per query point.

## Definition (prof's framing)

> "It's basically the same idea as the nearest neighbor algorithm, only now we're going to have a smoothing version of it." - [[L16-beyondlinear-1]]

At a target point $x_0$, fit
$$\hat\beta(x_0) = \arg\min_\beta \sum_{i=1}^{n} K_\lambda(x_0, x_i) \, (y_i - \beta_0 - \beta_1 x_i)^2$$
and predict $\hat f(x_0) = \hat\beta_0(x_0) + \hat\beta_1(x_0) \cdot x_0$. Move $x_0$ along the axis and you get a smooth curve, but a different fit at every query point.

## Notation & setup

- $K_\lambda(x_0, x_i)$: a **kernel weight**, Gaussian centred at $x_0$, or the tricube $K_{i0} = (1 - |(x_0 - x_i) / (x_0 - x_\kappa)|^3)_+^3$ used by R's `loess()` (where $x_\kappa$ is the $k$-th nearest neighbour of $x_0$).
- **span** $s = k/n$: fraction of points used in each local fit. The headline tuning parameter, same role as $\lambda$ in smoothing splines or $K$ in KNN.
- Linear-in-$x$ vs quadratic-in-$x$ inside the local fit: R defaults to local linear ($\beta_0 + \beta_1 x$); local quadratic adds $\beta_2 x^2$. The slide deck's formula is $\beta_0 + \beta_1 x + \beta_2 x^2$ for the quadratic version.
- "Memory-based": you need all the training data at prediction time (same property as KNN). Not stored as a finite parameter vector.

## The LOESS algorithm (ISLR Algorithm 7.1)

For prediction at $x_0$:

1. **Gather the fraction $s = k/n$ of training points** with $x_i$ closest to $x_0$.
2. **Assign weights** $K_{i0} = K(x_i, x_0)$, highest at $x_i = x_0$, decaying to zero at the edge of the neighbourhood; outside the neighbourhood the weight is exactly zero.
3. **Fit a weighted least squares regression** of $y$ on $x$ using these weights.
4. **Predict** $\hat f(x_0) = \hat\beta_0 + \hat\beta_1 x_0$.

The weights differ for every $x_0$, so a *new* weighted regression must be fit for *every* prediction point.

## Insights & mental models

- **Smooth KNN.** Plain KNN regression assigns weight 1 to the $K$ closest neighbours and 0 to everything else (a rectangular kernel). LOESS replaces that hard cutoff with a smooth Gaussian (or tricube), same idea, smoother edges.
- **The kernel width is the flexibility knob**: same role as $\lambda$ in smoothing splines, $K$ in KNN, $d$ in polynomials. The prof:

  > "You have the same notion with the local regression that now the Gaussian is related to the degrees of freedom." - [[L16-beyondlinear-1]]

- **Effective degrees of freedom** carry over: LOESS is a linear smoother, so $\hat{\mathbf y} = \mathbf S \mathbf y$ for some $\mathbf S$, and $\mathrm{df} = \mathrm{tr}(\mathbf S)$. R's `loess()` reports `trace.hat`, that's the effective df.

## Behaviour vs kernel width

| Kernel width / span | What happens |
|---|---|
| **Wide** (span → 1) | All points contribute equally to every local fit → degenerates to a single global linear fit |
| **Narrow** (span → 0) | Only a couple of points contribute at each $x_0$ → choppy, high variance |

Lecture observation on the wage data:

> "Only now you can see it gets wiggly in a rather different way... it can actually get kind of choppy because it's only looking at the first derivative and the Gaussian can get really tiny." - [[L16-beyondlinear-1]]

The wage data has integer ages (no months), so very narrow Gaussians produce visible step-like behaviour because there's *literally no data* between adjacent integer ages.

## Where it sits in module 7

The prof groups smoothing splines and local regression as the two "fit-a-function-directly" methods, in contrast to the basis-function methods (polynomial, step, regression splines):

> "The local regression and the smoothing spline are both kind of functions that you have to fit, where it's not these knots and stuff." - [[L16-beyondlinear-1]]

So:
- **Basis-function family**: polynomial, step, regression spline → OLS on a transformed design matrix.
- **Direct-function family**: smoothing spline (penalised loss), local regression (per-query weighted fit) → no global $\beta$.

## Exam signals

> "It's basically the same idea as the nearest neighbor algorithm, only now we're going to have a smoothing version of it." - [[L16-beyondlinear-1]]

The 2023 exam Q3d included a true/false on the related KNN concept:

> "(iv) The K-nearest neighbors regression (local regression) has a high bias when its parameter, K, is high.", **TRUE**: large K → over-smooths → high bias, low variance. (Same direction logic as wide span in LOESS.)

Note the prof's exam treats LOESS and KNN regression as essentially the same animal for direction-of-effect purposes.

## Pitfalls

- **Curse of dimensionality.** LOESS needs neighbours, and neighbours become meaningless in high $p$. ISLR §7.6: "local regression can perform poorly if $p$ is much larger than about 3 or 4." Same failure mode as KNN.
- **Span direction**: large span → smooth → low variance, high bias; small span → wiggly → high variance, low bias. Same direction as $K$ in KNN, *opposite* to $\lambda$ in smoothing splines.
- **Memory-based prediction**: LOESS doesn't compress the data into a finite parameter set. Every prediction requires the full training set in memory and a fresh weighted regression.
- **Kernel-shape vs span**: span (the fraction of data used) is the headline knob; kernel shape (Gaussian vs tricube vs Epanechnikov) makes a much smaller difference. Don't confuse the two as separate hyperparameters.
- **Local quadratic vs local linear**: ISLR's general formula uses local linear; the slide deck's formula uses local quadratic ($\beta_0 + \beta_1 x + \beta_2 x^2$). Both are reasonable; quadratic is more flexible at the boundaries.

## Scope vs ISLR

- **In scope:** the algorithm (Algorithm 7.1), the "smooth KNN" framing, span as the flexibility knob, behaviour at the two extremes, the `lo(...)` component for GAMs.
- **Look up in ISLR:** §7.6, Algorithm 7.1, Figure 7.9 (the kernel-weighted local fit), Figure 7.10 (span = 0.7 vs 0.2 on wage data).
- **Skip in ISLR:** the brief mention of *varying coefficient models* in §7.6, name-checked only; bivariate and multi-dim local regression also not in scope (the curse-of-dimensionality observation *is* in scope, but not the multi-dim algorithm).
- **Skip:** kernel-density-estimation theory, Nadaraya-Watson estimator algebra, not in slides or lectures.

## Exercise instances

- **Exercise 7.5**: fit a GAM with `lo(age, span = 0.6)` as the local-linear component for `age` (alongside cubic spline for displacement, polynomial for horsepower, linear for weight, factor for origin). Comment: span = 0.6 is moderate-smooth, similar in effect to a moderate `df` in a smoothing spline.

## How it might appear on the exam

- **Direction T/F**: "increasing the span makes LOESS more flexible." (False, wider span → smoother / less flexible.)
- **Method-comparison**: "When would you use LOESS instead of a regression spline?", when you don't want to commit to a specific knot structure; when you want adaptivity to local data density; when interpretability is less important than fit quality.
- **Comparison with KNN**: "How is LOESS related to KNN regression?", same neighbourhood idea, smoothed via a continuous kernel weighting instead of a rectangular cutoff.
- **GAM interpretation**: given a GAM output with an `lo(...)` component, identify what it's doing and how `span` controls it.
- **Curse of dimensionality**: LOESS fails for $p > 3$ or 4, same reason as KNN.

## Related

- [[smoothing-splines]]: the other "fit a function directly" method in module 7. Same loss-plus-some-form-of-regularization spirit, but smoothing splines use a global penalty, LOESS uses a per-query window.
- [[knn-regression]]: direct precursor; LOESS is "smoothed KNN."
- [[knn-classification]]: same neighbourhood idea applied to classification.
- [[generalized-additive-models]]: LOESS slots in as one of the $f_j$ choices via `lo(...)` in `gam()`.
- [[regression-splines]]: the basis-function alternative for the same flexibility budget; splines are global, LOESS is local.
