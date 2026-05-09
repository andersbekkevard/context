---
concept: smoothing-splines
module: 07-beyondlinear
lectures: [L16, L17, L27]
isl-ref: 7.5
exercises:
  - Exercise7.5 — fit GAM with a smoothing-spline component s(acceleration, df=3) on Auto data
related: [regression-splines, ridge-regression, cross-validation, regularization, generalized-additive-models, leave-one-out-cv]
tags:
  - concept
  - module/07-beyondlinear
aliases:
  - smoothing spline
  - effective degrees of freedom
  - smoother matrix
---

# Smoothing splines and effective degrees of freedom

A different objective: instead of choosing a basis and fitting OLS, **minimise RSS plus a curvature penalty over the space of all functions $g$**. The minimiser turns out to be a natural cubic spline with a knot at every unique $x_i$ — but shrunk by $\lambda$. This is the prof's favourite example of regularization escaping the "basis function" frame.

## Definition (prof's framing)

> "Now we're going to do something more than just finding betas." - [[L16-beyondlinear-1]]

A smoothing spline is the function $g$ that minimises
$$\sum_{i=1}^{n} (y_i - g(x_i))^2 + \lambda \int g''(t)^2 \, dt.$$

The first term is fit; the second term penalises **curvature** (integrated squared second derivative). $\lambda \ge 0$ controls the trade-off. Direct analogue to ridge:

> "We've looked at situations where we have more than one objective before — we had regularizers... $Y - \beta X$ squared plus sum of $\beta$ squared, that was our ridge regression... a.k.a. $L_2$ norm or regularizer. Really what you're doing is you're adding another objective to your optimization." - [[L16-beyondlinear-1]]

The optimization is over **functions $g$**, not over a finite parameter vector. **Solution**: $g$ turns out to be a natural cubic spline with a knot at every unique $x_i$ — then shrunk via $\lambda$. So smoothing splines are still splines, but with a *very* large basis (one knot per data point) and heavy shrinkage to keep them well-behaved.

## Notation & setup

- $\lambda \ge 0$: smoothing parameter. **Big $\lambda$ → smooth, small $\lambda$ → wiggly.** ([Direction trap](#direction-of-effect-trap-exam-bait), see below.)
- $\mathbf S_\lambda$: $n \times n$ smoother matrix; $\hat{\mathbf y} = \mathbf S_\lambda \mathbf y$. The fitted values are linear in $\mathbf y$, smoothing splines are **linear smoothers**, just like OLS.
- $\mathrm{df}_\lambda = \mathrm{tr}(\mathbf S_\lambda)$: **effective degrees of freedom**, sum of diagonal entries of the smoother matrix.

## Direction-of-effect trap (exam bait)

> [!important] $\lambda$ direction is **opposite** from polynomial / spline degree
> - **$\lambda \uparrow$ → smoother → straight line** (in the limit $\lambda \to \infty$).
> - **$\lambda \downarrow$ → wigglier → interpolates** (at $\lambda = 0$).
>
> This is the *opposite* direction from polynomial degree $d$ (where $d \uparrow$ → wigglier) and the *opposite* direction from spline knot count $K$ (where $K \uparrow$ → wigglier). It is the **same** direction as ridge $\lambda$ and lasso $\lambda$ (where $\lambda \uparrow$ → more shrinkage).
>
> The 2025 exam (Problem 4d) tested this directly:
> *"In smoothing splines, increasing the smoothing parameter $\lambda$ will:*
>   *(i) Make the fitted function more flexible and wiggly.* — **FALSE**
>   *(ii) Make the fitted function smoother and potentially underfit the data.* — **TRUE**
>   *(iii) Increase the penalty for wiggliness.* — **TRUE**
>   *(iv) Decrease the effective degrees of freedom.* — **TRUE**"
>
> $\mathrm{df}_\lambda$ moves the same direction as flexibility (high df → wiggly), so $\mathrm{df}_\lambda$ moves *opposite* to $\lambda$. Lecture: "high df → small $\lambda$ → wiggly. Low df → large $\lambda$ → close to straight."

## Formula(s) to know cold

The objective:
$$\boxed{\;\sum_{i=1}^{n} (y_i - g(x_i))^2 + \lambda \int g''(t)^2 \, dt\;}$$

Effective degrees of freedom:
$$\mathrm{df}_\lambda = \mathrm{tr}(\mathbf S_\lambda)$$

LOOCV shortcut:
$$\mathrm{RSS}_{\text{cv}}(\lambda) = \sum_{i=1}^{n} \left( \frac{y_i - \hat y_i}{1 - \{\mathbf S_\lambda\}_{ii}} \right)^2$$

> "Note that we only need one fit to do cross-validation!" - slide deck [[L16-beyondlinear-1]]

The LOOCV shortcut is structurally identical to the OLS LOOCV shortcut $(y_i - \hat y_i)/(1 - h_{ii})$ — replace the hat-matrix diagonal by the smoother-matrix diagonal.

## The two extremes (memorise direction)

| $\lambda$ | What $g$ does | df |
|---|---|---|
| $\lambda = 0$ | regularizer vanishes; $g$ interpolates every $y_i$ (RSS = 0) | $\to n$ |
| $\lambda \to \infty$ | curvature forced to zero everywhere → $g''(t) \equiv 0$ → **straight line** (the OLS line) | $\to 2$ |

The prof:

> "If you zero out an objective then that thing doesn't do anything." - [[L16-beyondlinear-1]] (re $\lambda = 0$)

> "$\lambda \to \infty$ → $f$ is the straight line we would get from linear least squares regression." - slide deck

In between, $g$ approximates the data while staying smooth. **$\lambda$ slides continuously between "very flexible" and "completely straight."**

## Effective degrees of freedom

Construction:
1. Smoothing-spline fit is linear in $\mathbf y$: $\hat{\mathbf y} = \mathbf S_\lambda \mathbf y$ (smoothing splines are linear smoothers, like OLS).
2. **Effective df** $= \mathrm{tr}(\mathbf S_\lambda)$.

> "It's not obvious, but that's how they define it." - [[L16-beyondlinear-1]]

R parameterises smoothness via `df` (which you can pass as a non-integer like 6.8) **or** via `cv = TRUE` (let LOOCV pick $\lambda$). The package back-solves $\lambda$ for the requested df:

> "My guess is the way to make this work is that they would try a different value of $\lambda$ until you get the degree of freedom you want." - [[L16-beyondlinear-1]]

This is also why "you can get non-integer values of degrees of freedom, which... typically we think of degrees of freedom as being integer values. But here it's an effective degree of freedom."

## Choosing $\lambda$

> "How do I choose $\lambda$? One could be just a decision you make. Or [[cross-validation]], which is a thing I think is particularly useful, because you're still letting the data tell you or give you indications as to what to do." - [[L16-beyondlinear-1]]

The book recommends **leave-one-out CV** because of the closed-form shortcut above — only one fit needed for the entire LOOCV computation. On the wage data the LOOCV-chosen $\lambda$ gave $\mathrm{df}_\lambda \approx 6.8$ — a smoother fit than the arbitrary $\mathrm{df} = 16$ comparison fit.

## Insights & mental models

- **Smoothing spline = ridge in function space.** Same loss-plus-penalty structure; same shrinkage behaviour; same role for $\lambda$. The book makes this explicit (§7.5.1: "Loss + Penalty formulation"). The slide deck (optional Section 7.5.3) makes the algebraic parallel: ridge has $\mathbf S = \mathbf X(\mathbf X^T \mathbf X + \lambda \mathbf I)^{-1} \mathbf X^T$; smoothing spline has $\mathbf S = \mathbf X(\mathbf X^T \mathbf X + \lambda \boldsymbol\Omega)^{-1} \mathbf X^T$ with the curvature matrix $\boldsymbol\Omega$ replacing the identity. The Reinsch-matrix derivation is **optional / out of scope**.
- **Knot at every $x_i$, but heavily shrunk.** A smoothing spline doesn't choose $K$; it *uses everything* and then shrinks. This sidesteps knot-placement entirely.
- **Why $g''(t)$ measures roughness:** $g'(t)$ is the slope; $g''(t)$ is how fast the slope changes. Constant slope → $g''$ near zero → smooth; rapidly changing slope → $|g''|$ large → wiggly. The integral $\int g''(t)^2 \, dt$ aggregates that across the whole range.
- **Compare to a regression spline:** regression spline picks $K$ (a discrete choice, integer dof, no penalty); smoothing spline picks $\lambda$ (continuous, non-integer effective dof, penalty). Both are natural cubic splines under the hood.

## Exam signals

> "Now we're going to do something more than just finding betas." - [[L16-beyondlinear-1]]

> "Adding another objective to your optimization." - [[L16-beyondlinear-1]] (the explicit ridge analogy)

> "We can compute each of these leave-one-out fits using only $\hat g_\lambda$, the original fit to all of the data!" - slide deck (LOOCV shortcut as a clean exam-style fact)

The 2025 exam Problem 4d tested the direction trap explicitly (see callout above). The 2023 exam tested the **wrong-formula trap**:

> "(iii) The smoothing spline ensures smoothness of its function, $g$, by having a penalty term $\int g'(t)^2 dt$ in its loss." — **FALSE**: penalty is $\int g''(t)^2 dt$ (second derivative, not first). 2023 Q3d.

So **memorize**: penalty integrand is $g''(t)^2$, second derivative squared.

## Pitfalls

- **Direction trap** (the one above): $\lambda \uparrow \Rightarrow$ smoother / lower df. Easy T/F mistake, opposite from polynomial degree.
- **Wrong derivative in the penalty**: it's $\int g''(t)^2 \, dt$, not $\int g'(t)^2 \, dt$ or $\int (g'(t))^2 \, dt$. Caught explicitly in 2023 exam Q3d.
- **Effective dof is non-integer.** Don't expect $\mathrm{df}_\lambda$ to be an integer; "df = 6.8" is a normal answer.
- **Smoothing spline ≠ regression spline** despite the name overlap. Regression spline = fixed knots, OLS, integer dof. Smoothing spline = knot at every $x_i$, penalised loss, non-integer dof. They give similar fits in practice, but the *machinery* is different, and the prof was careful to draw the distinction.
- **$\lambda \to \infty$ goes to the OLS straight line, not zero.** When the curvature penalty kills everything wiggly, you don't get the constant-mean fit, you get the *least-squares line*: that's the smoothest function that can still respond to data trend.
- The optional smoother-matrix derivation (Reinsch matrix, eigendecomposition, the long algebra in slide §7.5.3 and Exercise 7.6) is **explicitly not on the exam**.

## Scope vs ISLR

- **In scope:** the loss + curvature-penalty objective; behaviour at the two extremes; effective df = $\mathrm{tr}(\mathbf S)$; LOOCV shortcut; the analogy to ridge.
- **Look up in ISLR:** §7.5.1 (objective and the "natural cubic spline at every $x_i$" claim), §7.5.2 (effective df, LOOCV formula). Both quite short.
- **Skip in ISLR (and slides):** the **optional Section "Computing $\mathbf S$"** in the slide deck (Reinsch matrix construction, eigendecomposition trick) — explicitly optional and not lectured. The proof that the minimiser of $\sum(y_i - g(x_i))^2 + \lambda \int g''^2$ is a natural cubic spline with knots at $x_1, \ldots, x_n$ is stated but not derived in either source.
- **Skip in ISLR (book-only material):** the "ridge connection" optional section in the slide deck is informative but not lectured — same conceptual point that the smoothing spline is "ridge in function space" *is* lectured, just not the algebra.

## Exercise instances

- **Exercise 7.5** — fit a GAM with `s(acceleration, df = 3)` (a smoothing spline component for acceleration with effective df 3) alongside a cubic spline for displacement, polynomial for horsepower, linear for weight, factor for origin. The smoothing spline is one of five different $f_j$ choices in the GAM. Note that `df = 3` is on the *low* end → smooth-ish fit (high $\lambda$).
- **Exercise 7.6 (advanced, optional)** — implement the smoother matrix $\mathbf S$ from scratch via the Reinsch decomposition. *Optional / explicitly out of exam scope.*

## How it might appear on the exam

- **Direction T/F** — the 2025 Problem 4d pattern: which way does $\lambda$ push smoothness, df, wiggliness, fit quality? **Highest-confidence exam pattern for this atom.**
- **Penalty-formula recognition** — 2023 Q3d-style: spot the wrong derivative ($g'$ vs $g''$) in a stated penalty term.
- **Effective dof intuition** — "What is the effective df when $\lambda \to \infty$?" → 2 (straight line, two parameters: intercept + slope).
- **Why use LOOCV here?** Closed-form shortcut means LOOCV is essentially free for smoothing splines (and for OLS, via the hat matrix).
- **Ridge analogy** — "Smoothing splines are to function fitting what ridge regression is to coefficient estimation." Stating this analogy is a high-confidence exam-quality answer.
- **Method-comparison** — when prefer smoothing spline over regression spline? When you don't want to choose knots; when you want a continuous tuning parameter; when LOOCV is cheap.

## Related

- [[regression-splines]]: same family of functions (natural cubic splines), different fitting objective. The smoothing spline minimiser *is* a natural cubic spline.
- [[ridge-regression]]: the canonical "loss + L2 penalty" structural analogue. The prof draws this analogy explicitly.
- [[regularization]]: the cross-cutting Specials atom; smoothing-spline $\lambda$ is one of the prof's headline regularizers.
- [[cross-validation]]: preferred way to choose $\lambda$; LOOCV is the standard because of the closed-form shortcut.
- [[leave-one-out-cv]]: the LOOCV shortcut formula here is structurally identical to the OLS hat-matrix shortcut.
- [[generalized-additive-models]]: smoothing splines slot in as one of the $f_j$ choices via `s(...)` in `gam()`.
