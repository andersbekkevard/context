---
concept: regression-splines
module: 07-beyondlinear
lectures: [L16, L17]
isl-ref: 7.4
exercises:
  - Exercise7.3  -  derive the natural-cubic-spline design matrix for `year` with one internal knot at 2006
  - Exercise7.4  -  build the cubic-spline design matrix by hand using the truncated-power basis, verify gam() gives the same fit
related: [basis-functions, polynomial-regression, smoothing-splines, generalized-additive-models, cross-validation]
tags:
  - concept
  - module/07-beyondlinear
aliases:
  - cubic spline
  - natural cubic spline
  - piecewise polynomial regression
---

# Regression splines (cubic and natural)

The headline module-7 method: piecewise cubics joined at knots with continuity through the second derivative. **Natural splines** add boundary constraints that force linear extrapolation past the outer knots, this is the "splines beat high-degree polynomials" punchline.

## Definition (prof's framing)

> "Splines come from shipbuilding... where you have pegs and then you take a piece of wood and bend it down over these pegs." - [[L16-beyondlinear-1]]

A **cubic spline** with $K$ knots $c_1 < \ldots < c_K$ is a piecewise cubic function that is continuous in value, first derivative, and second derivative at every knot. The third derivative is allowed to jump.

A **natural cubic spline** is a cubic spline with the extra constraint that the function is **linear** outside the boundary knots, killing the wild tail behaviour that plain cubic splines suffer from.

> "Where the basis functions meet, and certainly at the boundaries, then the spline degenerates from a cubic to just a linear function, and they do that by adding these boundary knots." - [[L16-beyondlinear-1]]

## Notation & setup

- $K$ = number of **interior** knots; their positions $c_1, \ldots, c_K$ are chosen by hand or at uniform quantiles.
- The truncated-power notation:
$$ (x - c_j)^3_+ = \begin{cases} (x - c_j)^3 & x > c_j \\ 0 & \text{otherwise.} \end{cases} $$
- Fit by ordinary least squares on the basis-expanded design matrix.

## The cubic-spline basis (textbook)

The standard truncated-power basis for a cubic spline with $K$ knots:
$$x, \quad x^2, \quad x^3, \quad (x - c_1)^3_+, \quad (x - c_2)^3_+, \ldots, (x - c_K)^3_+.$$

This is **$K + 3$ basis functions** plus an intercept = **$K + 4$ parameters total**. Adding each truncated cubic introduces a knot whose only discontinuity is in the third derivative, value, slope, and curvature stay continuous "for free." The prof:

> "I'm not gonna go through the math of this, they don't do it in the book either, so I'm not really justifying it but this gives you nice smooth functions." - [[L16-beyondlinear-1]]

For $K = 3$ knots: $3 + 3 + 1 = 7$ parameters. (Lecture worked example: knots at 25, 40, 60 on the wage data.)

## Natural-spline basis

Plain cubic splines have wild boundary tails, the cubic pieces in the outermost regions can swing dramatically, inflating CIs. Natural splines kill this by requiring the second derivative to be zero outside the boundary knots, i.e. linearity at both ends.

Two boundary constraints = **two fewer effective parameters**: a natural cubic spline with $K$ interior knots uses $K + 2$ parameters (intercept absorbed → ISLR counts this as $K$ degrees of freedom in some places, $K + 4 - 4 = K$ for the constraint-counting, or as written in ISLR §7.4.4: a natural cubic spline with $K$ knots has $K$ parameters total when intercept is excluded, see the **dof-counting trap** below).

> [!important] Direction-of-effect / dof counting trap
> ISLR §7.4 / 2024 exam Q2c: "natural cubic spline with **3 cut points** → **4 dof**." The convention: degree-of-freedom count = (intercept) + (one per knot) for a natural cubic spline = $K + 1$ when intercept counts, or $K$ when it doesn't. **A plain cubic spline with $K$ knots = $K + 4$ dof (or $K + 3$ excluding intercept).** Don't mix up the two, natural splines have *fewer* dof than plain cubic splines for the same number of knots because boundary linearity = two constraints.

The textbook basis (Exercise 7.3):
$$ b_1(x) = x, \quad b_{k+2}(x) = d_k(x) - d_K(x), \quad k = 0, \ldots, K-1, $$
$$ d_k(x) = \frac{(x - c_k)^3_+ - (x - c_{K+1})^3_+}{c_{K+1} - c_k}. $$

The math (why this enforces linearity past the boundaries) is **not in the slides or book**:

> "In other courses they go through the math of what these natural splines are. The book doesn't, so I won't either." - [[L16-beyondlinear-1]]

So the formula is something to *use*, not something the prof will ask you to *derive*.

## R interface (interpretation only, no syntax memorisation)

- `bs(x, df=K+3)` or `bs(x, knots=c(...))`, cubic regression spline. The "B" in `bs` is for **B-spline**, an equivalent re-parameterisation that gives the same fit. Prof: "I don't know why they call it BS." Cosmetic.
- `ns(x, df=K+1)` or `ns(x, knots=...)`, natural cubic spline.
- Knot placement: pass `knots=c(25, 40, 60)` for explicit positions; pass `df=K+3` (cubic) or `df=K+1` (natural) to let R place knots at uniform quantiles.

## Why splines beat high-degree polynomials

> "If I was going to fit this with like a degree five polynomial it's going to get funky... but with a spline I can always just make a smooth function through it and it becomes very natural." - [[L16-beyondlinear-1]]

ISLR §7.4.5 / Figure 7.7 makes the same point: a degree-15 polynomial vs a 15-dof natural cubic spline on the wage data, the polynomial blows up at the boundaries, the spline doesn't. **Splines add flexibility by adding knots, not by raising the degree**, a much more local, controllable form of flexibility.

## Insights & mental models

- **Continuity hierarchy** (lecture sketch - [[L16-beyondlinear-1]]):
  1. *Piecewise polynomial*, pieces don't even meet at knots.
  2. *Continuous piecewise polynomial*, meet at knots, but kink ("kind of looks like a butt crack").
  3. *Cubic spline*, second derivative continuous; visually smooth.
- **The shipbuilding analogy.** Bend a wood plank over knots-as-pegs: the wood naturally enforces derivative continuity. If you "put two points like this and then you pull this one down here and you pull that one like here, it just goes wiggly, wiggly, wiggly", same intuition as fitting a spline through too few knots in a wiggly region.
- **Natural splines = boundary regularization.** The two boundary constraints are explicitly ridge-like (kill curvature where data is sparse). They reduce variance at the cost of a tiny bit of bias.
- **Choosing $K$.** The prof recommends "find some visualization tool where they have knots with splines to play with." More principled: pick by [[cross-validation]] (ISLR §7.4.4 / Figure 7.6 shows 10-fold CV on wage data choosing 3–4 dof).

## Exam signals

> "I recommend going into, if you've never played with splines before, find some visualization tool where they have knots with splines to play with." - [[L16-beyondlinear-1]]

> "It's nonlinear, but linear. It's linear in the parameters $\beta$, but it's nonlinear in what you get." - [[L16-beyondlinear-1]]

> "In other courses they go through the math of what these natural splines are. The book doesn't, so I won't either." - [[L16-beyondlinear-1]] (= **derivation of natural-spline basis is OUT of scope**)

The 2024 and 2025 exams **both** asked degree-of-freedom counting questions for splines (2024 Q2c: natural cubic spline, 3 knots → 4 dof; 2025 Q4e(i): cubic spline `bs(age, knots=quantile(...))` with 4 knots → 7 dof = $K + 3$ excluding intercept). This is the canonical exam-style question for module 7.

## Pitfalls

- **Cubic vs natural dof counting.** Plain cubic spline with $K$ knots = $K + 4$ params (incl. intercept) or $K + 3$ (excl.). Natural cubic spline with $K$ interior knots = $K + 2$ params (incl. intercept) or $K + 1$ (excl.). 2025 exam was deliberately ambiguous about the intercept, read carefully whether the model already has an intercept column.
- **Boundary knots ≠ interior knots** for natural splines. The "$c_0$" and "$c_{K+1}$" in the textbook formula are the boundary knots, typically set to the min/max of $x$. They don't add basis functions; they add constraints.
- **Knot placement.** Equally spaced is rarely optimal. Quantile-based placement (R default for `df=` argument) puts more knots where data is denser, usually a better default.
- **High variance at the boundaries** for plain cubic splines is the *only* reason to prefer natural splines for extrapolation; otherwise the two are nearly indistinguishable in the interior.
- The prof is dismissive of memorising basis-function formulas: don't waste exam time deriving the truncated-cubic-power basis from scratch unless asked.

## Scope vs ISLR

- **In scope:** the truncated-power cubic-spline basis; what natural splines do (linear past boundary knots) and *why* (boundary variance reduction); dof counting for both; the comparison to high-degree polynomials; that fitting is OLS.
- **Look up in ISLR:** §7.4.1 (piecewise polynomials), §7.4.2 (continuity constraints), §7.4.3 (truncated-power basis derivation), §7.4.4 (knot placement and CV, Figures 7.5–7.6), §7.4.5 (vs polynomials, Figure 7.7).
- **Skip in ISLR:** the **B-spline (de Boor) basis algorithmic details** are not in the book and not in lecture; the **natural-spline boundary-knot derivation** is explicitly bracketed off by the prof; ISLR's "thin-plate splines" footnote is name-checked only.
- **Skip in ISLR (Bezier / shipbuilding history):** [[L16-beyondlinear-1]]: "in shipbuilding... Renault... Bézier", pedagogical context only, not exam material.

## Exercise instances

- **Exercise 7.3**: derive the natural-cubic-spline design matrix $\mathbf X_2$ for `year` with one interior knot at 2006 (boundary knots at the min/max of `year`). Pure formula-application; uses the textbook basis $b_1(x) = x$, $b_{k+2}(x) = d_k(x) - d_K(x)$, see the natural-spline-basis section above.
- **Exercise 7.4**: build the full additive-model design matrix $\mathbf X = (\mathbf 1, \mathbf X_1, \mathbf X_2, \mathbf X_3)$ where $\mathbf X_1$ is a cubic spline in `age` with knots at 40, 60 (truncated-power basis), $\mathbf X_2$ is the natural spline from Problem 3, $\mathbf X_3$ is dummy-coded `education`. Verify $\hat y = \mathbf X (\mathbf X^T \mathbf X)^{-1} \mathbf X^T \mathbf y$ matches `gam()`'s fitted values. Punchline: different basis (truncated-power vs B-spline) → different design matrix → **same column space → same predictions**.

## How it might appear on the exam

- **Degree-of-freedom counting** (most common past-exam pattern):
  - "Natural cubic spline with $K$ cut points consumes how many dof?" → $K + 1$ (incl. intercept) or $K$ (if intercept already in model). 2024 Q2c.
  - "Cubic spline `bs(age, knots=quantile(age, c(0.2,0.4,0.6,0.8)))` consumes how many dof?" → $K + 3 = 7$ if intercept already in the model. 2025 Q4e(i).
- **Construct the design matrix** for a cubic / natural spline given knots, Exercise 7.3-style, plug into the truncated-power formula.
- **T/F**: "A cubic spline with $K$ knots has continuous third derivative at every knot." (False, only continuous through second derivative; third derivative jumps.)
- **Method-comparison**: "Why prefer a natural cubic spline over a polynomial of equal flexibility?", boundary stability; flexibility comes from knot count not from degree.
- **Interpretation**: given a `bs()` or `ns()` GAM output, identify how many dof each spline term consumes and what the term contributes.

## Related

- [[basis-functions]]: splines are the headline basis-function instance; the truncated-power basis is the canonical example.
- [[polynomial-regression]]: the alternative basis-function fit; spline beats poly at the boundaries, same flexibility at lower degree.
- [[smoothing-splines]]: drops the knot-and-basis frame; uses a curvature penalty instead. The connection: a smoothing spline is a natural cubic spline with a knot at every unique $x_i$, then shrunk via $\lambda$.
- [[generalized-additive-models]]: splines slot in as one of the $f_j$ basis-function choices; the GAM machinery just stacks the $\mathbf X_j$ blocks.
- [[cross-validation]]: preferred way to choose $K$ (number of knots) in a regression spline.
