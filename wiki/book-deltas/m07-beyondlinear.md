---
title: "M07: Moving Beyond Linearity — Book delta"
module: 07-beyondlinear
isl-ch: 7
lectures: [L16, L17]
---

# Module 07: Moving Beyond Linearity — Book delta

ISLP ch. 7 covers the conceptual sweep of module 7 well ([[basis-functions|basis-functions framing]], piecewise polynomials, [[regression-splines|regression splines]], [[smoothing-splines|smoothing splines]], [[local-regression|LOESS]], [[generalized-additive-models|GAMs]]). The deltas are mostly **explicit formulas and design-matrix templates** that the prof wrote on the board / on the slides but ISLP either skips, hides in a footnote, or states only in the abstract form. The Exercise-7.3 and Exercise-7.4 hand-construction artifacts are the load-bearing ones: ISLP does not give the natural-spline basis formula in usable form, and does not write out an additive-model design matrix block-by-block.

What's deliberately out: the Reinsch-matrix / optional smoother-matrix algebra (slide §"Computing $\mathbf S$" + Exercise 7.6) — the prof bracketed this off as optional and the natural-spline boundary-knot derivation as "the book doesn't, so I won't either" [[L16-beyondlinear-1|L16]]. Both are listed in the MOC's `## Out of scope` block.

---

## 1. Basis-function design matrix in OLS form

[[L16-beyondlinear-1|L16]], [[basis-functions]], slide deck §"Basis Functions"

ISLP §7.3 states the [[basis-functions|basis-function]] model

$$
y_i = \beta_0 + \beta_1 b_1(x_i) + \beta_2 b_2(x_i) + \cdots + \beta_K b_K(x_i) + \varepsilon_i
$$

and remarks that "all of the inference tools for linear models … are available in this setting." It does **not** write out the design matrix or the OLS estimator with the basis columns plugged in. The prof's slide does, and the entire module rests on it.

The design matrix is

$$
\mathbf{X} =
\begin{pmatrix}
1 & b_1(x_1) & b_2(x_1) & \cdots & b_k(x_1) \\
1 & b_1(x_2) & b_2(x_2) & \cdots & b_k(x_2) \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & b_1(x_n) & b_2(x_n) & \cdots & b_k(x_n)
\end{pmatrix},
$$

an $n \times (k+1)$ matrix with $k$ basis-function columns plus the intercept column. The OLS estimator is unchanged:

$$
\hat{\boldsymbol\beta} = (\mathbf{X}^\mathsf{T}\mathbf{X})^{-1}\mathbf{X}^\mathsf{T}\mathbf{y},
$$

and so are all of its consequences (sampling distribution of $\hat{\boldsymbol\beta}$, $\mathrm{Var}\,\hat{f}(x_0) = \ell_0^\mathsf{T} \hat{\mathbf{C}}\,\ell_0$, pointwise SEs, F-statistics).

> Slogan the prof keeps coming back to: *"It's nonlinear, but linear. It's linear in the parameters $\beta$, but it's nonlinear in what you get."* [[L16-beyondlinear-1|L16]]

### Worked numerical example (slides only)

Take $b_1(x) = x$, $b_2(x) = x^2$, data $\mathbf{x} = (6,3,6,8)^\mathsf{T}$, $\mathbf{y} = (3,-2,5,10)^\mathsf{T}$. Then

$$
\mathbf{X} =
\begin{pmatrix}
1 & 6 & 36 \\
1 & 3 & 9 \\
1 & 6 & 36 \\
1 & 8 & 64
\end{pmatrix},
\qquad
\hat{\boldsymbol\beta} = (\mathbf{X}^\mathsf{T}\mathbf{X})^{-1}\mathbf{X}^\mathsf{T}\mathbf{y} = (-4.4,\; 0.2,\; 0.2)^\mathsf{T}.
$$

This is the lecture-time concrete "build the design matrix, run OLS, done" demo. It's the template the prof reuses for every method in module 7.

---

## 2. General-order spline basis (truncated-power form)

[[L16-beyondlinear-1|L16]], [[regression-splines]], slide deck §"Regression Splines"

ISLP §7.4.3 gives only the cubic ($M=4$) version of the truncated-power basis (Eq. 7.10). The slide deck states the **general degree-$(M-1)$ spline** version, which the prof uses to derive both the cubic and the natural-cubic versions in one move.

A **spline of order $M$** is a piecewise polynomial of degree $M-1$ joined at $K$ knots $c_1, \dots, c_K$, with continuous derivatives up to order $M-2$ at each knot. The truncated-power basis is

$$
(x - c_j)^{M-1}_+ \;=\;
\begin{cases}
(x - c_j)^{M-1} & x > c_j \\
0 & \text{otherwise.}
\end{cases}
$$

The standard basis is

$$
\begin{aligned}
b_j(x) &= x^j, & j &= 1, \ldots, M-1, \\
b_{M-1+k}(x) &= (x - c_k)^{M-1}_+, & k &= 1, \ldots, K.
\end{aligned}
$$

There are **$M + K - 1$ basis functions** plus an intercept, so $M + K$ parameters total.

| $M$ (order) | Degree $M-1$ | Basis | # basis funcs | Total params (incl. intercept) |
|---|---|---|---|---|
| 2 | 1 (linear spline) | $x$, $(x-c_k)_+$ | $1 + K$ | $2 + K$ |
| 3 | 2 (quadratic spline) | $x, x^2$, $(x-c_k)^2_+$ | $2 + K$ | $3 + K$ |
| 4 | 3 (cubic spline) | $x, x^2, x^3$, $(x-c_k)^3_+$ | $3 + K$ | $4 + K$ |

ISLP gives only the $M = 4$ row. The cubic case ($M=4$, $K$ knots → $K+4$ degrees of freedom incl. intercept, $K+3$ excl.) is the one tested by past exams.

---

## 3. Natural cubic spline basis: closed-form formula

[[L16-beyondlinear-1|L16]], [[regression-splines]], slide deck §"Natural Cubic Splines", Exercise 7.3

This is the single most load-bearing m07 delta. ISLP §7.4 introduces natural splines only **conceptually** ("linear past the boundary knots") and confines the parameter accounting to a footnote (§7.4.4 footnote 4) without writing the basis. The prof's slide gives the **explicit basis formula** that Exercise 7.3 plugs into. The prof himself flagged that he won't derive *why* this enforces linearity past the boundary — "in other courses they go through the math … the book doesn't, so I won't either" [[L16-beyondlinear-1|L16]] — so the formula is something to **apply**, not **derive**.

### Setup

- $K$ interior knots $c_1 < c_2 < \cdots < c_K$.
- Two **boundary knots** $c_0$ and $c_{K+1}$, conventionally set to $\min(x)$ and $\max(x)$. They add **constraints** (second derivative $= 0$ past the boundary), not basis columns.
- Two boundary constraints kill two truncated-cubic basis functions, so a natural cubic spline with $K$ interior knots has $K + 1$ basis functions (plus the intercept) → **$K + 2$ parameters total**.

### Basis formula

$$
b_1(x) = x,
\qquad
b_{k+2}(x) = d_k(x) - d_K(x), \quad k = 0, 1, \ldots, K-1,
$$

with the helper

$$
d_k(x) \;=\; \frac{(x - c_k)^3_+ \;-\; (x - c_{K+1})^3_+}{c_{K+1} - c_k}.
$$

So the column count is: one "$x$" column + $K$ columns of the form $d_k(x) - d_K(x)$ = **$K+1$ non-intercept columns**.

### Worked instance: one interior knot at 2006 (Exercise 7.3)

`year` with one interior knot $c_1 = 2006$ and boundary knots $c_0 = 2003$, $c_2 = 2009$ (the min and max of `year` in the Wage data). Here $K = 1$, so the only index is $k = 0$ and we need two basis functions:

$$
b_1(x) = x,
$$

$$
\begin{aligned}
b_2(x) &= d_0(x) - d_1(x) \\
&= \frac{(x - c_0)^3_+ - (x - c_2)^3_+}{c_2 - c_0}
   \;-\; \frac{(x - c_1)^3_+ - (x - c_2)^3_+}{c_2 - c_1} \\
&= \frac{1}{c_2 - c_0}(x - c_0)^3_+
   \;-\; \frac{1}{c_2 - c_1}(x - c_1)^3_+
   \;+\; \Big(\frac{1}{c_2 - c_1} - \frac{1}{c_2 - c_0}\Big)(x - c_2)^3_+ \\
&= \tfrac{1}{6}(x - 2003)^3_+
   \;-\; \tfrac{1}{3}(x - 2006)^3_+
   \;+\; \tfrac{1}{6}(x - 2009)^3_+.
\end{aligned}
$$

Because $c_2 = 2009$ is the upper boundary knot and the data satisfies $2003 \le x \le 2009$, the term $(x - 2009)^3_+ = 0$ for every observation. So **on the data range** the basis simplifies to

$$
b_2(x) \;=\; \tfrac{1}{6}(x - 2003)^3 \;-\; \tfrac{1}{3}(x - 2006)^3_+.
$$

(The leading $(x - 2003)^3_+$ also loses its $_+$ because $x \ge 2003$ already.) The design matrix is then

$$
\mathbf{X} =
\begin{pmatrix}
1 & x_1 & \tfrac{1}{6}(x_1 - 2003)^3 - \tfrac{1}{3}(x_1 - 2006)^3_+ \\
1 & x_2 & \tfrac{1}{6}(x_2 - 2003)^3 - \tfrac{1}{3}(x_2 - 2006)^3_+ \\
\vdots & \vdots & \vdots \\
1 & x_n & \tfrac{1}{6}(x_n - 2003)^3 - \tfrac{1}{3}(x_n - 2006)^3_+
\end{pmatrix},
$$

three columns, two non-intercept basis functions → degrees of freedom = 2 (excl. intercept) = $K + 1$.

This is the canonical Exercise-7.3 / "construct the natural-cubic-spline design matrix by hand" object the prof signaled is fair game.

---

## 4. Local regression: full slide-form objective and the tricube kernel

[[L16-beyondlinear-1|L16]], [[local-regression]], slide deck §"Local Regression"

ISLP Algorithm 7.1 / Eq. 7.14 gives only the **local-linear** weighted objective and is deliberately silent on the kernel formula ("we will avoid getting into the technical details … there are books written on the topic"). The slide deck gives the **local-quadratic** objective and writes out the **tricube kernel** that R's `loess()` actually uses. Both extras are lookup-able and unique to the slides.

### Local-quadratic objective

At a target point $x_0$, find $\hat\beta_0, \hat\beta_1, \hat\beta_2$ minimising

$$
\sum_{i=1}^{n} K_{i0}\,\bigl(y_i - \beta_0 - \beta_1 x_i - \beta_2 x_i^2\bigr)^2,
$$

and predict $\hat f(x_0) = \hat\beta_0 + \hat\beta_1 x_0 + \hat\beta_2 x_0^2$. (Local-linear drops the $\beta_2$ term and matches ISLP Eq. 7.14.)

### Tricube kernel (R `loess()` default)

Let $x_\kappa$ denote the $k$-th nearest neighbour of $x_0$, where $k = \lfloor s \cdot n\rfloor$ is set by the span $s$. The tricube weight is

$$
K_{i0} \;=\;
\Bigl(\,
1 - \Bigl| \frac{x_0 - x_i}{x_0 - x_\kappa} \Bigr|^3
\,\Bigr)_{\!+}^{\,3}.
$$

Key properties:
- $K_{i0} = 1$ at $x_i = x_0$ (max weight at the target).
- $K_{i0} = 0$ at $|x_0 - x_i| \ge |x_0 - x_\kappa|$ (everything outside the neighbourhood gets weight zero, *exactly*).
- Smooth (twice-differentiable) decay in between — the smooth replacement for the hard rectangular cutoff of KNN.
- Normalised so that the boundary point of the neighbourhood gets weight 0 and the closest neighbour gets weight 1 — span $s = k/n$ is the only hyperparameter that matters.

This is the formula that lets the prof's "smooth KNN" framing be made precise. ISLP doesn't give it; books on local regression do.

---

## 5. GAM as a block-structured design matrix (additive-OLS regime)

[[L16-beyondlinear-1|L16]], [[L17-trees-1|L17]], [[generalized-additive-models]], slide deck §"Additive Models", Exercise 7.4

ISLP §7.7.1 says GAMs with basis-function components are fit "as a big regression onto spline basis variables and dummy variables, all packed into one big regression matrix." It then shows Figure 7.11 and moves on. It does **not** write the design matrix. The slide deck does. The construction is the object Exercise 7.4 asks you to build by hand.

### The wage-GAM template (slides, Exercise 7.4)

Model:
$$
\texttt{wage} = \beta_0 + f_1(\texttt{age}) + f_2(\texttt{year}) + f_3(\texttt{education}) + \varepsilon,
$$

with $f_1$ a cubic spline in age with knots at 40 and 60, $f_2$ a natural cubic spline in year with one interior knot at 2006, and $f_3$ a dummy-coded factor in education with 5 levels (`< HS Grad` as baseline).

#### Per-predictor blocks

**$\mathbf{X}_1$** (cubic spline in age, 5 columns from truncated-power basis):

$$
\mathbf{X}_1 =
\begin{pmatrix}
 x_{11} & x_{11}^2 & x_{11}^3 & (x_{11} - 40)^3_+ & (x_{11} - 60)^3_+ \\
 x_{21} & x_{21}^2 & x_{21}^3 & (x_{21} - 40)^3_+ & (x_{21} - 60)^3_+ \\
 \vdots & \vdots & \vdots & \vdots & \vdots \\
 x_{n1} & x_{n1}^2 & x_{n1}^3 & (x_{n1} - 40)^3_+ & (x_{n1} - 60)^3_+
\end{pmatrix}.
$$

**$\mathbf{X}_2$** (natural cubic spline in year, 2 columns from the simplified basis in §3 above):

$$
\mathbf{X}_2 =
\begin{pmatrix}
 x_{12} & \tfrac{1}{6}(x_{12} - 2003)^3_+ - \tfrac{1}{3}(x_{12} - 2006)^3_+ + \tfrac{1}{6}(x_{12} - 2009)^3_+ \\
 x_{22} & \tfrac{1}{6}(x_{22} - 2003)^3_+ - \tfrac{1}{3}(x_{22} - 2006)^3_+ + \tfrac{1}{6}(x_{22} - 2009)^3_+ \\
 \vdots & \vdots \\
 x_{n2} & \tfrac{1}{6}(x_{n2} - 2003)^3_+ - \tfrac{1}{3}(x_{n2} - 2006)^3_+ + \tfrac{1}{6}(x_{n2} - 2009)^3_+
\end{pmatrix}.
$$

**$\mathbf{X}_3$** (dummy-coded education, 4 columns for 5 levels with `<HSG` as reference):

$$
\mathbf{X}_3 =
\begin{pmatrix}
 \mathbb{1}(x_{13} = \mathrm{HSG}) & \mathbb{1}(x_{13} = \mathrm{SC}) & \mathbb{1}(x_{13} = \mathrm{CG}) & \mathbb{1}(x_{13} = \mathrm{AD}) \\
 \mathbb{1}(x_{23} = \mathrm{HSG}) & \mathbb{1}(x_{23} = \mathrm{SC}) & \mathbb{1}(x_{23} = \mathrm{CG}) & \mathbb{1}(x_{23} = \mathrm{AD}) \\
 \vdots & \vdots & \vdots & \vdots \\
 \mathbb{1}(x_{n3} = \mathrm{HSG}) & \mathbb{1}(x_{n3} = \mathrm{SC}) & \mathbb{1}(x_{n3} = \mathrm{CG}) & \mathbb{1}(x_{n3} = \mathrm{AD})
\end{pmatrix}.
$$

#### The full GAM design

Stack horizontally, intercept up front:

$$
\mathbf{X} \;=\;
\begin{pmatrix}
\mathbf{1} & \mathbf{X}_1 & \mathbf{X}_2 & \mathbf{X}_3
\end{pmatrix}.
$$

Total column count = $1 + 5 + 2 + 4 = 12$. Fit by OLS:

$$
\hat{\mathbf{y}} \;=\; \mathbf{X}(\mathbf{X}^\mathsf{T}\mathbf{X})^{-1}\mathbf{X}^\mathsf{T}\mathbf{y}.
$$

This is what `gam(wage ~ bs(age, knots=c(40,60)) + ns(year, knots=2006) + education)` does internally **when all components are basis-function** (no `s()` smoothing-spline or `lo()` LOESS terms — those require backfitting, which is out of scope as a derivation).

### Column-space invariance (the Exercise-7.4 punchline)

This is the conceptual delta that ISLP doesn't make explicit. The hand-built $\mathbf{X}$ (truncated-power basis) and R's `gam()`-built design matrix (B-spline basis under the hood) are *not equal column-by-column*, yet they produce **the same fitted values**:

$$
\hat{\mathbf{y}}_{\text{hand}} \;=\; \hat{\mathbf{y}}_{\text{gam}}.
$$

The mechanism: both matrices span the **same column space** (the space of natural cubic splines with the given knots ⊕ the dummy-encoded education space). The OLS projection $\mathbf{X}(\mathbf{X}^\mathsf{T}\mathbf{X})^{-1}\mathbf{X}^\mathsf{T}$ depends *only* on this column space, not on the particular basis used to write it. The individual $\hat\beta_j$ change with the basis; $\hat{\mathbf{y}}$ does not.

This is what Exercise 7.4's punchline question — *"How can `myhat` equal `yhat` when the design matrices differ?"* — is fishing for, and the kind of "method-comparison" question the prof flagged as exam-likely.

---

## 6. Degrees-of-freedom cheat-sheet (consolidated)

[[L16-beyondlinear-1|L16]], [[L17-trees-1|L17]], [[regression-splines]], [[smoothing-splines]], [[step-functions]], [[polynomial-regression]]

ISLP scatters the dof counts across §7.1, §7.2, §7.4 (main text + footnote 4), and §7.5.2. The prof drilled them together because the 2024 Q2c / 2025 Q4e(i) exam patterns turn on precise counting (the 2025 paper was *deliberately ambiguous* about whether the intercept counted, so reading the question carefully is the trap). Worth having in one place.

For each method below, "params" = total number of fitted real numbers; "df incl. intercept" / "df excl. intercept" splits out the convention.

| Method | Parameters | df (incl. intercept) | df (excl. intercept) |
|---|---|---|---|
| Polynomial degree $d$ | $d + 1$ | $d + 1$ | $d$ |
| Step function with $K$ cutpoints | $K + 1$ | $K + 1$ | $K$ |
| Linear spline, $K$ knots | $K + 2$ | $K + 2$ | $K + 1$ |
| Cubic spline, $K$ knots | $K + 4$ | $K + 4$ | $K + 3$ |
| Natural cubic spline, $K$ interior knots | $K + 2$ | $K + 2$ | $K + 1$ |
| Smoothing spline | $n$ nominal | $\mathrm{tr}(\mathbf{S}_\lambda)$ effective | same (non-integer) |

The cubic-spline cell uses the slide-deck count from §2 above ($M + K = 4 + K$). The natural-cubic-spline cell follows from §3 ($K + 1$ non-intercept columns plus the intercept). The smoothing-spline cell is the **effective** df — bounded between 2 (when $\lambda \to \infty$, $\hat g$ is the OLS line) and $n$ (when $\lambda = 0$, $\hat g$ interpolates every $y_i$); see ISLP §7.5.2 for the underlying formula.

### The intercept-counting trap, restated

The 2024 paper called a natural cubic spline with 3 cut points "4 dof" — that's the **excl.-intercept** convention applied to a model with $K = 3$ interior knots → $K + 1 = 4$. The 2025 paper called `bs(age, knots = quantile(age, c(0.2, 0.4, 0.6, 0.8)))` (a *plain* cubic spline with $K = 4$ knots) "7 dof" — that's also excl. intercept: $K + 3 = 7$. **In both cases the intercept was already supplied by the surrounding `gam(...)` model.** When in doubt: ask whether the question already gives you an intercept; if yes, subtract one from the incl.-intercept count.

---

## 7. Smoothing-spline LOOCV shortcut: parallel to the OLS hat-matrix shortcut

[[L16-beyondlinear-1|L16]], [[smoothing-splines]], slide deck §"The smoother matrix"

ISLP §7.5.2 gives the LOOCV shortcut for smoothing splines:

$$
\mathrm{RSS}_{cv}(\lambda) \;=\; \sum_{i=1}^{n} \left( \frac{y_i - \hat g_\lambda(x_i)}{1 - \{\mathbf{S}_\lambda\}_{ii}} \right)^{\!2},
$$

and remarks (footnote 5) that "we have a very similar formula (5.2) in Chapter 5 for least squares linear regression." It stops there. The slide deck and the prof make the **structural parallel** explicit, and it generalizes to "any linear smoother."

A **linear smoother** is any method whose fitted values are linear in $\mathbf{y}$:

$$
\hat{\mathbf{y}} \;=\; \mathbf{S}\,\mathbf{y}
$$

for some $n \times n$ smoother matrix $\mathbf{S}$. The members of this family in the course:

| Method | Smoother matrix $\mathbf{S}$ |
|---|---|
| OLS (incl. basis-function OLS: polynomial, step, regression spline) | Hat matrix $\mathbf{H} = \mathbf{X}(\mathbf{X}^\mathsf{T}\mathbf{X})^{-1}\mathbf{X}^\mathsf{T}$ |
| [[ridge-regression|Ridge regression]] | $\mathbf{X}(\mathbf{X}^\mathsf{T}\mathbf{X} + \lambda \mathbf{I})^{-1}\mathbf{X}^\mathsf{T}$ |
| Smoothing spline | $\mathbf{S}_\lambda$ from the curvature-penalty objective |
| LOESS (linear or quadratic local fit) | A weight-dependent $\mathbf{S}$ (R reports `trace.hat`) |

For **any** linear smoother, the LOOCV shortcut takes the same form:

$$
\mathrm{RSS}_{cv}(\lambda) \;=\; \sum_{i=1}^{n} \left( \frac{y_i - \hat y_i}{1 - \{\mathbf{S}\}_{ii}} \right)^{\!2},
$$

with $\{\mathbf{S}\}_{ii}$ replacing the OLS hat-matrix diagonal $h_{ii}$ (ISLP §5.1.2 Eq. 5.2). The structural fact "linear smoother ⇒ LOOCV is essentially free" is what the prof keeps pointing at: it's why one fit is enough to do LOOCV for OLS, ridge, and smoothing splines, and it's also why the **effective dof = trace of the smoother** definition is natural — both quantities are properties of the same $\mathbf{S}$.

This linear-smoother frame, with $\mathrm{df} = \mathrm{tr}(\mathbf{S})$ and the unified LOOCV formula, is the cross-cutting object that ties together module 3 (OLS), module 6 (ridge), and module 7 ([[smoothing-splines|smoothing splines]], LOESS). ISLP discusses each piece in isolation; the prof's slide deck merges them.

---

## Notation / terminology drift

A few places where the prof's notation diverges from ISLP's. Worth noting only because they show up in the worked formulas above.

- **Boundary knots.** ISLP §7.4 calls them "boundary knots" but introduces them only verbally and does not separate them from interior knots in any formula. The prof uses $c_0$ and $c_{K+1}$ explicitly (or sometimes "$a$, $b$") in the natural-cubic-spline basis above. They are min and max of $x$ by default; they impose constraints but do not add basis columns.
- **Spline order vs degree.** The prof's slides use $M$ = "order" with $M - 1$ = "degree" (so $M = 4$ ⇒ cubic spline). ISLP uses only "degree" ($d = 3$). Order is the standard convention from approximation theory; degree is more common in textbook statistics.
- **Truncated-power notation.** ISLP writes $h(x, \xi) = (x - \xi)^3_+$ with a single knot $\xi$. The prof writes $(x - c_j)^{M-1}_+$ with $c_j$ for the $j$-th knot. Both mean the same thing.
- **Knot variable.** ISLP §7.4 uses $\xi_1, \ldots, \xi_K$. The prof uses $c_1, \ldots, c_K$. (And $\xi$ shows up nowhere else in the course.)
- **"GAM" vs "AM".** The prof's slides distinguish: an *additive model* (AM) is the Gaussian-response form $y = \beta_0 + \sum f_j(x_j) + \varepsilon$; a *generalized additive model* (GAM) is the same idea with a GLM link (logit, etc.). ISLP uses "GAM" for both. This course only ever uses GAMs with the Gaussian (identity) and logit links; the broader GLM-link generality is name-checked only.
- **`bs` vs `BS`.** The R interface labels cubic-spline columns `bs` ("B-spline"). The prof: *"I don't know why they call it BS."* [[L16-beyondlinear-1|L16]] B-spline columns and truncated-power columns span the same column space (different basis, same fit), so the relabeling is cosmetic. The B-spline algorithmic construction itself is out of scope.
- **Smoother matrix symbol.** ISLP uses $\mathbf{S}_\lambda$ (with the $\lambda$ subscript explicit). The slide deck drops the subscript and writes $\mathbf{S}$. Same object.
- **"df" overloading.** ISLP uses "degrees of freedom" for three different things — number of fitted parameters (cubic-spline = $K + 4$), the `df=` argument to `bs()` / `ns()` / `s()` in R (which is the parameter count *excluding* the intercept, i.e. the argument is what R will add to the supplied formula), and the effective df $\mathrm{tr}(\mathbf{S}_\lambda)$ for smoothing splines. The prof keeps the same three uses but doesn't always flag the convention switch. See §6 above for the consolidated table.
