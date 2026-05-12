---
module: 03-linreg
isl-ch: 3
---

# Module 03 — Book delta

Module 03 is the **heaviest-delta module** of the course. ISLP ch. 3 covers simple-LR algebra in detail (eq. 3.4), the SE / CI / t-test machinery for the simple case (eq. 3.7–3.10), the $F$-statistic formula (eq. 3.23–3.24), $R^2$ (eq. 3.17) and adjusted $R^2$ (in passing), categorical encoding, interactions, polynomial regression, the "potential problems" list, and the simple-LR leverage formula (eq. 3.37). But the **matrix-form theory** that Benjamin built in L06 is largely absent: the book explicitly says of multiple regression "the coefficient estimates have somewhat complicated forms that are most easily represented using matrix algebra. For this reason, we do not provide them here" (§3.2.1). Everything downstream of that statement — the closed-form derivation, the hat matrix and its properties, the multivariate-normal sampling distribution of $\hat{\boldsymbol\beta}$, the residual covariance $\sigma^2(\mathbf{I} - \mathbf{H})$, the MLE-equals-LS proof, the matrix-form $\mathrm{SE}(\hat\beta_j)$ via $(\mathbf{X}^\top\mathbf{X})^{-1}$, the matrix CI / PI formulas — is delta and is reproduced here in full.

Out-of-scope material per `docs/scope.md` (F-test mechanics beyond stating the null, VIF, Moore–Penrose details, formal normality tests, spectral theory of $\mathbf{X}^\top\mathbf{X}$) is excluded.

---

## 1. The matrix-form linear model and design matrix

[[[L06-linreg-2|L06]], [[linear-regression]], [[design-matrix-and-hat-matrix]]]

ISLP §3.2 presents the multiple-LR model in scalar form (eq. 3.19) but never writes the matrix form $\mathbf{y} = \mathbf{X}\boldsymbol\beta + \boldsymbol\varepsilon$ or defines the design matrix as an object. Benjamin's matrix-form setup is the foundation for everything that follows.

### Model

$$
\mathbf{y} = \mathbf{X}\boldsymbol\beta + \boldsymbol\varepsilon, \qquad \boldsymbol\varepsilon \sim N_n(\mathbf{0},\,\sigma^2 \mathbf{I}_n).
$$

Dimensions:

- $\mathbf{y} \in \mathbb{R}^{n \times 1}$ — response vector.
- $\mathbf{X} \in \mathbb{R}^{n \times (p+1)}$ — **design matrix** (Benjamin's grudging name: "*never understood why. It's not really a design of any kind. But it's what people call it*" [[L06-linreg-2|L06]]).
- $\boldsymbol\beta \in \mathbb{R}^{(p+1) \times 1}$ — parameter vector, intercept $\beta_0$ plus $p$ slopes.
- $\boldsymbol\varepsilon \in \mathbb{R}^{n \times 1}$ — error vector.

### Explicit design matrix

The leading column of ones absorbs the intercept so the bias term disappears from the matrix equation. "Behind this beta is actually an X. It's just all the values of X are one. So you don't need to write it" [[L06-linreg-2|L06]].

$$
\mathbf{X} = \begin{bmatrix}
1 & x_{11} & x_{12} & \cdots & x_{1p} \\
1 & x_{21} & x_{22} & \cdots & x_{2p} \\
\vdots & \vdots & \vdots &        & \vdots \\
1 & x_{n1} & x_{n2} & \cdots & x_{np}
\end{bmatrix}.
$$

### Multivariate-normal form of the assumptions

ISLP §3.1.2 gives the scalar assumptions ($\varepsilon_i$ uncorrelated, common $\sigma^2$). Benjamin restates them as one $n$-dimensional multivariate-normal statement:

$$
\mathsf{E}[\boldsymbol\varepsilon] = \mathbf{0}, \qquad \mathrm{Cov}(\boldsymbol\varepsilon) = \sigma^2 \mathbf{I}_n.
$$

The covariance matrix has $\sigma^2$ on the diagonal, zeros off — so the **off-diagonals encode independence** (assumption 5 in L05's list), the **diagonal equality encodes homoscedasticity** (assumption 3). Geometrically the error vector is a spherical $n$-dimensional Gaussian — "no matter which direction you look in, has the same variance, and it's just kind of a big n-dimensional bell" [[L06-linreg-2|L06]].

### Notation gotcha (flagged by the prof)

"You can define $p$ as including or not the intercept or bias term. This is just a note for those who are taking both classes that the notation is different in the books" [[L06-linreg-2|L06]]. In this delta file, $p$ = number of slopes, so $\mathbf{X}$ has $p+1$ columns and df-for-noise is $n - p - 1$. ISLP uses the same convention (eq. 3.25).

### Classical regime

"$n \gg p$: more data points than parameters." [[L06-linreg-2|L06]] When $p > n$, $\mathbf{X}^\top\mathbf{X}$ is necessarily singular (rank $\le n < p+1$) and OLS has no unique solution. This sets up module 6.

---

## 2. The OLS derivation in matrix form

[[[L06-linreg-2|L06]], [[least-squares-and-mle]]]

ISLP states (eq. 3.4) the simple-LR closed form, and **explicitly declines to derive the multiple-LR matrix form** (§3.2.1). Benjamin did it on the board, three lines, and flagged it as exam-template material via Exercise 6.1a / [[L12-modelsel-1|L12]]. Here is the full derivation.

### Step 1 — write RSS in matrix form

$$
\mathrm{RSS}(\boldsymbol\beta) = (\mathbf{y} - \mathbf{X}\boldsymbol\beta)^\top (\mathbf{y} - \mathbf{X}\boldsymbol\beta).
$$

### Step 2 — expand

$$
\mathrm{RSS}(\boldsymbol\beta)
= \mathbf{y}^\top\mathbf{y} - \mathbf{y}^\top\mathbf{X}\boldsymbol\beta - \boldsymbol\beta^\top\mathbf{X}^\top\mathbf{y} + \boldsymbol\beta^\top\mathbf{X}^\top\mathbf{X}\boldsymbol\beta
= \mathbf{y}^\top\mathbf{y} - 2\boldsymbol\beta^\top\mathbf{X}^\top\mathbf{y} + \boldsymbol\beta^\top\mathbf{X}^\top\mathbf{X}\boldsymbol\beta.
$$

The two cross terms $\mathbf{y}^\top\mathbf{X}\boldsymbol\beta$ and $\boldsymbol\beta^\top\mathbf{X}^\top\mathbf{y}$ combine because each is a $1\times1$ scalar and a scalar equals its transpose.

### Step 3 — differentiate w.r.t. $\boldsymbol\beta$, set to zero

Using $\partial(\mathbf{a}^\top\boldsymbol\beta)/\partial\boldsymbol\beta = \mathbf{a}$ and $\partial(\boldsymbol\beta^\top \mathbf{A} \boldsymbol\beta)/\partial\boldsymbol\beta = (\mathbf{A} + \mathbf{A}^\top)\boldsymbol\beta = 2\mathbf{A}\boldsymbol\beta$ for symmetric $\mathbf{A}$:

$$
\frac{\partial \mathrm{RSS}}{\partial \boldsymbol\beta} = -2\mathbf{X}^\top\mathbf{y} + 2\mathbf{X}^\top\mathbf{X}\boldsymbol\beta = \mathbf{0}.
$$

### Step 4 — normal equations

$$
\boxed{\;\mathbf{X}^\top\mathbf{X}\,\hat{\boldsymbol\beta} = \mathbf{X}^\top\mathbf{y}\;} \qquad\text{(normal equations)}.
$$

### Step 5 — invert when full rank

If $\mathbf{X}^\top\mathbf{X}$ is invertible (i.e. $\mathbf{X}$ has full column rank $p+1$, which requires $n \ge p+1$ and no [[collinearity]]):

$$
\boxed{\;\hat{\boldsymbol\beta} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}.\;}
$$

### Uniqueness

"I'm not showing, but you can prove if you want to take more derivatives, that this problem has a unique solution. There's only one solution" [[L06-linreg-2|L06]]. The second-derivative is $2\mathbf{X}^\top\mathbf{X}$, positive definite when $\mathbf{X}$ has full column rank — so the stationary point is a strict minimum, and it is the unique one.

### Why this matters at the meta-level

"Most of the time, you actually have to go iteratively… most of the time when we're trying to find this peak, we have to like climb up and go there. In fact, I would argue most of machine learning is finding good tricks to get to that peak… But in the case of linear regression with full-rank X, just right to the top. Very convenient" [[L06-linreg-2|L06]]. This is *why* OLS is the canonical model for everything downstream (CIs, t-tests, exact distributions): we have the estimator in closed form, so every other quantity is also exact.

### Reduction to simple-LR

In the $p=1$ case, $\mathbf{X} = [\mathbf{1}\ \mathbf{x}]$, and direct calculation of $(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ reproduces ISLP eq. 3.4:

$$
\hat\beta_1 = \frac{\sum_i (x_i - \bar x)(y_i - \bar y)}{\sum_i (x_i - \bar x)^2}, \qquad \hat\beta_0 = \bar y - \hat\beta_1 \bar x.
$$

This consistency check is the recommended exercise [[L06-linreg-2|L06]].

---

## 3. The MLE ⇔ least-squares equivalence

[[[L05-linreg-1|L05]], [[L06-linreg-2|L06]], [[L27-summary|L27]], [[least-squares-and-mle]]]

ISLP never proves this. It is the **prof's flagged theory-question template** for the exam: "*I do generally like to keep one theory question. … assume an additive Gaussian error model … Show that maximum likelihood and least squares are equivalent in $\theta$*" [[L27-summary|L27]]. Reproduced here in full.

### Setup

Assume $\varepsilon_i \overset{\text{iid}}\sim N(0, \sigma^2)$, so $y_i \mid \mathbf{x}_i \sim N(\mathbf{x}_i^\top\boldsymbol\beta,\, \sigma^2)$.

### Likelihood

The joint density of $(y_1, \dots, y_n)$ given $\mathbf{X}$:

$$
L(\boldsymbol\beta, \sigma^2) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(y_i - \mathbf{x}_i^\top\boldsymbol\beta)^2}{2\sigma^2}\right).
$$

### Log-likelihood

$$
\log L(\boldsymbol\beta, \sigma^2) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - \mathbf{x}_i^\top\boldsymbol\beta)^2.
$$

### The argument

The first term $-\frac{n}{2}\log(2\pi\sigma^2)$ does not depend on $\boldsymbol\beta$. The factor $1/(2\sigma^2)$ in front of the sum is a positive constant that doesn't change the location of the maximum. So

$$
\arg\max_{\boldsymbol\beta} \log L(\boldsymbol\beta, \sigma^2) = \arg\max_{\boldsymbol\beta} \left[-\sum_{i=1}^n (y_i - \mathbf{x}_i^\top\boldsymbol\beta)^2\right] = \arg\min_{\boldsymbol\beta} \sum_{i=1}^n (y_i - \mathbf{x}_i^\top\boldsymbol\beta)^2 = \arg\min_{\boldsymbol\beta} \mathrm{RSS}(\boldsymbol\beta).
$$

So $\hat{\boldsymbol\beta}_{\text{MLE}} = \hat{\boldsymbol\beta}_{\text{LS}}$. $\square$

### MLE for $\sigma^2$ (as a side-effect)

Differentiating the log-likelihood w.r.t. $\sigma^2$ and solving gives $\hat\sigma^2_{\text{MLE}} = \mathrm{RSS}/n$. The unbiased estimator (which is what is used everywhere else in the course, including residual standard error) is $\hat\sigma^2 = \mathrm{RSS}/(n - p - 1)$, with the $n - p - 1$ accounting for the $p+1$ parameters consumed by $\hat{\boldsymbol\beta}$. Benjamin says of the difference: "if $n = 300$ it barely matters" [[L05-linreg-1|L05]].

### What other loss → what other distribution

A useful by-product the prof drew on the board: a different choice of penalty implies a different error distribution.

| Loss | MLE-equivalent error distribution | Property |
|---|---|---|
| $\sum_i (y_i - \hat y_i)^2$ | $N(0, \sigma^2)$ Gaussian | quadratic cost; outlier-sensitive |
| $\sum_i \lvert y_i - \hat y_i\rvert$ | Laplace (double-exponential) | linear cost; **robust to outliers** |
| $\sum_i (y_i - \hat y_i)^4$ | exotic, never used | "would really, really penalize anything far away" [[L05-linreg-1|L05]] |

"If we had our data was like this and then there was a point here, that point would have a stronger effect when fitting the model with a least squares fit, whereas a Laplace fit it wouldn't be pulling it as strongly" [[L05-linreg-1|L05]].

### Pitfall the prof himself stumbled on

"A student caught the prof on a sign during the L27 walkthrough" [[least-squares-and-mle]]. On the exam, state explicitly that maximizing $\log L$ is the same as minimizing $-\log L$, which is the same as minimizing RSS up to constants. The sign-flip is the standard place to lose a point.

---

## 4. The hat matrix $\mathbf{H}$

[[[L06-linreg-2|L06]], [[L08-classif-2|L08]], [[design-matrix-and-hat-matrix]]]

ISLP §3.3.3 mentions $h_i$ (the diagonal element only) in eq. 3.37 for simple LR and notes "there is a simple extension of $h_i$ to the case of multiple predictors, though we do not provide the formula here." Everything else about $\mathbf{H}$ is delta. This is the matrix the prof said "*has all the shit you need to get your hats for your parameters. So it's called the hat matrix*" [[L06-linreg-2|L06]].

### Definition

$$
\boxed{\;\mathbf{H} \;=\; \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top \;\in\; \mathbb{R}^{n\times n}.\;}
$$

### Why "hat matrix"

Predictions are obtained from $\mathbf{y}$ by applying $\mathbf{H}$:

$$
\hat{\mathbf{y}} = \mathbf{X}\hat{\boldsymbol\beta} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y} = \mathbf{H}\mathbf{y}.
$$

"In math we call it a hat. It's a pointy hat. But it's a hat. And so this matrix H has all the shit you need to get your hats for your parameters" [[L06-linreg-2|L06]].

### Properties (provable in two lines each)

These are the structural facts Benjamin emphasizes and that ISLP never lists.

**(P1) Symmetric.**

$$
\mathbf{H}^\top = \bigl(\mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\bigr)^\top = \mathbf{X}\bigl((\mathbf{X}^\top\mathbf{X})^{-1}\bigr)^\top\mathbf{X}^\top = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top = \mathbf{H},
$$

using that $\mathbf{X}^\top\mathbf{X}$ is symmetric, so its inverse is symmetric.

**(P2) Idempotent.** $\mathbf{H}^2 = \mathbf{H}$.

$$
\mathbf{H}^2 = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\underbrace{\mathbf{X}^\top\mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}}_{=\mathbf{I}}\mathbf{X}^\top = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top = \mathbf{H}.
$$

**(P3) Orthogonal projection.** $\mathbf{H}$ is the orthogonal projection of $\mathbb{R}^n$ onto the column space of $\mathbf{X}$. Combined: a symmetric idempotent matrix is exactly an orthogonal projector. Geometrically: $\hat{\mathbf{y}}$ is the closest point to $\mathbf{y}$ in the column-space of $\mathbf{X}$ (which is exactly what least squares means).

**(P4) Residual projector.** $\mathbf{I} - \mathbf{H}$ is also symmetric and idempotent, and projects onto the orthogonal complement (the "residual space"):

$$
\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}} = (\mathbf{I} - \mathbf{H})\mathbf{y}.
$$

**(P5) Orthogonality of fitted values and residuals.**

$$
\hat{\mathbf{y}}^\top \mathbf{e} = (\mathbf{H}\mathbf{y})^\top(\mathbf{I} - \mathbf{H})\mathbf{y} = \mathbf{y}^\top \mathbf{H}(\mathbf{I} - \mathbf{H})\mathbf{y} = \mathbf{y}^\top(\mathbf{H} - \mathbf{H}^2)\mathbf{y} = \mathbf{0}.
$$

The fitted values and the residuals are orthogonal vectors in $\mathbb{R}^n$.

**(P6) Trace = rank = $p + 1$.**

$$
\mathrm{tr}(\mathbf{H}) = \mathrm{tr}\bigl(\mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\bigr) = \mathrm{tr}\bigl((\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{X}\bigr) = \mathrm{tr}(\mathbf{I}_{p+1}) = p+1,
$$

using the cyclic property of trace. Equivalently $\sum_{i=1}^n h_{ii} = p+1$. Average leverage is $(p+1)/n$ (ISLP §3.3.3 mentions this fact without deriving it).

**(P7) Each diagonal entry $h_{ii} \in [1/n, 1]$.** Lower bound follows from including the intercept column; upper bound from idempotency.

**(P8) Leverage in multiple LR (the formula ISLP declined to give).**

$$
h_{ii} = \mathbf{x}_i^\top (\mathbf{X}^\top\mathbf{X})^{-1} \mathbf{x}_i,
$$

where $\mathbf{x}_i$ is the $i$-th row of $\mathbf{X}$ written as a column vector (including the leading 1).

**(P9) Leverage depends only on $\mathbf{X}$, not on $\mathbf{y}$.** So a high-leverage point can be flagged from the design alone, before any response data is observed.

### Residual covariance

The fact ISLP never states. Starting from $\mathbf{e} = (\mathbf{I} - \mathbf{H})\mathbf{y}$ and $\mathrm{Cov}(\mathbf{y}) = \sigma^2 \mathbf{I}$:

$$
\boxed{\;\mathrm{Cov}(\mathbf{e}) = (\mathbf{I} - \mathbf{H})\,\sigma^2 \mathbf{I}\,(\mathbf{I} - \mathbf{H})^\top = \sigma^2 (\mathbf{I} - \mathbf{H}).\;}
$$

Implications:

- $\mathrm{Var}(e_i) = \sigma^2(1 - h_{ii})$ — raw residuals have **unequal variances**. High-leverage points have **smaller** residual variance (they pull the fit toward themselves, so their residual is small).
- $\mathrm{Cov}(e_i, e_j) = -\sigma^2 h_{ij}$ — raw residuals are **correlated**, even when the true errors $\varepsilon_i$ are independent.

This motivates **[[residual-diagnostics|standardized residuals]]**:

$$
\tilde r_i = \frac{e_i}{\hat\sigma\sqrt{1 - h_{ii}}},
$$

which have approximately unit variance and let the QQ plot / residuals-vs-fitted plot be read with the assumed Gaussian behaviour. "Your betas stay the same. It's just a way to say, is my model any good?" [[L08-classif-2|L08]].

**Studentized residuals** swap in $\hat\sigma_{(i)}$ (the residual SE computed from the data with point $i$ deleted) to remove the circular use of $y_i$ in fitting and evaluating point $i$. For $n \gtrsim 50$, $\tilde r_i$ and $r^*_i$ are essentially indistinguishable [[L08-classif-2|L08]].

### Leverage in simple LR (the prof flagged this as the exercise question)

Direct algebra on $h_{ii} = \mathbf{x}_i^\top (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{x}_i$ for $\mathbf{X} = [\mathbf{1}\ \mathbf{x}]$ gives

$$
h_{ii} = \frac{1}{n} + \frac{(x_i - \bar x)^2}{\sum_{j=1}^n (x_j - \bar x)^2}.
$$

ISLP states this (eq. 3.37) but the derivation is delta. The point: $h_{ii}$ grows with distance from $\bar x$ — extreme $x$-values are high-leverage.

### LOOCV shortcut for OLS

For OLS fits only, [[leave-one-out-cv|leave-one-out cross-validation]] can be computed from one full-data fit using the hat matrix:

$$
\mathrm{CV}_n = \frac{1}{n}\sum_{i=1}^n \left(\frac{y_i - \hat y_i}{1 - h_{ii}}\right)^2.
$$

ISLP §5.1.2 states this for OLS without much justification. The reason it works is exactly **P9** — leverage depends only on $\mathbf{X}$, not on $\mathbf{y}$ — so leaving out the $y_i$ leaves $h_{ii}$ unchanged and the leave-one-out fitted value can be recovered analytically. Owned by [[leave-one-out-cv]] in module 5.

---

## 5. Sampling distribution of $\hat{\boldsymbol\beta}$

[[[L05-linreg-1|L05]], [[L06-linreg-2|L06]], [[sampling-distribution-of-beta]]]

ISLP gives diagonal SE formulas for the **simple-LR** case in eq. 3.8 and waves vaguely at multiple LR. The clean multivariate theorem and its derivation are delta and are the load-bearing fact for all of regression inference.

### Theorem

Under the Gaussian linear model $\mathbf{y} = \mathbf{X}\boldsymbol\beta + \boldsymbol\varepsilon$ with $\boldsymbol\varepsilon \sim N_n(\mathbf{0}, \sigma^2 \mathbf{I})$:

$$
\boxed{\;\hat{\boldsymbol\beta} \;\sim\; N_{p+1}\bigl(\boldsymbol\beta,\;\sigma^2 (\mathbf{X}^\top\mathbf{X})^{-1}\bigr).\;}
$$

### Derivation (three lines)

Write $\mathbf{C} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$, so $\hat{\boldsymbol\beta} = \mathbf{C}\mathbf{y}$. Use $\mathbf{y} \sim N_n(\mathbf{X}\boldsymbol\beta,\,\sigma^2\mathbf{I})$ and the linear-transformation property of the multivariate normal: $\mathbf{C}\mathbf{y}$ is multivariate normal with

$$
\mathsf{E}[\hat{\boldsymbol\beta}] = \mathbf{C}\,\mathsf{E}[\mathbf{y}] = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{X}\boldsymbol\beta = \boldsymbol\beta,
$$

$$
\mathrm{Cov}(\hat{\boldsymbol\beta}) = \mathbf{C}\,\sigma^2\mathbf{I}\,\mathbf{C}^\top = \sigma^2 (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1} = \sigma^2 (\mathbf{X}^\top\mathbf{X})^{-1}. \qquad\square
$$

### Consequences

**(C1) Unbiasedness.** $\mathsf{E}[\hat{\boldsymbol\beta}] = \boldsymbol\beta$. "That's what we want. If it was biased then we'd be upset because then our model is not going to give us the right shit" [[L06-linreg-2|L06]].

**(C2) Per-coefficient variance.** $\mathrm{Var}(\hat\beta_j) = \sigma^2 [(\mathbf{X}^\top\mathbf{X})^{-1}]_{jj} \equiv \sigma^2 c_{jj}$ — the $j$-th diagonal of $(\mathbf{X}^\top\mathbf{X})^{-1}$. The SE estimator is $\widehat{\mathrm{SE}}(\hat\beta_j) = \hat\sigma\sqrt{c_{jj}}$.

**(C3) Coefficients are correlated.** The off-diagonals of $(\mathbf{X}^\top\mathbf{X})^{-1}$ are generally nonzero. In simple LR, $\mathrm{Cov}(\hat\beta_0, \hat\beta_1) = -\sigma^2\bar x / \sum_i(x_i - \bar x)^2$; zero iff $\bar x = 0$ (center the data and the intercept becomes uncorrelated from the slope). ISLP nowhere states this.

**(C4) [[collinearity|Collinearity]] blow-up.** As columns of $\mathbf{X}$ become near-linearly-dependent, $\mathbf{X}^\top\mathbf{X}$ becomes near-singular, its inverse's entries blow up, and individual $\mathrm{Var}(\hat\beta_j) \to \infty$. The prof's load-bearing observation: "*This factor X transpose X comes into play in particular when two variables are basically the same, because then they can trade off each other and then this variance explodes*" [[L06-linreg-2|L06]]. ISLP discusses collinearity qualitatively in §3.3.3 but never connects it to the matrix algebra explicitly.

**(C5) Why centering helps numerics.** Centering $\mathbf{x}$ around its mean kills the $\bar x$ term in the off-diagonal of $(\mathbf{X}^\top\mathbf{X})^{-1}$ for simple LR (and reduces correlations between intercept and slopes in MLR), giving a better-conditioned inversion.

### Residual standard error (matrix-form derivation)

The unbiased estimator of $\sigma^2$ is

$$
\hat\sigma^2 = \frac{\mathrm{RSS}}{n - p - 1} = \frac{\mathbf{e}^\top\mathbf{e}}{n - p - 1}.
$$

Unbiasedness: $\mathsf{E}[\mathbf{e}^\top\mathbf{e}] = \mathsf{E}[\mathbf{y}^\top(\mathbf{I} - \mathbf{H})\mathbf{y}] = \sigma^2\,\mathrm{tr}(\mathbf{I} - \mathbf{H}) = \sigma^2(n - (p+1))$ using property P6 of $\mathbf{H}$. Dividing by $n - p - 1$ gives an unbiased estimator. ISLP states the divisor (eq. 3.25) without this derivation; the $n - p - 1$ is the rank of the residual projector $\mathbf{I} - \mathbf{H}$.

In simple LR this collapses to $n - 2$ (ISLP eq. 3.15). "Two degrees of freedom are eaten by $\hat\beta_0$ and $\hat\beta_1$" [[L05-linreg-1|L05]]. In general, "*$p+1$ degrees of freedom are eaten by the $p+1$ entries of $\hat{\boldsymbol\beta}$.*"

### Independence of $\hat{\boldsymbol\beta}$ and $\hat\sigma^2$

A classical result Benjamin invokes implicitly when justifying t-tests with df $n - p - 1$: under the Gaussian linear model, $\hat{\boldsymbol\beta}$ and $\mathrm{RSS}$ are independent random variables. Sketch: $\hat{\boldsymbol\beta} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ depends on $\mathbf{H}\mathbf{y}$; $\mathrm{RSS} = \mathbf{y}^\top(\mathbf{I} - \mathbf{H})\mathbf{y}$ depends on $(\mathbf{I} - \mathbf{H})\mathbf{y}$. The two random vectors $\mathbf{H}\mathbf{y}$ and $(\mathbf{I} - \mathbf{H})\mathbf{y}$ are jointly Gaussian and uncorrelated ($\mathrm{Cov} = \mathbf{H}\sigma^2(\mathbf{I}-\mathbf{H}) = \mathbf{0}$), hence independent. This independence is what licenses the t-statistic $\hat\beta_j / \widehat{\mathrm{SE}}(\hat\beta_j)$ to have an exact $t_{n-p-1}$ distribution under $H_0$, rather than just approximately Gaussian.

Walpole is the prof's recommended classical reference; the result is needed to get the $n - p - 1$ df exactly right [[sampling-distribution-of-beta]].

---

## 6. The $\hat{\boldsymbol\beta}$-aware t-statistic and matrix-form CI

[[[L06-linreg-2|L06]], [[t-test-and-significance]], [[confidence-and-prediction-intervals]]]

ISLP gives the simple-LR t-statistic (eq. 3.14) with $n-2$ df. The matrix-form version with $n - p - 1$ df and $\hat\sigma\sqrt{c_{jj}}$ — and the explicit pointer to which diagonal of $(\mathbf{X}^\top\mathbf{X})^{-1}$ — is delta.

### Per-coefficient t-test

$$
t_j = \frac{\hat\beta_j}{\widehat{\mathrm{SE}}(\hat\beta_j)} = \frac{\hat\beta_j}{\hat\sigma\sqrt{c_{jj}}} \;\sim\; t_{n - p - 1} \quad\text{under } H_0: \beta_j = 0,
$$

where $c_{jj} = [(\mathbf{X}^\top\mathbf{X})^{-1}]_{jj}$.

### Per-coefficient CI

$$
\hat\beta_j \;\pm\; t_{1-\alpha/2,\,n-p-1}\cdot \hat\sigma\sqrt{c_{jj}}.
$$

### CI for the mean response at $\mathbf{x}_0$

ISLP §3.2.2 mentions CIs for $\hat y_0$ verbally but does not give the matrix-form formula. Delta:

$$
\boxed{\;\mathbf{x}_0^\top \hat{\boldsymbol\beta} \;\pm\; t_{1-\alpha/2,\,n-p-1}\cdot \hat\sigma\sqrt{\mathbf{x}_0^\top (\mathbf{X}^\top\mathbf{X})^{-1} \mathbf{x}_0}.\;}
$$

Derivation: $\hat y_0 = \mathbf{x}_0^\top\hat{\boldsymbol\beta}$ is a linear function of the multivariate-normal $\hat{\boldsymbol\beta}$, so $\hat y_0 \sim N(\mathbf{x}_0^\top\boldsymbol\beta,\,\sigma^2 \mathbf{x}_0^\top(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{x}_0)$.

### PI for a future observation at $\mathbf{x}_0$

$$
\boxed{\;\mathbf{x}_0^\top \hat{\boldsymbol\beta} \;\pm\; t_{1-\alpha/2,\,n-p-1}\cdot \hat\sigma\sqrt{1 + \mathbf{x}_0^\top (\mathbf{X}^\top\mathbf{X})^{-1} \mathbf{x}_0}.\;}
$$

Derivation: a future observation $y_{\text{new}} = \mathbf{x}_0^\top\boldsymbol\beta + \varepsilon_{\text{new}}$ has $\varepsilon_{\text{new}} \sim N(0, \sigma^2)$ independent of the past data, so

$$
\mathrm{Var}(y_{\text{new}} - \hat y_0) = \mathrm{Var}(\hat y_0) + \sigma^2 = \sigma^2\bigl(1 + \mathbf{x}_0^\top(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{x}_0\bigr).
$$

The **+1 under the square root** is the irreducible noise — the source of "PI always wider than CI" [[L06-linreg-2|L06]].

### Band shape

Both bands are narrowest where $\mathbf{x}_0$ is near the centroid of the data (because $\mathbf{x}_0^\top(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{x}_0$ is small there) and fan out at the extremes. The CI band hugs the line; the PI band is wider by a constant-σ² floor.

---

## 7. The F-statistic and the partial F-statistic (statement only)

[[[L06-linreg-2|L06]], [[f-test]]]

**Scope flag.** Benjamin was emphatic: "*I'm going to say right now I probably won't ask any questions about an F-test. … I'm not going to make you compute it because I honestly don't care*" [[L06-linreg-2|L06]]. The mechanics are out of scope; the null hypothesis and the *why-you'd-use-it* reasoning are in. ISLP §3.2.2 gives both eq. 3.23 and eq. 3.24 in full. **No delta on F-test math.**

The one structural point that *is* delta and worth stating: $F_{1, n-p-1} = t_{n-p-1}^2$, so the F-test on a single coefficient is equivalent to the squared t-test [[f-test]]. ISLP §3.2.2 mentions this in a footnote (footnote 7) but does not derive it.

---

## 8. Notation and naming differences

### "Bias" vs "intercept"

ISLP uses "intercept" throughout. Benjamin **prefers "bias"** for $\beta_0$ and uses both interchangeably. "The bias is $\beta_0$" / "the intercept is $\beta_0$" — same object. [[L05-linreg-1|L05]]

### $p$ counting

ISLP and Benjamin both use $p$ to mean **number of slopes**, with the design matrix having $p+1$ columns and df $n - p - 1$. The convention is consistent across both sources, but Benjamin flagged that some books count the intercept inside $p$. [[L06-linreg-2|L06]]

### "Residuals are predictions of errors"

Benjamin draws a sharp distinction that ISLP does not:

> "The error terms are random variables and cannot be estimated. They can be predicted." [[L05-linreg-1|L05]]

So $\varepsilon_i$ is an unobservable random variable, and $e_i = y_i - \hat y_i$ is a **prediction** of $\varepsilon_i$, not an **estimate**. ISLP uses "estimate" loosely. Stating this distinction may earn marks on a careful T/F.

### "Design matrix"

ISLP uses "design matrix" with no commentary. Benjamin keeps the name but mocks it: "*It's often called the design matrix. The data. Never understood why. It's not really a design of any kind. But it's what people call it*" [[L06-linreg-2|L06]].

### Independence as the load-bearing assumption

ISLP §3.3.3 lists "correlation of error terms" as item 2 of six "potential problems" with no ranking. Benjamin **ranks the assumptions**, with independence (4 and 5 in his list) as the dangerous ones and Gaussian / zero-mean / homoscedastic as relatively benign: "*violations [of independence] ruin everything*" [[L05-linreg-1|L05]]. This is not a formula difference, but it is a framing the exam might test (e.g. "which assumption violation most invalidates the SE estimates?").

### "Main-effects rule" vs "hierarchical principle"

Same thing. ISLP §3.3.2 calls it the *hierarchical principle*. Benjamin calls it the **main-effects rule** [[L06-linreg-2|L06]] — verbatim: "*whenever you include an interaction, you want to include what is referred to as the main effects.*"

### "Statistical vs practical significance"

Not in ISLP. Benjamin's organizing framing for the t-test discussion: large $n$ makes everything statistically significant; the slope size is what tells you whether it actually matters. "*Significance is just sample size*" [[L05-linreg-1|L05]]. This is a framing the exam will likely test via T/F.

### Five-item assumption list (vs ISLP's six-item problem list)

L05's positive assumption list:

1. Normally distributed $\varepsilon_i$.
2. Mean zero $\mathsf{E}[\varepsilon_i] = 0$.
3. Common variance $\mathrm{Var}(\varepsilon_i) = \sigma^2$.
4. Independent of any other variable.
5. Independent of each other.

ISLP §3.3.3 gives the **violations**:

1. Non-linearity.
2. Correlation of error terms.
3. Non-constant variance.
4. Outliers.
5. High-leverage points.
6. Collinearity.

These are inverse views. Benjamin's framing is "what you assumed"; ISLP's is "what can go wrong." Mapping: (1)→(non-Gaussian residuals shown on QQ), (3)→(non-constant variance), (4)+(5)→(error correlation, e.g. time series tracking).
