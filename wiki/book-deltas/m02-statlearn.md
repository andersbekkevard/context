---
module: 02-statlearn
isl-ch: 2
---

# Module 02 — Book delta

ISLP chapter 2 sets up vocabulary (supervised vs unsupervised, regression vs classification, prediction vs inference, parametric vs nonparametric), states the reducible/irreducible split (eq. 2.3), states the bias-variance decomposition (eq. 2.7), and introduces the Bayes classifier and KNN. What the chapter *does not* contain — and what Benjamin built on the board across L02–L04 — is the **derivation** of the bias-variance decomposition, the **full random-vector / multivariate-normal machinery** that the rest of the course rides on, and the **over-parameterized / double-descent digression** with the minimum-norm interpolator. ISLP states the bias-variance result as a fact: *"Though the mathematical proof is beyond the scope of this book, it is possible to show that …"* (§2.2.2, eq. 2.7). The prof's derivation is the lookup-able piece this file reproduces.

The module-2 atoms also pull in the multivariate-normal density, the covariance-of-linear-transformation identity $\mathrm{Cov}(C\mathbf X) = C\boldsymbol\Sigma C^\top$, the correlation-from-covariance formula, contrasts, and the pseudo-inverse / minimum-norm story for double descent. None of these appear in ISLP ch. 2 (some appear later — MVN density in §4.4.2, double descent in §10.8 — but those are **outside the mapped chapter**, so they are delta for this module).

---

## 1. Bias-variance decomposition — the full derivation

[[[L03-statlearn-2|L03]], [[bias-variance-tradeoff]], [[reducible-vs-irreducible-error]]; ISLP eq. 2.7 states the result, not the proof]

The prof flagged this derivation as **exam-likely** ("I most likely will put an exam question about bias variance"; "there will be a question on the bias-variance decomposition") and reiterated the guarantee in L13, L26, and L27. CE1 problem 1b is exactly this derivation. ISLP §2.2.2 states the conclusion (eq. 2.7) and explicitly skips the proof; reproducing it here.

### 1.1 Setup

Let $(x_0, y_0)$ be an unseen test point with

$$y_0 = f(x_0) + \varepsilon, \qquad \mathbb{E}[\varepsilon] = 0, \qquad \mathrm{Var}(\varepsilon) = \sigma^2, \qquad \varepsilon \perp x_0.$$

Let $\hat f$ be a model fit on a random training set $\mathcal D$. The expectation $\mathbb{E}[\cdot]$ below is taken **jointly over the training set $\mathcal D$ and the noise $\varepsilon$ in $y_0$**.

### 1.2 Step 1 — Reducible / irreducible split

Substitute $y_0 = f(x_0) + \varepsilon$ inside the squared prediction error:

$$
\mathbb{E}\!\left[(y_0 - \hat f(x_0))^2\right]
= \mathbb{E}\!\left[\bigl(f(x_0) + \varepsilon - \hat f(x_0)\bigr)^2\right].
$$

Group as $(A + \varepsilon)^2$ with $A = f(x_0) - \hat f(x_0)$ and expand:

$$
= \mathbb{E}\!\left[A^2\right] + 2\,\mathbb{E}\!\left[A\,\varepsilon\right] + \mathbb{E}\!\left[\varepsilon^2\right].
$$

**Cross term vanishes.** $A = f(x_0) - \hat f(x_0)$ is a function of $x_0$ and $\mathcal D$ only; $\varepsilon$ is independent of both and has mean zero. So $\mathbb{E}[A\varepsilon] = \mathbb{E}[A]\,\mathbb{E}[\varepsilon] = 0$. *Note (prof, L03):* the cross term vanishes because **$\mathbb{E}[\varepsilon] = 0$**, not merely because of "noise/fit independence". Don't conflate the two.

**The $\varepsilon^2$ term.** $\mathbb{E}[\varepsilon^2] = \mathrm{Var}(\varepsilon) + (\mathbb{E}[\varepsilon])^2 = \sigma^2 + 0 = \sigma^2$.

Result:

$$
\boxed{\;\mathbb{E}\!\left[(y_0 - \hat f(x_0))^2\right]
= \underbrace{\mathbb{E}\!\left[(f(x_0) - \hat f(x_0))^2\right]}_{\text{reducible}}
+ \underbrace{\sigma^2}_{\text{irreducible}}\;}
$$

This is the L03 board step that gives ISLP eq. 2.3 as the first algebraic move.

### 1.3 Step 2 — Decompose the reducible part with an add-and-subtract trick

Insert $\pm\,\mathbb{E}[\hat f(x_0)]$ inside the squared bracket:

$$
f(x_0) - \hat f(x_0)
= \bigl(f(x_0) - \mathbb{E}[\hat f(x_0)]\bigr) + \bigl(\mathbb{E}[\hat f(x_0)] - \hat f(x_0)\bigr).
$$

Call these two pieces $B$ (deterministic — depends only on $x_0$) and $V$ (random — depends on $\mathcal D$). Square the sum:

$$
(f(x_0) - \hat f(x_0))^2 = B^2 + 2BV + V^2.
$$

Take expectation over $\mathcal D$:

- $\mathbb{E}[B^2] = B^2 = \bigl(f(x_0) - \mathbb{E}[\hat f(x_0)]\bigr)^2$ — deterministic, comes out of the expectation. **This is $\mathrm{Bias}^2(\hat f(x_0))$.**
- $\mathbb{E}[2BV] = 2B \cdot \mathbb{E}[V] = 2B \cdot \mathbb{E}\!\left[\mathbb{E}[\hat f(x_0)] - \hat f(x_0)\right] = 2B \cdot \bigl(\mathbb{E}[\hat f(x_0)] - \mathbb{E}[\hat f(x_0)]\bigr) = 0$. **Cross term vanishes by the law of iterated expectations** ($\mathbb{E}[\mathbb{E}[\hat f]] = \mathbb{E}[\hat f]$).
- $\mathbb{E}[V^2] = \mathbb{E}\!\left[(\hat f(x_0) - \mathbb{E}[\hat f(x_0)])^2\right] = \mathrm{Var}(\hat f(x_0))$ — the definition of variance.

Substitute back:

$$
\mathbb{E}\!\left[(f(x_0) - \hat f(x_0))^2\right]
= \bigl(f(x_0) - \mathbb{E}[\hat f(x_0)]\bigr)^2 + \mathrm{Var}(\hat f(x_0))
= \mathrm{Bias}^2(\hat f(x_0)) + \mathrm{Var}(\hat f(x_0)).
$$

### 1.4 Putting it together

Combining steps 1 and 2:

$$
\boxed{\;\mathbb{E}\!\left[(y_0 - \hat f(x_0))^2\right]
= \underbrace{\sigma^2}_{\text{irreducible}}
+ \underbrace{\bigl(f(x_0) - \mathbb{E}[\hat f(x_0)]\bigr)^2}_{\mathrm{Bias}^2(\hat f(x_0))}
+ \underbrace{\mathrm{Var}(\hat f(x_0))}_{\text{variance}}\;}
$$

This is ISLP eq. 2.7, derived. Two cross-term cancellations: (i) $\mathbb{E}[\varepsilon] = 0$ kills the noise/fit cross term, (ii) $\mathbb{E}[\hat f] - \mathbb{E}[\hat f] = 0$ kills the bias/variance cross term.

### 1.5 What each term means (prof's framing)

- **Irreducible $\sigma^2$** — pure noise; cannot be reduced by any choice of $\hat f$. Only better data (lower-noise sensors, measuring the missing covariates) lowers it.
- **Squared bias $(f(x_0) - \mathbb{E}[\hat f(x_0)])^2$** — the gap between the truth and the *expected* fit across resampled training sets. Captures both "wrong model class" and "wrong sample" errors (L04 explicit). Bias is **truth-relative**, not fit-relative.
- **Variance $\mathrm{Var}(\hat f(x_0))$** — how much the prediction wobbles across resamples of the training data. Variance is taken over the training distribution, **not** over $\varepsilon$. These two are separate sources.

### 1.6 Why the prof rejects the word "trade-off"

The decomposition is **mathematically exact** at every $x_0$ and every model. "Trade-off" implies that reducing one term forces the other up. That isn't always true:

1. **Better model class flattens the variance curve at no bias cost.** A regularized flexible class ([[ridge-regression|ridge]] / [[lasso]] / a polynomial-degree-20 fit with shrinkage on the high-order coefficients) can hold bias low while suppressing variance — both terms drop together.
2. **Double descent / benign overfitting (see §6)** — past the interpolation threshold, bias and variance can both shrink as $p$ grows, the U-shape isn't a law of nature.
3. **Local geometry of the U-shape.** Bias is **squared** in the decomposition, so a small absolute increase in bias contributes very little to MSE; variance can drop by orders of magnitude in exchange. This is the operating principle of every regularizer in the course (ridge, lasso, smoothing splines, dropout, mini-batch SGD's implicit L2, bagging, pruning).

Verbatim, L03: *"What you can do is you can change how your model behaves and how much variance it has to give up in order to reduce the bias … ultimately, the goal is to minimize both of these terms. And one way to do that is actually change what model you're fitting to your data."* L13: *"if you increase the bias a little bit, you can reduce the variance a lot. Because you have the squared term there."*

### 1.7 Classification analogue

The same decomposition idea applies in classification, with the **Bayes error rate** $1 - \mathbb{E}[\max_j P(Y = j \mid X)]$ playing the role of $\sigma^2$. The Bayes classifier $\hat y(x_0) = \arg\max_j P(Y = j \mid X = x_0)$ is the optimal classifier; its error rate is the floor any classifier must respect. ISLP states this (eq. 2.11) but does not derive a classification analogue of the bias-variance split.

---

## 2. Random vectors — expectation and covariance algebra

[[[L04-statlearn-3|L04]], [[random-vector-and-covariance]]; slides `modules/2StatLearn/2StatLearn.2.md`; NOT in ISLP ch. 2]

ISLP ch. 2 has no formal treatment of random vectors, covariance matrices, or expectation rules for matrix products. The prof developed all of this on the board in L04 because modules 3 (sampling distribution of $\hat\beta$) and 4 (LDA/QDA, Naive Bayes) require it.

### 2.1 Random vector and mean vector

A **random vector** $\mathbf X_{(p \times 1)}$ is a $p$-dimensional vector of random variables:

$$
\mathbf X = \begin{pmatrix} X_1 \\ X_2 \\ \vdots \\ X_p \end{pmatrix},
\qquad
\boldsymbol\mu = \mathbb{E}(\mathbf X) = \begin{pmatrix} \mathbb{E}(X_1) \\ \mathbb{E}(X_2) \\ \vdots \\ \mathbb{E}(X_p) \end{pmatrix}.
$$

$\mathbb{E}(X_j)$ is computed from the marginal of $X_j$ and carries no information about dependencies with $X_k$ for $k \ne j$.

The **joint distribution** $f(\mathbf x)$ governs the whole vector. The marginal of one coordinate is obtained by integrating out the others:

$$
f_1(x_1) = \int_{-\infty}^{\infty} \cdots \int_{-\infty}^{\infty} f(x_1, x_2, \ldots, x_p) \, dx_2 \cdots dx_p.
$$

### 2.2 Rule I — expectation is linear and matrix-additive

For random matrices $\mathbf X_{(n \times p)}$ and $\mathbf Y_{(n \times p)}$:

$$
\mathbb{E}(\mathbf X + \mathbf Y) = \mathbb{E}(\mathbf X) + \mathbb{E}(\mathbf Y).
$$

### 2.3 Rule II — constants pull out of expectation (with a board proof)

For random matrix $\mathbf X_{(n \times p)}$ and conformable **constant** matrices $A$ and $B$:

$$
\boxed{\;\mathbb{E}(A \mathbf X B) = A \cdot \mathbb{E}(\mathbf X) \cdot B\;}
$$

**Element-wise proof (L04 on the board).** The $(i, j)$ element of $A \mathbf X B$ is

$$
e_{ij} = \sum_{k=1}^{n} \sum_{l=1}^{p} a_{ik}\, X_{kl}\, b_{lj}.
$$

The $a_{ik}$ and $b_{lj}$ are constants; only $X_{kl}$ is random. So

$$
\mathbb{E}(e_{ij}) = \sum_{k} \sum_{l} a_{ik}\, \mathbb{E}(X_{kl})\, b_{lj},
$$

which is exactly the $(i, j)$ element of $A \cdot \mathbb{E}(\mathbf X) \cdot B$. ∎

**Univariate analogue:** $\mathbb{E}(aX + b) = a\,\mathbb{E}(X) + b$.

### 2.4 Covariance matrix

For each pair $(i, j)$:

$$
\sigma_{ij} = \mathrm{Cov}(X_i, X_j) = \mathbb{E}\bigl[(X_i - \mu_i)(X_j - \mu_j)\bigr] = \mathbb{E}(X_i X_j) - \mu_i \mu_j.
$$

When $i = j$: $\sigma_{ii} = \sigma_i^2 = \mathrm{Var}(X_i)$ (the prof flagged this as a quiz-style fact: "what is $\mathrm{Cov}(X_i, X_i)$?" → variance).

Sign reading: high positive covariance → variables vary together; negative → they vary opposite; near zero → no **linear** co-variation (this is the prof's most-emphasized framing).

Stack into the **covariance matrix**:

$$
\boldsymbol\Sigma = \mathrm{Cov}(\mathbf X) = \mathbb{E}\bigl[(\mathbf X - \boldsymbol\mu)(\mathbf X - \boldsymbol\mu)^\top\bigr]
= \begin{bmatrix}
\sigma_{1}^2 & \sigma_{12} & \cdots & \sigma_{1p} \\
\sigma_{12} & \sigma_{2}^2 & \cdots & \sigma_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
\sigma_{1p} & \sigma_{2p} & \cdots & \sigma_{p}^2
\end{bmatrix}.
$$

**Shortcut identity** (slides + L04):

$$
\boxed{\;\boldsymbol\Sigma = \mathbb{E}(\mathbf X \mathbf X^\top) - \boldsymbol\mu \boldsymbol\mu^\top\;}
$$

$\boldsymbol\Sigma$ is **symmetric** (by construction) and **positive semi-definite**: for any constant vector $\mathbf b \ne \mathbf 0$,

$$
\mathbf b^\top \boldsymbol\Sigma \mathbf b = \mathrm{Var}(\mathbf b^\top \mathbf X) \ge 0.
$$

A variance can never be negative, so $\boldsymbol\Sigma$ is PSD by the variance interpretation. If $|\boldsymbol\Sigma| = 0$ then $\boldsymbol\Sigma$ is **singular**, which means some linear combination $\mathbf b^\top \mathbf X$ has zero variance, i.e. is deterministic given the others. In that case the multivariate normal density is not defined (division by $|\boldsymbol\Sigma|^{1/2} = 0$).

> **Key conceptual point** (L04, verbatim flavor): *"Covariance is really getting at this notion of a slope … we're sort of assuming a linear line."* Zero covariance does **not** imply independence in general — it only means no linear co-variation. The exception is **joint normality**, where zero covariance does imply independence (see §3).

### 2.5 Correlation matrix

Rescale $\boldsymbol\Sigma$ by the standard-deviation diagonal:

$$
\rho_{ij} = \frac{\sigma_{ij}}{\sqrt{\sigma_i^2 \, \sigma_j^2}} = \frac{\sigma_{ij}}{\sigma_i \sigma_j} \in [-1, 1].
$$

In matrix form, with $\mathbf V^{1/2} = \mathrm{diag}(\sigma_1, \ldots, \sigma_p)$ the diagonal matrix of standard deviations:

$$
\boxed{\;\boldsymbol\rho = (\mathbf V^{1/2})^{-1} \, \boldsymbol\Sigma \, (\mathbf V^{1/2})^{-1}\;}
$$

The correlation matrix has all 1's on the diagonal and Pearson correlations off-diagonal. **The hand-calculation trap** (CE1 1f): given $\boldsymbol\Sigma = \begin{pmatrix} 9 & 0.3 \\ 0.3 & 4 \end{pmatrix}$, then $\rho_{12} = 0.3 / \sqrt{9 \cdot 4} = 0.3 / 6 = 0.05$. Distractors: $0.3 / 36 = 0.0083$ (forgot the square root), $0.3/2 = 0.15$ (used only $\sqrt{4}$), $0.3/3 = 0.10$ (used only $\sqrt{9}$). **Take the square root of the product of variances.**

### 2.6 Linear-transformation identities

Let $\mathbf Z = C \mathbf X$ where $C$ is a constant $(k \times p)$ matrix. Then:

$$
\boxed{\;\mathbb{E}(\mathbf Z) = C \boldsymbol\mu, \qquad \mathrm{Cov}(\mathbf Z) = C \boldsymbol\Sigma C^\top\;}
$$

These two identities are the working engine of the rest of the course. They give:

- The OLS sampling distribution $\hat\beta = (X^\top X)^{-1} X^\top \mathbf y$ has covariance $\sigma^2 (X^\top X)^{-1}$ (module 3).
- The LDA discriminant linearity and the QDA quadratic term (module 4).
- The variance of any contrast / hypothesis test on $\hat\beta$ (module 3).
- [[principal-component-analysis|PCA]] loadings as linear combinations whose variance you can read off (module 10).

**Pitfall:** order matters. It is $C \boldsymbol\Sigma C^\top$, **not** $C^\top \boldsymbol\Sigma C$. Transpose goes on the right.

---

## 3. Contrasts (linear combinations)

[[[L04-statlearn-3|L04]], [[contrasts]]; NOT in ISLP ch. 2 — ISLP discusses contrasts only in the dummy-coding sense in §3.3.1]

A **contrast** is any linear combination of the components of $\mathbf X$. The prof uses "contrast" loosely (not in the strict statistical sense of "coefficients sum to zero" — e.g. $E + W$ counts as a contrast here).

Given $\mathbf X_{(p \times 1)}$ and a contrast matrix $C_{(k \times p)}$:

$$
\mathbf Z = C \mathbf X = \begin{pmatrix}
\sum_{j=1}^{p} c_{1j} X_j \\
\sum_{j=1}^{p} c_{2j} X_j \\
\vdots \\
\sum_{j=1}^{p} c_{kj} X_j
\end{pmatrix}
$$

Apply §2.6:

$$
\mathbb{E}(\mathbf Z) = C \boldsymbol\mu, \qquad \mathrm{Cov}(\mathbf Z) = C \boldsymbol\Sigma C^\top.
$$

### 3.1 The cork worked example (L04 / slides)

Running dataset: $\mathbf X = (X_N, X_E, X_S, X_W)^\top$ — cork weight in 4 directions on a tree, $n = 28$ trees, Rao (1948). The three contrasts of interest are $N - S$, $E + W$, and $(E + W) - (N + S)$. The contrast matrix is

$$
C = \begin{bmatrix}
1 & 0 & -1 & 0 \\
0 & 1 & 0 & 1 \\
-1 & 1 & -1 & 1
\end{bmatrix}.
$$

Then $\mathbb{E}(Y_1) = \mu_N - \mu_S$, and $\mathrm{Cov}(Y_1, Y_3)$ is the $(1, 3)$ entry of $C \boldsymbol\Sigma C^\top$.

### 3.2 Why this matters downstream

Each item below is exam-relevant; the **covariance-of-linear-transformation identity is the bridge**:

- **Hypothesis tests on regression coefficients** (M3): testing $\beta_1 = \beta_2$ or any combination $\mathbf c^\top \boldsymbol\beta$ uses $\mathrm{Var}(\mathbf c^\top \hat\beta) = \sigma^2 \, \mathbf c^\top (X^\top X)^{-1} \mathbf c$.
- **[[linear-discriminant-analysis|LDA]] / [[quadratic-discriminant-analysis|QDA]] discriminants** (M4): linear in $\mathbf x$ for LDA precisely because $\mathbf x^\top \Sigma^{-1} \mu_k$ is a contrast on $\mathbf x$.
- **[[principal-component-analysis|PCA]] loadings** (M10): each principal component is a contrast on the standardized predictors; the loadings $\phi_{jm}$ are the contrast coefficients.
- **[[categorical-encoding-and-interactions|Categorical encoding]] with $K$ levels** (M3): the $K - 1$ dummy variables define $K - 1$ contrasts against the reference level.

---

## 4. Multivariate normal distribution

[[[L04-statlearn-3|L04]] / [[L05-linreg-1|L05]], [[multivariate-normal]]; NOT in ISLP ch. 2 — ISLP introduces MVN only in §4.4.2 inside the LDA discussion]

The prof introduced the MVN density in L04 as the generalization of the univariate normal, with the explicit motivation that minimizing the negative log-likelihood of a normal model with mean parameterized by $f(\mathbf x)$ **is** linear regression — and the multivariate case is the route into multiple regression. Three downstream uses: joint distribution of a random vector (M2), sampling distribution of $\hat\beta$ (M3), class-conditional density in LDA / QDA / Naive Bayes (M4).

### 4.1 The density

For $\mathbf x \in \mathbb{R}^p$ with mean vector $\boldsymbol\mu$ and **positive-definite** covariance matrix $\boldsymbol\Sigma$:

$$
\boxed{\;
f(\mathbf x) = \frac{1}{(2\pi)^{p/2} \, |\boldsymbol\Sigma|^{1/2}} \,
\exp\!\left(-\tfrac{1}{2} (\mathbf x - \boldsymbol\mu)^\top \boldsymbol\Sigma^{-1} (\mathbf x - \boldsymbol\mu)\right)
\;}
$$

Notation: $\mathbf X \sim N_p(\boldsymbol\mu, \boldsymbol\Sigma)$.

**Mapping from the univariate case** $f(x) = (2\pi\sigma^2)^{-1/2} \exp\bigl(-(x - \mu)^2 / (2\sigma^2)\bigr)$:

| Univariate piece | Multivariate piece |
|---|---|
| $x - \mu$ (scalar) | $\mathbf x - \boldsymbol\mu$ (vector) |
| $(x - \mu)^2 / \sigma^2$ | $(\mathbf x - \boldsymbol\mu)^\top \boldsymbol\Sigma^{-1} (\mathbf x - \boldsymbol\mu)$ (Mahalanobis distance) |
| $\sigma$ in the normalizer | $|\boldsymbol\Sigma|^{1/2}$ |
| $\sqrt{2\pi}$ | $(2\pi)^{p/2}$ |
| $1/\sigma^2$ in the exponent | $\boldsymbol\Sigma^{-1}$ (the **precision matrix**) |

When $p = 1$ the density reduces to the univariate Gaussian.

The exponent $-\tfrac{1}{2} D_M^2$ where $D_M^2 = (\mathbf x - \boldsymbol\mu)^\top \boldsymbol\Sigma^{-1} (\mathbf x - \boldsymbol\mu)$ is the squared **Mahalanobis distance** — Euclidean distance after rotating and rescaling by $\boldsymbol\Sigma^{-1/2}$. The natural distance under joint normality.

### 4.2 Singular $\boldsymbol\Sigma$

If $|\boldsymbol\Sigma| = 0$, the density is **not defined** (you would divide by zero). This is the same pathology as [[collinearity]] in OLS ($X^\top X$ singular) and as the failure mode of LDA when class-covariance estimates are singular. Practical fix: drop a redundant variable, regularize, or reduce dimensions via PCA.

### 4.3 Four useful properties (slides + L04)

Let $\mathbf X \sim N_p(\boldsymbol\mu, \boldsymbol\Sigma)$.

1. **Contours are ellipsoids.** Level sets $\{\mathbf x : (\mathbf x - \boldsymbol\mu)^\top \boldsymbol\Sigma^{-1} (\mathbf x - \boldsymbol\mu) = b\}$ are ellipsoids centered at $\boldsymbol\mu$, oriented and stretched by $\boldsymbol\Sigma$.
2. **Linear combinations are normal.** $A \mathbf X + \mathbf b \sim N_k(A\boldsymbol\mu + \mathbf b, A \boldsymbol\Sigma A^\top)$ for any constant matrix $A$ and vector $\mathbf b$. (Bridge to contrasts.)
3. **Marginals are normal.** Any subset of components is multivariate normal in its own right.
4. **Zero covariance implies independence — under joint normality only.** $\mathrm{Cov}(X_i, X_j) = 0 \Rightarrow X_i \perp X_j$ when $\mathbf X$ is jointly normal. **Crucially false in general** for non-Gaussian distributions.

The prof emphasizes property 4 as the headline reason MVN is exam-bait — it's the one place where the zero-cov-→-independence shortcut is valid.

### 4.4 Quantile / probability statement from the ellipsoid

$(\mathbf X - \boldsymbol\mu)^\top \boldsymbol\Sigma^{-1} (\mathbf X - \boldsymbol\mu) \sim \chi^2_p$. So the ellipsoid

$$
\{\mathbf x : (\mathbf x - \boldsymbol\mu)^\top \boldsymbol\Sigma^{-1} (\mathbf x - \boldsymbol\mu) \le \chi^2_p(\alpha)\}
$$

has probability $1 - \alpha$ — useful for confidence ellipsoids and the LDA contour pictures in M4.

### 4.5 Constructing independent standard normals from $\mathbf X$

Slide quiz answer (5C):

$$
\boldsymbol\Sigma^{-1/2} (\mathbf X - \boldsymbol\mu) \sim N_p(\mathbf 0, I_p).
$$

This is the **whitening transform** — subtract the mean, rescale by the inverse-square-root of the covariance, get back to standard MVN.

### 4.6 Reading 2D contour plots — Σ-to-shape map

For $\boldsymbol\Sigma = \begin{pmatrix} \sigma_x^2 & \rho \sigma_x \sigma_y \\ \rho \sigma_x \sigma_y & \sigma_y^2 \end{pmatrix}$:

| Σ pattern | Contour appearance |
|---|---|
| $\sigma_x^2 = \sigma_y^2$, $\rho = 0$ | Circle, centered at $\boldsymbol\mu$ |
| $\sigma_x^2 \ne \sigma_y^2$, $\rho = 0$ | Axis-aligned ellipse, long axis along the larger-variance direction |
| $\rho > 0$ | Ellipse tilted upward-right (positive diagonal) |
| $\rho < 0$ | Ellipse tilted upward-left (negative diagonal) |
| $\sigma_x^2 \ne \sigma_y^2$, $\rho \ne 0$ | Tilted ellipse, long axis closer to whichever variance is larger |

This is the CE1 1g question template.

### 4.7 The connection to regression (L04 verbatim setup)

Why this density is module 2's destination: minimizing the negative log-likelihood of $\mathbf y \mid \mathbf x$ under a Gaussian model with $\mu_{Y \mid X} = \mathbf x^\top \boldsymbol\beta$ **is** least-squares regression. The multivariate normal is the joint distribution view that gives the conditional view of regression in module 3.

If $(\mathbf X, Y)$ is jointly MVN, then the conditional $Y \mid \mathbf X = \mathbf x$ is normal with **mean linear in $\mathbf x$** and **constant variance** — the [[linear-regression|linear regression model]] is the exact conditional structure of a joint MVN. This is the bridge into M3.

---

## 5. KNN regression — the averaging formula

[[[L03-statlearn-2|L03]] (introduced) / [[L10-resample-1|L10]] (used), [[knn-regression]]; ISLP ch. 2 develops KNN for classification only; KNN regression's formula appears in ISLP §3.5]

ISLP §2.2.3 introduces KNN for **classification** as the running example. The **regression** variant is in §3.5, not §2 — so the formula is delta for the mapped chapter:

$$
\boxed{\;\hat f(x_0) = \frac{1}{K} \sum_{i \in \mathcal N_0} y_i\;}
$$

where $\mathcal N_0$ is the index set of the $K$ training points closest to $x_0$ in Euclidean distance.

Same flexibility-knob story as classification: small $K$ → wiggly fit, low bias, high variance; $K = n$ → constant horizontal line at $\bar y$ (pure underfit). Optimal $K$ is intermediate, chosen by [[cross-validation]].

---

## 6. Over-parameterized regime — pseudoinverse, double descent, benign overfitting

[[[L04-statlearn-3|L04]], [[double-descent]]; NOT in ISLP ch. 2 — ISLP discusses double descent only in §10.8 (deep-learning chapter)]

The L04 digression Benjamin ran with his own simulations because the textbook example only goes up to degree 8 or 9 and he wanted to enter "this ridiculous region of like a degree 50,000 or 100,000." None of this is in ISLP ch. 2.

### 6.1 The setup and the phenomenon

Truth: a **step function** (deliberately a poor fit for any polynomial). Sample $n = 100$ noisy points; fit polynomials of degrees $d = 1, 2, \ldots, 100{,}000$ by minimum-norm least squares (pseudoinverse). The training MSE vs. degree curve has **three regimes**:

1. **Classical regime** ($d \ll n$): test MSE U-shape, minimum somewhere around $d \sim 24$.
2. **Interpolation peak** at $d = n$: test MSE **explodes**. Same number of parameters as data points; noise blows up.
3. **Second descent** ($d \gg n$): test MSE drops **again**, often below the classical minimum.

This **double-descent** curve is the prof's hobbyhorse and returns in L11, L13, L24, L26. The decomposition still adds up: bias² + variance + $\sigma^2 = $ test MSE at every $d$. *"It doesn't break any of the math. It doesn't break any of the statistics."*

### 6.2 The optimization changes character across the interpolation point

**Classical regime** ($p < n$, regularized version): minimize a fit+penalty objective

$$
L_{\text{cls}}(\boldsymbol\beta) = \sum_i \bigl(y_i - \mathbf x_i^\top \boldsymbol\beta\bigr)^2 + \lambda \sum_j \beta_j^2.
$$

The fit term is non-zero at the optimum.

**Over-parameterized regime** ($p \gg n$, post-interpolation): minimize the L2 norm subject to **exactly** fitting every training point

$$
\boxed{\;
\min_{\boldsymbol\beta} \sum_j \beta_j^2
\quad \text{subject to} \quad
y_i = \mathbf x_i^\top \boldsymbol\beta \text{ for all } i = 1, \ldots, n
\;}
$$

The data-fit term has become a **hard constraint**; the L2 penalty is the only objective. This is the **minimum-norm interpolator** — among the infinitely many $\boldsymbol\beta$'s that achieve zero training error, the one with the smallest $\|\boldsymbol\beta\|_2$. The Moore–Penrose pseudoinverse and mini-batch SGD both converge to this solution.

### 6.3 The minimum-norm interpolator via the pseudoinverse

When $p > n$ and $X$ has full row rank, the unique solution to the constrained problem above is

$$
\hat\boldsymbol\beta = X^+ \mathbf y = X^\top (X X^\top)^{-1} \mathbf y.
$$

Here $X^+ = X^\top (X X^\top)^{-1}$ is the **Moore–Penrose pseudoinverse** of the $n \times p$ design matrix in the wide case ($p > n$). (In the classical $p < n$ case the pseudoinverse coincides with $(X^\top X)^{-1} X^\top$.) The pseudoinverse mechanics themselves are out of scope per [[docs/scope]]; the only fact the prof needs you to know is that it **selects the minimum-norm zero-training-error solution**.

### 6.4 Why the second descent happens — the prof's explanation

Past the interpolation point, the model class contains **infinitely many** zero-training-error solutions; the training loss can't distinguish them. The optimization picks via a **secondary criterion** — implicit L2 norm minimization — via the pseudoinverse or via mini-batch SGD. The minimum-norm solution has **low variance** because small coefficients mean small wobble under resampling. So past the interpolation point, you're effectively running [[ridge-regression|ridge regression]] with an implicit $\lambda$ chosen by the geometry of the optimization.

This is the **"benign"** in *benign overfitting*: zero training error and good generalization simultaneously, because the implicit regularization controls variance.

### 6.5 When over-parameterization wins (L04 explicit)

The over-parameterized win requires **the truth not to be in the assumed function class**. Prof's two illustrations:

- True $f = $ step function (NOT a polynomial), fit polynomials → high-$d$ regime beats the low-$d$ minimum.
- True $f = x^2$ (IS a polynomial), fit polynomials → low-$d$ poly2 recovers truth perfectly; high-$d$ regime cannot improve on it.

> *"If the underlying model … exists as part of the functions that you're assuming in your model, then even if you increasingly add more and more degrees of flexibility, you're not going to improve over what you can get with a few parameters … But in the real world, I don't know how often you really can assume that you have the right model."* — L04

### 6.6 What it implies for the bias-variance picture

The decomposition still holds exactly:

$$
\mathbb{E}\!\left[(y_0 - \hat f(x_0))^2\right] = \sigma^2 + \mathrm{Bias}^2(\hat f(x_0)) + \mathrm{Var}(\hat f(x_0)).
$$

What changes profile:

- Variance shoots up at $p \approx n$ (ill-conditioned).
- Past $p = n$, variance drops back down because the implicit norm-minimization is a variance-control device.
- Bias² stays low (or grows slowly, visible only on a log scale per L04).
- Sum traces the double-descent shape.

This is **the canonical answer** to "why is the prof critical of the word 'trade-off'?": (i) regularization can flatten the variance curve without paying bias cost, (ii) double descent shows you can have low bias **and** low variance **and** zero training error simultaneously.

### 6.7 Slide-deck caveats

- Double descent is achievable mostly in **high signal-to-noise** problems (image classification, language modelling).
- **Most statistical learning methods covered in this course do not exhibit double descent** — trees, GAMs, explicit-regularization regression don't go past the interpolation point in the relevant sense.
- "Though double descent can sometimes occur in neural networks, we typically do not want to rely on this behavior" — slide, qualified by the prof: *"depends on what you're trying to do."*

---

## Notation and naming differences

These are points where the prof's wording or framing differs meaningfully from ISLP ch. 2. None of them is a new artifact — they are just naming conventions to be aware of.

- **"Decomposition" vs "trade-off."** Benjamin calls eq. 2.7 the **bias-variance decomposition** and is explicit that he dislikes "trade-off" because it implies a forced exchange. ISLP §2.2.2 uses "trade-off" throughout. Treat them as the same equation; the framing critique is the prof's contribution.
- **"Fit data" for "training data."** Benjamin uses these interchangeably; ISLP uses only "training data."
- **Data-matrix convention.** Benjamin's lecture board uses **columns = individuals/samples, rows = variables**. He flags that ISLP uses the opposite convention (rows = observations, columns = variables) and says it doesn't matter for the math as long as you're consistent.
- **"Independent variables" — avoided.** Benjamin specifically warns against the term "independent variables" for predictors (they're rarely independent of each other). Preferred terms: predictors, regressors, covariates, features, variables. ISLP uses "predictors / independent variables" interchangeably (§2.1).
- **$K$ in KNN vs. $K$ in classification.** The $K$ in $K$-nearest neighbors (number of neighbors) is **not** the same as the $K$ in "$K$ classes" (number of categories). Both ISLP and the prof use $K$ for both; Benjamin flags the collision explicitly in L07.
- **"Contrast" — loose vs strict.** Benjamin uses "contrast" for **any** linear combination of components (so $E + W$ counts). The strict statistical definition requires the coefficients to sum to zero. ISLP ch. 2 doesn't use the term; ISLP §3.3.1 uses it in the strict dummy-coding sense.
- **Bias term in linear regression.** Benjamin calls $\beta_0$ the "bias term / intercept" — note that "bias" here is the ML usage (the constant offset of a linear unit) and is **unrelated** to the statistical bias of an estimator that appears in the bias-variance decomposition. Same word, two meanings.
- **"Sigma" overload.** $\sigma^2$ is the noise variance $\mathrm{Var}(\varepsilon)$ in the regression model **and** the variance of a single component $\mathrm{Var}(X_j) = \sigma_j^2$ in the random-vector context. Context disambiguates. $\boldsymbol\Sigma$ (capital) is always the $p \times p$ covariance matrix.
