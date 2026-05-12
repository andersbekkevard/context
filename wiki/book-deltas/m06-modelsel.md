---
title: "M06: Model Selection and Regularization — Book delta"
module: 06-modelsel
isl-ch: 6
lectures: [L12, L13, L14, L15]
---

# Module 06: Model Selection and Regularization — Book delta

ISLP chapter 6 covers the same skeleton the prof teaches ([[subset-selection|best subset / forward / backward]], [[ridge-regression|ridge]], [[lasso]], [[principal-component-regression|PCR]], [[partial-least-squares|PLS]], [[high-dimensional-regression|high-dim]]), but a handful of **concrete, lookup-able artifacts** that Benjamin used at the board or that he made central to his framing are **absent from chapter 6**. The book's ridge derivation is verbal — it does not give the matrix closed form $(X^\top X + \lambda I)^{-1} X^\top y$; the [[principal-component-analysis|PCA]]-as-eigendecomposition machinery ISLP defers to ch. 12; and the per-PC ridge-shrinkage factor $\lambda_j^2/(\lambda_j^2+\lambda)$ that the prof read off the slide is footnoted to ESL §3.5 rather than reproduced. This file collects everything in that gap, plus the prof's matrix-form / closed-form / variance machinery that the book leaves implicit.

Anything *only* in the book and not reached by lectures/slides/exercises is out per [[docs/scope]] and is not in this file. Anything covered by ISLP ch. 6 in clean form (e.g. eqs 6.5, 6.7, 6.14, 6.15, [[ridge-vs-lasso-geometry|ridge/lasso constraint geometry]], the [[bias-variance-tradeoff|bias-variance]] figure, soft-thresholding for orthonormal $X$) is **not** reproduced here — go look it up.

---

## 1. Ridge regression: matrix closed form and what ISLP doesn't give

[[[L12-modelsel-1|L12]], [[L13-modelsel-2|L13]], concept: [[ridge-regression]]; ISLP §6.2.1]

ISLP §6.2.1 presents ridge as the minimizer of (6.5),

$$
\sum_{i=1}^n \left(y_i - \beta_0 - \sum_{j=1}^p \beta_j x_{ij}\right)^2 + \lambda \sum_{j=1}^p \beta_j^2,
$$

then immediately discusses the [[bias-variance-tradeoff|bias-variance trade-off]] (Fig 6.5) and [[standardization]] (eq 6.6). **It does not give a matrix-form solution.** The prof gestured at it but didn't write it on the board ("the math is just as easy as before: take derivatives, get a closed form" — [[L12-modelsel-1|L12]], [[L13-modelsel-2|L13]]). Here it is in full, since this is the artifact most-cited in the wiki and not lookup-able in ch. 6.

### 1.1 Setup (centered/standardized form)

Assume $X \in \mathbb{R}^{n \times p}$ has been **centered** (column means subtracted) and **standardized** (eq 6.6: divide each column by its sample standard deviation $\sqrt{\frac{1}{n}\sum_i(x_{ij} - \bar x_j)^2}$). Then we can drop the intercept from the penalty: $\hat\beta_0 = \bar y$ regardless of $\lambda$ (ISLP states this in the paragraph after eq 6.5; the centered form is its consequence).

In matrix form, the ridge objective is

$$
L(\beta) = (y - X\beta)^\top (y - X\beta) + \lambda \beta^\top \beta.
$$

### 1.2 Derivation

Differentiate w.r.t. $\beta$ and set to zero:

$$
\frac{\partial L}{\partial \beta} = -2 X^\top (y - X\beta) + 2 \lambda \beta = 0
\quad\Longrightarrow\quad
(X^\top X + \lambda I_p)\,\hat\beta^R_\lambda = X^\top y.
$$

Hence

$$
\boxed{\;\hat\beta^R_\lambda = (X^\top X + \lambda I_p)^{-1} X^\top y.\;}
$$

The intercept is recovered as $\hat\beta_0 = \bar y - \bar x^\top \hat\beta^R_\lambda$ (or simply $\bar y$ if $X$ was centered).

### 1.3 Why this matters (and why ISLP avoids it)

- **Unique solution at $p \geq n$**: $X^\top X$ is singular when $p \geq n$, but $X^\top X + \lambda I_p$ has eigenvalues bounded below by $\lambda > 0$ and is invertible for any $\lambda > 0$. This is exactly the prof's "ridge stays unique" claim from [[L13-modelsel-2|L13]]: *"If you have too many parameters, you can add a regularizer to keep the model finding a unique solution. It maintains the model as convex, meaning it will still have a unique solution for a given value of $\lambda$."* ISLP §6.2.1 states this verbally (last paragraph p. 256 in the printed edition) but never writes the matrix that makes it manifest.

- **Recovering OLS**: $\lambda = 0 \Rightarrow \hat\beta^R = (X^\top X)^{-1} X^\top y = \hat\beta^{OLS}$ when $X^\top X$ is non-singular.

- **Recovering the null model**: $\lambda \to \infty \Rightarrow \hat\beta^R_\lambda \to 0$.

### 1.4 Sampling distribution under the linear model

Assuming the standard linear-model setup $y = X\beta + \varepsilon$ with $\varepsilon \sim \mathcal{N}_n(0, \sigma^2 I_n)$:

Let $A_\lambda := (X^\top X + \lambda I_p)^{-1} X^\top$, so $\hat\beta^R_\lambda = A_\lambda y$.

- **Expectation (bias).**
  $$
  \mathbb{E}[\hat\beta^R_\lambda] = A_\lambda X \beta = (X^\top X + \lambda I_p)^{-1} X^\top X \beta.
  $$
  This is **not** $\beta$ unless $\lambda = 0$, so ridge is **biased** for $\lambda > 0$. The bias is
  $$
  \text{Bias}(\hat\beta^R_\lambda) = \mathbb{E}[\hat\beta^R_\lambda] - \beta = -\lambda (X^\top X + \lambda I_p)^{-1} \beta,
  $$
  i.e. ridge shrinks toward zero in the metric defined by $(X^\top X + \lambda I_p)^{-1}$.

- **Variance.**
  $$
  \mathrm{Var}(\hat\beta^R_\lambda) = A_\lambda \cdot \sigma^2 I_n \cdot A_\lambda^\top = \sigma^2 (X^\top X + \lambda I_p)^{-1} X^\top X (X^\top X + \lambda I_p)^{-1}.
  $$
  In the limit $\lambda \to 0$ this reduces to the OLS variance $\sigma^2 (X^\top X)^{-1}$; for $\lambda > 0$ the variance is **strictly smaller** in the Loewner order. This is the algebraic instance of the prof's *"variance can shrink to about half of the OLS variance for the cost of a small bias bump"* claim from [[L13-modelsel-2|L13]].

- **Distribution.** Linear combination of Gaussians ⇒
  $$
  \hat\beta^R_\lambda \sim \mathcal{N}_p\!\left(\,(X^\top X + \lambda I_p)^{-1} X^\top X \beta,\;\; \sigma^2 (X^\top X + \lambda I_p)^{-1} X^\top X (X^\top X + \lambda I_p)^{-1}\,\right).
  $$

ISLP §6.2.1 plots bias and variance (Fig 6.5) but does **not** give these closed-form expressions. They are useful at the exam table for any "show that ridge is biased" / "write down the ridge variance" question.

---

## 2. The ridge hat matrix and effective degrees of freedom

[[[L13-modelsel-2|L13]], [[L15-modelsel-4|L15]], concepts: [[ridge-regression]], [[principal-component-regression]]]

ISLP §6.4 (Fig 6.24) uses the phrase *"degrees of freedom"* for the lasso (number of non-zero coefficients) but **never defines ridge's effective degrees of freedom**. The slide deck and the prof's "ridge shrinks the small-eigenvalue directions" framing rely on this object, so reproduce it here.

### 2.1 Ridge hat matrix

Predictions are linear in $y$:

$$
\hat y_\lambda = X \hat\beta^R_\lambda = X (X^\top X + \lambda I_p)^{-1} X^\top y =: H_\lambda \, y,
$$

where

$$
\boxed{\;H_\lambda := X (X^\top X + \lambda I_p)^{-1} X^\top.\;}
$$

$H_\lambda$ is symmetric and idempotent only at $\lambda = 0$ (the OLS hat matrix $H_0 = X(X^\top X)^{-1} X^\top$); for $\lambda > 0$ it is symmetric but not idempotent — it is a **smoother**, not a projection.

### 2.2 Effective degrees of freedom

The effective degrees of freedom of a linear smoother are $\mathrm{tr}(H)$. For ridge:

$$
\boxed{\;\mathrm{df}(\lambda) := \mathrm{tr}(H_\lambda) = \mathrm{tr}\bigl((X^\top X + \lambda I_p)^{-1} X^\top X\bigr) = \sum_{j=1}^p \frac{d_j^2}{d_j^2 + \lambda},\;}
$$

where $d_1 \geq d_2 \geq \dots \geq d_p \geq 0$ are the singular values of $X$ (so $d_j^2$ are the eigenvalues of $X^\top X$).

Two limits:
- $\lambda = 0 \Rightarrow \mathrm{df}(0) = p$ (or $\mathrm{rank}(X)$): plain OLS spends one effective parameter per predictor.
- $\lambda \to \infty \Rightarrow \mathrm{df}(\lambda) \to 0$: full shrinkage, zero effective parameters.

So $\lambda$ slides $\mathrm{df}$ continuously from $0$ to $p$, the prof's *"reduce the number of effective parameters"* framing on the slide (and in [[L12-modelsel-1|L12]]: *"it tries to reduce the number of parameters effectively — effective parameters — because if the beta is zero, then essentially that parameter is not in there."*).

---

## 3. Per-principal-component ridge shrinkage factor

[[[L15-modelsel-4|L15]], slide deck "Example: Shrinkage Factor", concept: [[ridge-regression]]]

The slide deck (selection_regularization_presentation_lecture2.md, "PCR can be seen as discretized version of Ridge regression") explicitly gives the per-PC ridge shrinkage factor as

$$
\boxed{\;s_j(\lambda) = \frac{\lambda_j^2}{\lambda_j^2 + \lambda},\;}
$$

where $\lambda_j$ are the **eigenvalues** of the (standardized) $X^\top X$ matrix. Note: the slide uses $\lambda_j$ for eigenvalues *and* $\lambda$ for the ridge tuning parameter, an unfortunate clash. With singular values $d_j$ of $X$, $\lambda_j = d_j^2$, and the slide formula is exactly $d_j^2 / (d_j^2 + \lambda)$ — matching §2.2 above term-by-term.

### 3.1 Why this artifact matters

In the SVD basis $X = U D V^\top$ (where $D = \mathrm{diag}(d_1, \dots, d_p)$), ridge acts on each principal direction independently, with shrinkage strength $s_j(\lambda)$ on the $j$-th PC direction:

- Large $d_j$ (high-variance PC direction) $\Rightarrow s_j(\lambda) \approx 1$ (almost no shrinkage).
- Small $d_j$ (low-variance PC direction) $\Rightarrow s_j(\lambda) \approx 0$ (heavy shrinkage).

This is the **algebraic content** of the prof's *"higher pressure on less important PCs"* slide bullet and his *"PCR can be seen as a discretized version of ridge regression"* [[L15-modelsel-4|L15]]:

- **PCR**: discards the $p - M$ smallest-eigenvalue PC directions outright (hard threshold).
- **Ridge**: shrinks each direction by the continuous factor $\lambda_j^2/(\lambda_j^2 + \lambda)$ — heavier on the smaller-eigenvalue ones (soft threshold).

ISLP §6.3.1 says the connection between ridge and PCR exists ("one can even think of ridge regression as a continuous version of PCR!") and **points the reader to ESL §3.5** in footnote 8 — that is, ISLP ch. 6 deliberately does not derive this. The prof's slide shows the formula directly; this file reproduces it for exam lookup.

### 3.2 SVD form (the cleanest derivation, light)

Write $X = U D V^\top$ with $U \in \mathbb{R}^{n \times p}$, $V \in \mathbb{R}^{p \times p}$ orthogonal, $D = \mathrm{diag}(d_1, \dots, d_p)$. Then

$$
\hat\beta^R_\lambda = (V D^2 V^\top + \lambda I)^{-1} V D U^\top y = V (D^2 + \lambda I)^{-1} D U^\top y,
$$

and the fitted values are

$$
\hat y_\lambda = U D (D^2 + \lambda I)^{-1} D U^\top y = \sum_{j=1}^p u_j \frac{d_j^2}{d_j^2 + \lambda} u_j^\top y,
$$

where $u_j$ is the $j$-th column of $U$. The coefficient on $u_j u_j^\top y$ is exactly $s_j(\lambda) = d_j^2/(d_j^2 + \lambda)$. (Prof flagged spectral decomposition as out of scope per [[L04-statlearn-3|L04]], so this is the lightest form — present for completeness.)

---

## 4. PCA fraction of variance via eigenvalues

[[[L14-modelsel-3|L14]], [[L15-modelsel-4|L15]], slide deck, concept: [[principal-component-regression]]]

The slide deck contains the formula

$$
\boxed{\;R^2_{\text{PCA}} = \frac{\sum_{i=1}^M \lambda_i}{\sum_{j=1}^p \lambda_j},\quad \lambda_i \text{ eigenvalues of } \Sigma,\;}
$$

i.e. the fraction of $X$-variance captured by the first $M$ principal components is the cumulative sum of the top-$M$ eigenvalues of the (standardized) $X$-covariance matrix divided by the total sum of eigenvalues.

The prof made this explicit in [[L15-modelsel-4|L15]] after a student asked about it the prior day:

> "The eigenvalues of the covariance matrix of $X$ are equal to the variances of the corresponding principal components. So fraction of variance explained by the first $M$ PCs = $\sum_{i=1}^M \lambda_i / \sum_{i=1}^p \lambda_i$. That's what the explained-variance plot is showing: running normalized cumulative sum of eigenvalues."

### 4.1 Why this is a chapter-6 delta

**ISLP ch. 6 does not contain this formula.** The §6.3.1 treatment of [[principal-component-analysis|PCA]] introduces PCs verbally (the green line in Fig 6.14, the loadings $\phi_{11}=0.839, \phi_{21}=0.544$ in eq 6.19) and refers the eigendecomposition treatment to ch. 12 (unsupervised learning). The "fraction of variance explained" object is **defined in ISLP §12.2.3 / eq 12.10** (a Pearson correlation-based formula), not in ch. 6. At the exam table, a question about scree plots / PVE on a module-6 context will land you flipping between ch. 6 and ch. 12; this delta puts the eigenvalue formula in one place.

### 4.2 The eigenvalue = PC variance identity

Let $\hat\Sigma = \frac{1}{n} X^\top X$ (after standardization, so $\mathrm{diag}(\hat\Sigma) = 1$). The spectral decomposition $\hat\Sigma = V \Lambda V^\top$ with $\Lambda = \mathrm{diag}(\lambda_1, \dots, \lambda_p)$ ($\lambda_1 \geq \lambda_2 \geq \dots$) and orthogonal $V$ gives the loadings: $\phi_{\cdot m} = v_m$ (the $m$-th eigenvector). Then

$$
\mathrm{Var}(Z_m) = \mathrm{Var}(X v_m) = v_m^\top \hat\Sigma v_m = \lambda_m.
$$

So $\lambda_m$ is **literally the variance of the $m$-th PC**. The total variance in the standardized data is $\sum_j \mathrm{Var}(X_j) = p$ (each standardized column has variance $1$) and equivalently $\sum_j \lambda_j = \mathrm{tr}(\hat\Sigma) = p$. The slide formula is the cumulative fraction. The prof flagged the chain *"eigenvalue → PC variance → cumulative fraction"* as the right way to read a scree plot in [[L15-modelsel-4|L15]]; the algebra above is the one-line reason.

(Per [[docs/scope]], spectral-decomposition derivations are deferred to Linear Statistical Models, but the *identity* eigenvalue = PC variance is in scope as the slide gives the formula and the prof read it aloud.)

---

## 5. Elastic net penalized objective

[[[L13-modelsel-2|L13]], concept: [[elastic-net]]; ISLP §6.2.2 brief mention only]

ISLP §6.2.2 (last paragraphs before §6.2.3) mentions [[elastic-net|elastic net]] by name as a hybrid between [[ridge-regression|ridge]] and [[lasso]] but **does not write the objective**. The prof wrote it on the board in [[L13-modelsel-2|L13]]:

$$
\boxed{\;\hat\beta_{\text{enet}} = \arg\min_\beta \left\{ \mathrm{RSS} + \lambda \sum_{j=1}^p \beta_j^2 + \gamma \sum_{j=1}^p |\beta_j| \right\}.\;}
$$

Two regularization parameters: $\lambda \geq 0$ (L2 strength), $\gamma \geq 0$ (L1 strength), both tuned by [[cross-validation]] over a 2D grid.

### 5.1 Library reparameterization

The standard library form (glmnet, sklearn) uses a single overall strength $\Lambda$ and a mixing parameter $\alpha \in [0,1]$:

$$
\hat\beta_{\text{enet}} = \arg\min_\beta \left\{ \mathrm{RSS} + \Lambda \left[\,\alpha \sum_j |\beta_j| + (1-\alpha) \sum_j \beta_j^2 \,\right] \right\}.
$$

Match-up: $\gamma = \Lambda \alpha$, $\lambda = \Lambda(1-\alpha)$. Edge cases:
- $\alpha = 1$ ⇒ pure lasso.
- $\alpha = 0$ ⇒ pure ridge.
- $\alpha \in (0,1)$ ⇒ elastic net.

The prof noted *"often parameterized slightly differently in libraries, but the idea is the same"* [[L13-modelsel-2|L13]]. Both forms are reproduced here so neither comes as a surprise at the exam table.

### 5.2 Constraint-region geometry

The elastic-net constraint region is the **rounded diamond** $\{\beta : (1-\alpha) \|\beta\|_2^2 + \alpha \|\beta\|_1 \leq s\}$: corners on the axes (sparsity from L1) plus rounded edges (averaging from L2). See [[ridge-vs-lasso-geometry]] for the underlying logic; this object is *not* drawn in ISLP Fig 6.7.

---

## 6. Bias-variance decomposition under the linear model (slide-flagged scaling)

[[[L13-modelsel-2|L13]], slide deck; supplemental to ISLP §6.2.1 Fig 6.5]

The [[bias-variance-tradeoff|bias-variance trade-off]] is discussed in ISLP §2.2.2 (introduction) and visualized for ridge in §6.2.1 Fig 6.5. What ISLP does **not** explicitly carry into chapter 6 is the prof's restated **algebraic claim** that motivates the entire module:

> "If you increase the bias a little bit — you can reduce the variance a lot. Because you have the squared term there." [[L13-modelsel-2|L13]]

Spelled out: at a fixed prediction point $x_0$, the expected squared prediction error decomposes as

$$
\mathbb{E}\bigl[(y_0 - \hat f(x_0))^2\bigr] = \underbrace{\bigl(\,\mathbb{E}[\hat f(x_0)] - f(x_0)\,\bigr)^2}_{\text{Bias}^2} \;+\; \underbrace{\mathrm{Var}(\hat f(x_0))}_{\text{Variance}} \;+\; \sigma^2.
$$

The prof's stylised observation: if Bias is multiplied by a small factor $1 + \epsilon$, the Bias² term grows like $(1+\epsilon)^2 \approx 1 + 2\epsilon$ — **linearly small** in $\epsilon$. So a small *increase* in bias buys a comparatively *large* decrease in variance, provided the variance term was larger to begin with — which it is when $p$ is comparable to $n$.

This is in-scope per the prof's explicit exam flag ([[L13-modelsel-2|L13]]):

> "I mentioned that this is definitely going to be on the exam. I mean, just this concept. … It's like an onion with layers."

ISLP ch. 6 visualizes this with Fig 6.5 but doesn't repeat the §2.2.2 derivation; the prof rederived it on the board in L13. Treat the derivation as needing the algebra from ISLP §2.2.2 (eq 2.7) — but the **link to ridge/lasso shrinkage as the active lever** is the module-6 framing the book leaves out.

---

## 7. Algorithm counts (for fill-in / true-false)

[[[L12-modelsel-1|L12]], slide deck, concept: [[subset-selection]]; ISLP §6.1.1, §6.1.2 prose]

ISLP states the counts in prose ("there are $2^p$ models", "forward stepwise requires fitting only 211 models for $p=20$"). The prof drilled the *formulas* on the board. Reproduce them in clean lookup form:

$$
\boxed{\;\text{Best subset: } \sum_{k=0}^p \binom{p}{k} = 2^p \text{ models.}\;}
$$

$$
\boxed{\;\text{Forward / backward / hybrid stepwise: } 1 + \sum_{k=0}^{p-1}(p - k) = 1 + \frac{p(p+1)}{2} \text{ models.}\;}
$$

For $p = 20$: $2^{20} = 1{,}048{,}576$ vs $211$.

The $1 + p(p+1)/2$ formula is the same for all three stepwise variants because each scans through $p, p-1, p-2, \dots, 1$ candidates plus the null model. ISLP states the *result* "$1 + p(p+1)/2$" once mid-paragraph (§6.1.2); the **derivation $1 + \sum_{k=0}^{p-1}(p-k)$ that the slide shows** is what makes this a clean fill-in. Counted as delta because the prof gave both forms on the slide and the second form is not in ISLP.

### Hard requirement (slide-flagged, [[L12-modelsel-1|L12]]):

> "Backwards selection requires that the number of samples $n$ is larger than the number of parameters."

Because step 1 fits the full OLS model, which needs $X^\top X$ invertible, which needs $n > p$. **Forward stepwise has no such requirement** and can be run when $n \leq p$ by capping the algorithm at submodels $\mathcal{M}_0, \dots, \mathcal{M}_{n-1}$ (slide-flagged).

ISLP §6.1.2 states this in prose; the prof drilled it as an exam-pattern fill-in. The combination *forward survives $n < p$, backward doesn't* is the canonical MC.

---

## 8. PLS algorithm (clean form)

[[[L15-modelsel-4|L15]], slide deck "Partial Least Squares (Algorithm)", concept: [[partial-least-squares]]]

ISLP §6.3.2 (pp. 286–288) describes PLS in two paragraphs and writes only *"$\phi_{j1}$ equal to the coefficient from the simple linear regression of $Y$ onto $X_j$"* and the deflation step. The prof gave the same content but the slide labels it as a multi-step algorithm; reproduce as a numbered algorithm for exam lookup.

After standardizing each $X_j$ and centering $Y$:

**For $m = 1$:**

1. For each $j = 1, \dots, p$, regress $Y$ on $X_j$ alone (simple linear regression). The coefficient is
   $$
   \hat\beta_j^{\text{(simple)}} = \frac{\sum_i X_{ij} Y_i}{\sum_i X_{ij}^2} \propto \mathrm{Corr}(Y, X_j).
   $$
2. Set $\phi_{j1} := \hat\beta_j^{\text{(simple)}}$, so
   $$
   Z_1 = \sum_{j=1}^p \phi_{j1} X_j.
   $$

**For $m = 2, \dots, M$:**

3. Orthogonalize: for each $X_j$, regress $X_j$ on $Z_1, \dots, Z_{m-1}$ and take residuals. Call the residualized predictors $X_j^{\text{(orth)}}$.
4. Compute $Z_m$ from $X^{\text{(orth)}}$ using exactly the procedure of steps 1–2 (regress each $X_j^{\text{(orth)}}$ on $Y$, use those coefficients as $\phi_{jm}$).

**Final fit:**

5. Least-squares regression of $Y$ on $Z_1, \dots, Z_M$. $M$ chosen by cross-validation.

### The key contrast with PCR (the only conceptual delta worth memorizing)

| | PCR | PLS |
|---|---|---|
| Objective for $\phi_m$ | $\max \mathrm{Var}(X\phi_m)$ s.t. $\|\phi_m\| = 1$ | $\max \mathrm{Cov}(X\phi_m, Y)$ s.t. $\|\phi_m\| = 1$ |
| Supervision | Unsupervised (uses $X$ only) | Supervised (uses $X$ and $Y$) |
| First-component construction | Eigenvector of $X^\top X$ | $\phi_{j1} \propto \mathrm{Corr}(Y, X_j)$ |

ISLP §6.3.2 states the contrast; this table reproduces it as a side-by-side lookup with the explicit objective functions, which the book gives only in prose.

---

## 9. Slide-flagged scope items the book uses different definitions for

[[[L15-modelsel-4|L15]], slide deck; supplemental to ISLP §6.4]

Two artifacts the prof endorsed verbatim from the slide and the book hides differently:

### 9.1 Multicollinearity, extreme version

The prof read this aloud and endorsed it [[L15-modelsel-4|L15]]:

> **Multicollinearity in high dim:** *"any variable in the model can be written as a linear combination of all of the other variables in the model."* — slide deck

ISLP §6.4.4 states this in prose (p. 293 in the printed edition) — but the prof's wording is the version that appeared on the slide and is the version Anders should recognize on an exam stem.

### 9.2 What can and cannot be recovered

Verbatim slide endorsed by the prof:

> "We can never know exactly which variables (if any) truly are predictive of the outcome. We can never identify the best coefficients for use in the regression. At most, we can hope to assign large regression coefficients to variables that are correlated with the variables that truly are predictive of the outcome. We will find one of possibly many suitable predictive models."

ISLP §6.4.4 says the same thing in different words ("there are likely to be many sets of 17 SNPs that would predict blood pressure just as well as the selected model"). The slide framing is the one the prof committed to.

---

## 10. Notation / terminology drift between prof and ISLP

[[[L12-modelsel-1|L12]], [[L13-modelsel-2|L13]], [[L14-modelsel-3|L14]], [[L15-modelsel-4|L15]]]

Minor, mostly cosmetic. Listed for safety at the exam table.

- **Ridge regression**. Prof switches freely between "ridge", "L2", and (occasionally) **"Tikhonov"** ([[L13-modelsel-2|L13]]: *"It's entirely possible that I will just start calling it L2 one day."*). ISLP uses "ridge regression" exclusively.
- **Lasso**. Prof: "lasso" or "L1". ISLP: "lasso" or "the lasso". Both use $\ell_1$ / $\ell_2$ norm notation interchangeably with "L1" / "L2".
- **Tuning parameter** $\lambda$. Same symbol in both, with one *internal* clash that the slide deck does **not** resolve: in the PCR/ridge shrinkage-factor formula $\lambda_j^2/(\lambda_j^2 + \lambda)$, the $\lambda_j$ are **eigenvalues of $X^\top X$** and the $\lambda$ is the **ridge tuning parameter**. The prof flags this orally in [[L15-modelsel-4|L15]] (*"this is a confusing figure, I'll try to make another one for next time"*) but the formula stayed as-is on the slide. ISLP avoids the clash entirely by deferring to ESL.
- **Effective parameters / effective degrees of freedom**. Prof says "effective parameters" verbally ([[L12-modelsel-1|L12]]: *"reduce the number of parameters effectively"*); ISLP uses "degrees of freedom" only for lasso (number of non-zero coefficients) and never assigns a numerical $\mathrm{df}$ to ridge in ch. 6.
- **$M$ vs $d$ vs $k$**. Number of components / predictors / subset size: ISLP uses $d$ for subset selection (algorithms 6.1–6.3) and $M$ for dimension reduction (eq 6.16). Prof uses $k$ for subset size in [[L12-modelsel-1|L12]] then $M$ for PCR/PLS. No source uses these symbols consistently across all three families.
- **Standardization formula**. ISLP eq 6.6 uses $\tilde x_{ij} = x_{ij}/\sqrt{\frac{1}{n}\sum_i (x_{ij} - \bar x_j)^2}$ (centering implicit in $\bar x_j$). Slide deck reproduces this verbatim. Both also accept "subtract mean, divide by SD" interchangeably.
- **"Capitalist vs socialist."** Prof's only-in-lecture framing ([[L13-modelsel-2|L13]], [[L14-modelsel-3|L14]]) for lasso vs ridge. Not in ISLP. Useful for explanations; don't write it on the exam unironically.

---

## Coverage map (what's reproduced here vs. what's lookup-only in ISLP ch. 6)

| Item | Source | This file? | ISLP location |
|---|---|---|---|
| Best subset model count $2^p$ | §6.1.1 | Yes (§7) | eq before 6.1.1, prose |
| Stepwise model count $1 + p(p+1)/2$ | §6.1.2 | Yes (§7) | footnote 2, prose |
| Backward needs $n > p$ | [[L12-modelsel-1|L12]] / slide | Yes (§7) | §6.1.2 p. 247 prose |
| $C_p$ / AIC / BIC / adj-$R^2$ formulas | §6.1.3 eqs 6.2–6.4 | **No** (out of scope) | §6.1.3 |
| Ridge penalized objective | eq 6.5 | Lookup | §6.2.1 eq 6.5 |
| **Ridge closed form $(X^\top X+\lambda I)^{-1}X^\top y$** | — | **Yes (§1)** | not in ch. 6 |
| **Ridge sampling distribution** | — | **Yes (§1.4)** | not in ch. 6 |
| **Ridge hat matrix $H_\lambda$, $\mathrm{df}(\lambda)$** | — | **Yes (§2)** | not in ch. 6 |
| **Per-PC ridge shrinkage $\lambda_j^2/(\lambda_j^2+\lambda)$** | slide / [[L15-modelsel-4|L15]] | **Yes (§3)** | footnote 8 → ESL §3.5 |
| Ridge orthonormal-case $\hat\beta_j/(1+\lambda)$ | eq 6.14 | Lookup | §6.2.2 |
| Standardization formula | eq 6.6 | Lookup | §6.2.1 |
| Lasso penalized objective | eq 6.7 | Lookup | §6.2.2 |
| Lasso constraint form $\sum|\beta_j| \leq s$ | eq 6.8 | Lookup | §6.2.2 |
| Ridge/lasso constraint geometry (Fig 6.7) | Fig 6.7 | Lookup | §6.2.2 |
| Soft-thresholding $\hat\beta^L_j$ (orthonormal) | eq 6.15 | Lookup | §6.2.2 |
| Bayesian Gaussian/Laplace priors | §6.2.2 | **No** (out of scope) | §6.2.2 |
| **Elastic net penalized objective** | [[L13-modelsel-2|L13]] / slide | **Yes (§5)** | mentioned only |
| Bias-variance decomposition (for ridge) | Fig 6.5 | Pointer (§6) | §2.2.2 + §6.2.1 |
| Dimension reduction setup $Z_m = \sum \phi_{jm} X_j$ | eq 6.16 | Lookup | §6.3 |
| Back-transform $\beta_j = \sum_m \theta_m \phi_{jm}$ | eq 6.18 | Lookup | §6.3 |
| **PCA fraction of variance via eigenvalues** | slide / [[L15-modelsel-4|L15]] | **Yes (§4)** | not in ch. 6 (in §12.2.3) |
| **Eigenvalue = variance of corresponding PC** | [[L15-modelsel-4|L15]] | **Yes (§4.2)** | not in ch. 6 |
| PCR algorithm | §6.3.1 | Lookup | §6.3.1 |
| **PLS algorithm (numbered)** | slide | **Yes (§8)** | §6.3.2 prose |
| High-dim breakdown of OLS | §6.4.2 | Lookup | §6.4.2 |
| **Slide-flagged multicollinearity statement** | slide / [[L15-modelsel-4|L15]] | **Yes (§9)** | §6.4.4 in prose |

---

## What's deliberately *not* in this file

Per [[docs/scope]]:

- $C_p$ / AIC / BIC / adjusted $R^2$ algebra (out of scope, conceptual claim only — see [[aic-bic-conceptual]]).
- Bayesian interpretation of ridge / lasso (Gaussian / Laplace priors) — out of scope per [[L14-modelsel-3|L14]].
- L0 norm / "Optimal Brain Damage" — out of scope per [[L14-modelsel-3|L14]].
- Full spectral / eigendecomposition derivations beyond stating the identity — out of scope per [[L04-statlearn-3|L04]].
- Detailed PLS history / chemometrics-specific tuning — out of scope per [[L14-modelsel-3|L14]] / [[L15-modelsel-4|L15]].
- Elastic net detailed tuning (the *form* in §5 is in scope; the L1/L2-mixing optimization details are not).
- Moore-Penrose pseudoinverse details — out of scope (prof flagged in [[L08-classif-2|L08]]).
- R / Python code, package names, function syntax — out of scope per [[docs/scope]] §"Programming policy".
