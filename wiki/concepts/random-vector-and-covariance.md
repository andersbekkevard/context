---
concept: random-vector-and-covariance
module: 02-statlearn
lectures: [L04, L05, L09]
isl-ref: 2.1
exercises:
  - Exercise2.3g  -  compute correlations between mpg and (displacement, horsepower, weight) using only the covariance matrix; compare to cor()
  - Exercise2.4  -  simulate 1000 draws from a bivariate normal with mvrnorm() under four covariance settings (independent / scaled / positive cov / negative cov)
  - CE1 problem 1f  -  read the correlation between two variables off a 2×2 covariance matrix (single-choice MC)
related: [contrasts, multivariate-normal, linear-regression, sampling-distribution-of-beta, linear-discriminant-analysis, principal-component-analysis, collinearity]
tags:
  - concept
  - module/02-statlearn
aliases:
  - random vector
  - covariance matrix
  - correlation matrix
  - Sigma
---

# Random vectors, covariance and correlation matrices

The matrix-algebra plumbing the whole rest of the course rides on. A random vector $\mathbf X$ is a $p$-vector of random variables; its **covariance matrix** $\boldsymbol\Sigma$ stacks variances on the diagonal and covariances off; its **correlation matrix** is $\boldsymbol\Sigma$ rescaled by the standard-deviation diagonal so the diagonal is 1. Expectations of linear transformations follow $E(A\mathbf X B) = A \cdot E(\mathbf X) \cdot B$ and $\mathrm{Cov}(C\mathbf X) = C \boldsymbol\Sigma C^\top$. Covariance is a *linear* notion, zero covariance does not mean independence except under joint normality.

## Definition (prof's framing)

> "A random vector $\mathbf X_{(p \times 1)}$ is a $p$-dimensional vector of random variables." - [[L04-statlearn-3]]

Mean vector: element-wise $E(\mathbf X) = (E(X_1), \dots, E(X_p))^\top$. Covariance matrix:

> "$\sigma_{ij} = \mathrm{Cov}(X_i, X_j) = E[(X_i - \mu_i)(X_j - \mu_j)]$. … if they have a high covariance then they're varying together; if they have a negative covariance then they're varying the opposite way; and if it's just all random then the covariance is going to be small in magnitude." - [[L04-statlearn-3]]

When $i = j$: it's the **variance** $\sigma_i^2$. The prof flagged this as a quiz-style fact.

## Notation & setup

- $\mathbf X = (X_1, \dots, X_p)^\top$: $p$-dimensional random vector.
- $\boldsymbol\mu = E(\mathbf X) = (E(X_1), \dots, E(X_p))^\top$: mean vector.
- $\boldsymbol\Sigma = \mathrm{Cov}(\mathbf X) = E[(\mathbf X - \boldsymbol\mu)(\mathbf X - \boldsymbol\mu)^\top]$: $p \times p$ covariance matrix. Symmetric and (by construction) positive semi-definite.
- $\sigma_{ij}$ = $(i,j)$ entry of $\boldsymbol\Sigma$, the covariance of $X_i$ and $X_j$. Diagonal entries $\sigma_{ii} = \sigma_i^2$ are variances.
- $\rho_{ij} = \sigma_{ij} / (\sigma_i \sigma_j)$: correlation. Lives in $[-1, 1]$.
- $\boldsymbol{V}^{1/2}$ = diagonal matrix of standard deviations. Then correlation matrix $\boldsymbol\rho = (\boldsymbol V^{1/2})^{-1} \boldsymbol\Sigma (\boldsymbol V^{1/2})^{-1}$.

## Formula(s) to know cold

**Definition / shortcut**:
$$\boldsymbol\Sigma = E[(\mathbf X - \boldsymbol\mu)(\mathbf X - \boldsymbol\mu)^\top] = E(\mathbf X \mathbf X^\top) - \boldsymbol\mu \boldsymbol\mu^\top$$

**Correlation from covariance** (the CE1.1f / Exercise 2.3g calculation):
$$\rho_{ij} = \frac{\sigma_{ij}}{\sqrt{\sigma_i^2 \sigma_j^2}} = \frac{\sigma_{ij}}{\sigma_i \sigma_j}$$

**Expectation rules** ([[L04-statlearn-3]] proved on board):
$$E(\mathbf X + \mathbf Y) = E(\mathbf X) + E(\mathbf Y), \qquad E(A \mathbf X B) = A \cdot E(\mathbf X) \cdot B$$

**Covariance of a linear transformation** $\mathbf Z = C \mathbf X$ (used heavily in [[contrasts]] and in deriving the [[sampling-distribution-of-beta]]):
$$E(\mathbf Z) = C \boldsymbol\mu, \qquad \boxed{\;\mathrm{Cov}(\mathbf Z) = C \boldsymbol\Sigma C^\top\;}$$

These are the two formulas the M2 quiz drills (Q4 and Q5 of the random-vectors menti, `modules/2StatLearn/2StatLearn.2.md`).

**Univariate analogues** (for sanity checks): $E(aX + b) = a E(X) + b$, $\mathrm{Var}(aX + b) = a^2 \mathrm{Var}(X)$.

## Insights & mental models

**Element-wise proof of $E(A\mathbf X B) = A \cdot E(\mathbf X) \cdot B$** ([[L04-statlearn-3]] on the board):
$e_{ij} = \sum_k \sum_l a_{ik} X_{kl} b_{lj}$. Take $E$, pull constants out, element-wise it's the $(i, j)$ entry of $A \cdot E(\mathbf X) \cdot B$. The prof flagged this as basically obvious but did the proof anyway, it's the working machinery for everything in M3 onwards.

**Covariance is a *linear* notion**, the prof's most-emphasized framing in [[L04-statlearn-3]]:

> "We talk about things, I'm kind of always doing this [drawing a line], because we're sort of assuming a linear line. We're assuming some kind of linear function. This covariance is really getting at this notion of a slope." - [[L04-statlearn-3]]

So **zero covariance does not mean independent** in general. It just means "no linear co-variation." Two variables can be perfectly dependent (e.g. $Y = X^2$) but have zero covariance if the dependence is symmetric around zero. The exception is **joint normality**, zero covariance ⇒ independence, but only there. (See [[multivariate-normal]] for that property.)

**The correlation matrix is the covariance matrix made unit-free**: divide entry $\sigma_{ij}$ by $\sqrt{\sigma_i^2 \sigma_j^2}$. Diagonal becomes 1; off-diagonals become Pearson correlations in $[-1, 1]$. Lets you compare across different variables / units.

**Positive semi-definite is forced by the variance interpretation** (`modules/2StatLearn/2StatLearn.2.md` hint): for any constant vector $\mathbf b \neq 0$, $\mathrm{Var}(\mathbf b^\top \mathbf X) = \mathbf b^\top \boldsymbol\Sigma \mathbf b \geq 0$. So $\boldsymbol\Sigma$ is positive semi-definite by construction. If it's *singular* (det = 0), some linear combination of the $X_j$'s has zero variance, meaning it's a deterministic function of the others, and the multivariate normal density doesn't exist (you divide by $|\boldsymbol\Sigma|^{1/2}$).

**The cork-deposit example** ([[L04-statlearn-3]] running dataset): $n = 28$ cork trees, $p = 4$ holes drilled (N, E, S, W), measured weight per direction. Variables are *very* correlated within a tree (sun exposure aside, dense in one direction usually means dense in all). Used as the toy multivariate dataset for the $\Sigma \to \rho$ calculation drill, the [[contrasts]] example (N − S, E + W, etc.), and the matching exercise.

**The $\Sigma \to \rho$ calculation by hand** (Exercise 2.3g and the CE1.1f single-choice): compute $\rho_{ij} = \sigma_{ij} / \sqrt{\sigma_i^2 \sigma_j^2}$. CE1.1f gives $\boldsymbol\Sigma = \begin{bmatrix} 9 & 0.3 \\ 0.3 & 4 \end{bmatrix}$ and asks for $\rho_{12} = 0.3 / \sqrt{9 \cdot 4} = 0.3 / 6 = 0.05$. Multiple-choice trap: "0.0083" comes from forgetting the square root; "0.15" from $0.3/2$; "0.10" from $0.3/3$. Take the square root.

### Where this plumbing leads

- [[multivariate-normal]]: generalizes the bell curve using $\boldsymbol\Sigma$ in the exponent (and $|\boldsymbol\Sigma|$ in the normalizer).
- [[sampling-distribution-of-beta]]: under Gaussian errors, $\hat\beta \sim N(\beta, \sigma^2 (X^\top X)^{-1})$. The covariance machinery is what gives you the standard errors.
- [[linear-discriminant-analysis]] / [[quadratic-discriminant-analysis]], class-conditional densities are multivariate normals with means $\boldsymbol\mu_k$ and covariance $\boldsymbol\Sigma$ (LDA) or $\boldsymbol\Sigma_k$ (QDA).
- [[principal-component-analysis]]: eigen-decomposition of $\boldsymbol\Sigma$ gives the PCs (the prof defers spectral theory to Linear Statistical Models, [[scope]] M2 out-of-scope, but the PVE story rides on $\Sigma$'s eigenvalues).
- [[collinearity]]: collinear predictors → $X^\top X$ near-singular → $(X^\top X)^{-1}$ blows up → SEs explode.

## Pitfalls

- **Zero covariance ≠ independence in general.** Only under joint normality. Easy T/F trap.
- **Forgetting the square root in $\rho_{ij} = \sigma_{ij} / \sqrt{\sigma_i^2 \sigma_j^2}$**: that's how the CE1.1f distractors are designed.
- **Symmetric proof for $\mathrm{Cov}(C\mathbf X) = C \boldsymbol\Sigma C^\top$, not $C^\top \boldsymbol\Sigma C$.** Order matters; transpose goes on the right.
- **Singular $\boldsymbol\Sigma$ (det = 0) means at least one variable is a perfect linear combination of the others**: multivariate normal density doesn't exist; LDA's $\Sigma^{-1}$ blows up; PCA has zero eigenvalue.
- **Be careful with row-vs-column conventions for the data matrix.** ISLR uses rows = observations, columns = variables. The prof's L02 notes the book convention and flags that other books transpose it.
- **Pearson correlation is *linear* correlation only.** A perfect quadratic relationship can have $\rho = 0$.

## Scope vs ISLR

- **In scope:** definition of random vector, mean vector, $\boldsymbol\Sigma$, $\boldsymbol\rho$, the two expectation rules, $\mathrm{Cov}(C\mathbf X) = C \boldsymbol\Sigma C^\top$, the linear-vs-independence distinction, $\Sigma \to \rho$ hand calculation.
- **Look up in ISLR:** §2.1 (introduction to notation), §3.2.4 (sampling distributions of regression coefficients) and §4.4 (LDA/QDA's use of $\boldsymbol\Sigma$). Hardle / Simar or Johnson & Wichern would be the deeper references; ISLR keeps the matrix algebra light.
- **Skip in ISLR:** spectral / eigen-decomposition theory of $\boldsymbol\Sigma$. The prof verbatim ([[L04-statlearn-3]]): "we don't talk about spectral decomposition", deferred to TMA4267 Linear Statistical Models. Eigenvalues come back as PC variances in M10, but the full spectral machinery is out.

## Exercise instances

- **Exercise 2.3g**: given the covariance matrix of the Auto data's quantitative columns, compute correlations between `mpg` and `displacement` / `horsepower` / `weight` by hand using $\rho_{ij} = \sigma_{ij} / \sqrt{\sigma_i^2 \sigma_j^2}$; verify against `cor(Auto[, quant])`. The drill that turns the formula into muscle memory.
- **Exercise 2.4**: simulate 1000 draws from a bivariate normal with `mvrnorm()` under four $\boldsymbol\Sigma$ settings: (i) $\mathrm{diag}(1, 1)$, (ii) $\mathrm{diag}(1, 5)$, (iii) $\begin{bmatrix} 1 & 2 \\ 2 & 5 \end{bmatrix}$, (iv) $\begin{bmatrix} 1 & -2 \\ -2 & 5 \end{bmatrix}$. Plot, identify which scatter goes with which $\boldsymbol\Sigma$. Builds the visual mapping from $\boldsymbol\Sigma$ to point-cloud shape, directly relevant to CE1.1g (contour matching).
- **CE1 problem 1f**: single-choice MC: given $\boldsymbol\Sigma = \begin{bmatrix} 9 & 0.3 \\ 0.3 & 4 \end{bmatrix}$, what's $\rho_{12}$? Answer: $0.3/\sqrt{9 \cdot 4} = 0.05$. The square-root-trap MC.

## How it might appear on the exam

- **MC: correlation from a 2×2 $\boldsymbol\Sigma$.** Direct CE1.1f format.
- **T/F: zero covariance ⇒ independence.** False in general; true only for joint normality. Classic trap.
- **Hand calculation: $\mathrm{Cov}(C \mathbf X) = C \boldsymbol\Sigma C^\top$ for a small $C$.** Plug-and-chug; the [[contrasts]] cork-data exercise is the template.
- **Match scatter / contour plot to $\boldsymbol\Sigma$.** The Exercise 2.4 visual: independent vs scaled vs positively-correlated vs negatively-correlated. The [[multivariate-normal]] atom owns the contour-matching question (CE1.1g), but the underlying $\boldsymbol\Sigma$-to-shape intuition lives here.
- **Identify which $\boldsymbol\Sigma$ is singular** (where $|\boldsymbol\Sigma| = 0$), and what that means for the density / for LDA.
- **Quiz-style fact recall.** "What is $\mathrm{Cov}(X_i, X_i)$?" → $\mathrm{Var}(X_i)$. From [[L04-statlearn-3]]'s flagged quiz fact.

## Related

- [[contrasts]]: the canonical application of $\mathrm{Cov}(C\mathbf X) = C \boldsymbol\Sigma C^\top$
- [[multivariate-normal]]: uses $\boldsymbol\Sigma$ in the density; gives the joint-normal-only result that zero cov ⇒ independence
- [[linear-regression]], [[sampling-distribution-of-beta]], $\hat\beta \sim N(\beta, \sigma^2 (X^\top X)^{-1})$ uses this same machinery
- [[linear-discriminant-analysis]] / [[quadratic-discriminant-analysis]], class-conditional Gaussians built on $\boldsymbol\Sigma$
- [[principal-component-analysis]]: eigen-decomposition of $\boldsymbol\Sigma$ (mechanics in M10; deferred from M2)
- [[collinearity]]: what happens when $\boldsymbol\Sigma$ (or $X^\top X$) is near-singular
