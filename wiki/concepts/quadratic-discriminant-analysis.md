---
concept: quadratic-discriminant-analysis
module: 04-classif
lectures: [L09]
isl-ref: 4.4.3
exercises:
  - Exercise4.2d  -  derive QDA rule, classify the bank note, compare to LDA result
  - Exercise4.6f  -  repeat lag-based classification with QDA on Weekly
  - CE1 problem 3g  -  perform QDA, confusion matrix, sensitivity/specificity
  - CE1 problem 3h  -  compare LDA, QDA, logistic decision boundaries
related: [linear-discriminant-analysis, discriminant-score-and-decision-boundary, naive-bayes, multivariate-normal, bias-variance-tradeoff, diagnostic-vs-sampling-paradigm]
tags:
  - concept
  - module/04-classif
aliases:
  - QDA
---

# Quadratic discriminant analysis (QDA)

The prof's framing: **"LDA but you stop pooling the covariance."** Each class gets its own $\Sigma_k$ → the $x^\top \Sigma_k^{-1} x$ term no longer cancels across classes → discriminant becomes **quadratic in $x$** → boundaries are curves. More flexible, more parameters, more variance, exactly the bias-variance trade-off Anders has seen since module 2. The "where does the quadratic come from?" question is exam-flagged.

## Definition (prof's framing)

> "QDA = LDA without pooling. Sigma becomes class-specific → the $x^\top \Sigma_k^{-1} x$ term no longer cancels across $k$ → discriminant is quadratic in $x$ → boundaries are curves, not lines." - [[L09-classif-3]]

> "Drop the pooled-variance assumption. Each class gets its own covariance matrix $\Sigma_k$. Everything else identical: still multivariate Gaussian for $f_k$, still estimate $\pi_k$ from frequencies, still apply Bayes." - [[L09-classif-3]]

The model assumption: $X \mid Y = k \sim \mathcal{N}(\mu_k, \Sigma_k)$, with $\Sigma_k$ class-specific.

## Notation & setup

- Same $\pi_k$, $\mu_k$, $f_k(x)$ as [[linear-discriminant-analysis|LDA]].
- $\Sigma_k$: **class-$k$**-specific covariance matrix (no pooling).
- Estimator: $\hat\Sigma_k = \dfrac{1}{n_k - 1}\sum_{i: y_i = k}(x_i - \hat\mu_k)(x_i - \hat\mu_k)^\top$, the within-class sample covariance.

## Formula(s) to know cold

**QDA discriminant score** (keep the $x^\top \Sigma_k^{-1} x$ term, it now depends on $k$):

$$\delta_k(x) = -\tfrac{1}{2} x^\top \Sigma_k^{-1} x + x^\top \Sigma_k^{-1} \mu_k - \tfrac{1}{2} \mu_k^\top \Sigma_k^{-1} \mu_k - \tfrac{1}{2}\log|\Sigma_k| + \log\pi_k$$

Equivalent compact form:

$$\delta_k(x) = -\tfrac{1}{2}(x - \mu_k)^\top \Sigma_k^{-1}(x - \mu_k) - \tfrac{1}{2}\log|\Sigma_k| + \log\pi_k$$

Classify by $\hat y = \arg\max_k \delta_k(x)$. Decision boundaries, set $\delta_k = \delta_\ell$, are **quadratic** in $x$ (conic sections).

## Where the quadratic comes from (the exam-flagged derivation)

In LDA, the $-\tfrac{1}{2} x^\top \Sigma^{-1} x$ piece has **no $k$** (since $\Sigma$ is shared) → cancels in $\arg\max_k$ → drops. Discriminant linear.

In QDA, $\Sigma$ becomes $\Sigma_k$ → that piece **does** depend on $k$ → cannot drop → survives as $-\tfrac{1}{2} x^\top \Sigma_k^{-1} x$ → discriminant quadratic in $x$. Plus an extra $-\tfrac{1}{2}\log|\Sigma_k|$ from the Gaussian normalizing constant (which previously canceled because $|\Sigma|$ was constant in $k$).

> "It's an interesting point that simply by making the sigma $k$-dependent, we introduced a new term, and that term is quadratic. So now the decision boundary is no longer going to be a line, it's going to be quadratic." - [[L09-classif-3]]

## Insights & mental models

- **Bias-variance trade-off in pure form.** QDA has lower bias (no pooling assumption) but higher variance (more parameters). Same algebra as linear-vs-polynomial regression: more flexibility → more parameters → more variance.
- **Parameter count is the headline number:**
  - **LDA:** one shared $\Sigma$ → $p(p+1)/2$ parameters.
  - **QDA:** $K$ class-specific $\Sigma_k$'s → $K \cdot p(p+1)/2$ parameters.
  - For $p = 100$, $K = 5$: LDA needs ~5,050; QDA needs ~25,250. - [[L09-classif-3]]
- **When QDA pays off:** large $n$, classes genuinely have different covariance structure, you can afford the extra parameters.
- **When LDA wins:** small $n$, variance dominates and the simpler pooled model generalizes better. The iris example below makes this tangible.
- **Curves not lines.** Decision boundaries are conic sections (parabolas, hyperbolas, ellipses). For a single class with $\Sigma_k$ much "fatter" than $\Sigma_\ell$, the boundary curls around the fat class.

## Worked iris example (the prof's case for LDA)

Two predictors (sepal length × width), three iris species. Random 50/50 train/test split:

| Method | Train error | Test error |
|--------|-------------|------------|
| LDA    | 0.19        | 0.17       |
| QDA    | 0.17        | 0.32       |

QDA fits training slightly better (it must, strictly more flexible) but **doubles** test error. Classic overfit.

> "If this was my situation… I would go with LDA. The argument would be it's doing better on held-out data." - [[L09-classif-3]]

(Foreshadows [[cross-validation]], would be more robust to compare with multiple splits.)

## Exam signals

> "That's another good exam question, where does the quadratic come from in QDA? Or show that, yeah… it's an interesting point that simply by making the sigma $k$-dependent, we introduced a new term, and that term is quadratic." - [[L09-classif-3]]

> "Why ever use LDA over QDA? Same reason linear regression often beats quadratic, when you don't have enough data to estimate the extra parameters reliably, the simpler model wins on test error." - [[L09-classif-3]]

> "QDA: Gaussian holds but $\Sigma_k$ unequal. Need enough data." - [[L09-classif-3]]

## Pitfalls

- **Forgetting the $\log|\Sigma_k|$ term.** It survives in QDA (didn't survive in LDA because $|\Sigma|$ was constant in $k$).
- **Treating QDA as always better than LDA.** Strictly more flexible ≠ better, bias-variance argument.
- **Computing parameter count without the symmetry constraint.** A $p \times p$ covariance has $p(p+1)/2$ free parameters (not $p^2$, symmetric matrix, only upper triangle is free).
- **Estimating $\Sigma_k$ with too few samples per class.** When $n_k < p$, $\hat\Sigma_k$ is singular and the classifier breaks. (Naive Bayes' diagonal-$\Sigma$ assumption is the standard rescue when $p$ is large.)
- **Reading too much into "QDA is more flexible."** Slide deck: "if the covariance matrices in theory are equal, will they not be estimated equal? Should we not always prefer QDA to LDA?" Answer: no, because of variance.

## Scope vs ISLR

- **In scope:** QDA model assumptions, $\delta_k(x)$ formula, where the quadratic comes from, parameter-count comparison vs LDA, LDA-vs-QDA bias-variance trade-off.
- **Look up in ISLR:** §4.4.3, pp. 152–155. Equation (4.28) is the canonical $\delta_k(x)$.
- **Skip in ISLR:** Detailed simulation scenarios in §4.5.2 (where QDA wins / loses), useful for intuition but not exam-relevant beyond "QDA wins when $\Sigma_k$ truly differ and $n$ is adequate."

## Exercise instances

- **Exercise4.2d**: derive QDA classification rule from the multivariate Gaussian; classify the bank note (length 214, diagonal 140.4) using QDA; compare to LDA result. Emphasizes the explicit $\Sigma_k^{-1}$ + $\log|\Sigma_k|$ algebra.
- **Exercise4.6f**: `qda(Direction ~ Lag2)` on the `Weekly` data; held-out confusion matrix.
- **CE1 problem 3g**: perform QDA in R on the tennis data; confusion matrix; sensitivity/specificity on the test set.
- **CE1 problem 3h**: compare LDA, QDA, and logistic decision boundaries; discuss which to prefer based on the confusion-matrix results.

## How it might appear on the exam

- **"Where does the quadratic come from?"**: flagged exam question. Walk through the algebra: $-\tfrac{1}{2} x^\top \Sigma_k^{-1} x$ no longer cancels when $\Sigma_k$ depends on $k$.
- **Parameter-count MCQ:** "For $p = 10$, $K = 3$, how many free parameters does QDA estimate just for the covariance matrices?" → $3 \cdot 10 \cdot 11 / 2 = 165$.
- **LDA-vs-QDA T/F:** "QDA always has lower test error than LDA" → false (variance can dominate). "QDA always has lower training error than LDA" → true on average (strictly more flexible).
- **Method-choice justification:** Given a confusion matrix or test-MSE table, pick LDA or QDA and justify with a bias-variance argument.
- **Hand-classification:** Given $\hat\mu_k$, $\hat\Sigma_k$, $\hat\pi_k$, plug a new $x_0$ into both $\delta_k(x_0)$'s and pick the larger.
- **Output interpretation:** Compare LDA-decision-boundary plot (line) vs QDA-decision-boundary plot (curve) for the same data; explain.

## Related

- [[linear-discriminant-analysis]]: pool $\Sigma$, get linear boundaries.
- [[discriminant-score-and-decision-boundary]]: procedural atom for deriving $\delta_k$ and equating across classes.
- [[naive-bayes]]: diagonal $\Sigma_k$ (predictors conditionally independent given class), a much-simplified QDA.
- [[multivariate-normal]]: the class-conditional density assumption.
- [[bias-variance-tradeoff]]: the canonical why-not-always-QDA argument.
- [[diagnostic-vs-sampling-paradigm]]: QDA is on the sampling/generative side.
- [[cross-validation]]: how to choose between LDA and QDA in practice.
