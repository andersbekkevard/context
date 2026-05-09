---
concept: naive-bayes
module: 04-classif
lectures: [L09]
isl-ref: 4.4.4
exercises: []
related: [linear-discriminant-analysis, quadratic-discriminant-analysis, diagnostic-vs-sampling-paradigm, multivariate-normal, bias-variance-tradeoff, curse-of-dimensionality]
tags:
  - concept
  - module/04-classif
aliases:
  - Naive Bayes classifier
  - Idiot's Bayes
---

# Naive Bayes

The prof's framing: **"LDA / QDA but you assume $\Sigma$ is diagonal."** Predictors are conditionally independent within each class, which is generally false, but it slashes the parameter count and works surprisingly well. The slide deck calls it "Idiot's Bayes." The win is in **large-$p$** settings where you can't afford the $O(p^2)$ covariance estimates.

## Definition (prof's framing)

> "Naive Bayes: assume the covariance matrix is diagonal, the predictors are conditionally independent given the class. Drops all off-diagonal covariance parameters." - [[L09-classif-3]]

> "Naive Bayes is *optimal or popular when $p$ is large* because of the parameter count, fewer things to estimate, more robust." - [[L09-classif-3]]

The single modeling assumption: within each class $k$, the predictors are independent.

$$f_k(x) = \prod_{j=1}^p f_{kj}(x_j)$$

Plug into Bayes' theorem like LDA/QDA, classify by max posterior.

## Notation & setup

- $f_{kj}(x_j)$: class-$k$, variable-$j$ marginal density. Often Gaussian: $X_j \mid Y = k \sim \mathcal{N}(\mu_{kj}, \sigma_{kj}^2)$, but doesn't have to be.
- $\pi_k$: class prior (same as LDA/QDA).
- For Gaussian marginals: parameter count per class = $2p$ (a mean and variance per predictor) → total $2pK + (K - 1)$ for the priors. **Linear in $p$**, vs $O(p^2)$ for LDA/QDA.

## Formula(s) to know cold

**Posterior under naive Bayes:**

$$\Pr(Y = k \mid X = x) = \frac{\pi_k \prod_{j=1}^p f_{kj}(x_j)}{\sum_\ell \pi_\ell \prod_{j=1}^p f_{\ell j}(x_j)}$$

**Discriminant (Gaussian marginals):**

$$\delta_k(x) \propto -\tfrac{1}{2}\sum_{j=1}^p \frac{(x_j - \mu_{kj})^2}{\sigma_{kj}^2} + \log\pi_k$$

(The $-\tfrac{1}{2}\sum_j \log\sigma_{kj}^2$ term is included if class-specific variances are kept.)

This is just **QDA with $\Sigma_k$ restricted to be diagonal**.

## Insights & mental models

- **Naive Bayes = LDA/QDA with diagonal $\Sigma$.** That's the cleanest mental file for it. ISLR §4.5.1 makes this precise: with Gaussian marginals it's QDA-with-diagonal-$\Sigma_k$; if you further pool across classes, it's LDA-with-diagonal-$\Sigma$.
- **The independence assumption is generally false.** "Do we really believe the naive Bayes assumption that the $p$ covariates are independent within each class? In most settings, we do not." (ISLR §4.4.4) But the resulting bias is often offset by the dramatic variance reduction, bias-variance argument in the simplest form.
- **Why it's "naive":** assuming $p$ predictors are conditionally independent given the class is a strong, usually-wrong assumption. It's "idiotic" hence the alternate name.
- **Why it works anyway:** for *classification* (vs density estimation), what matters is which class wins the argmax, not whether the densities are accurately estimated. Even rough-and-wrong densities often rank classes correctly.
- **Mixed predictor types are easy.** Continuous $X_j$ → Gaussian or kernel-smoothed marginal. Categorical $X_j$ → multinomial. They factor cleanly because of the independence assumption.
- **Prof's slot for it:** "popular when $p$ is large", the standard case where LDA/QDA's covariance estimates blow up.

## Exam signals

> "Naive Bayes: large $p$. Skip the cross-covariance." - [[L09-classif-3]]

> "$\Sigma_k$ is assumed diagonal, and only the diagonal elements are estimated.", slide deck

The prof dedicated only one slide stretch to it (toward end of L09); no exam-flagging quote, but the **bias-variance argument** (large $p$, small $n$, simpler model wins) is the recurring theme he'd test.

## Pitfalls

- **Confusing naive Bayes with the Bayes classifier.** Different things. The Bayes classifier is the abstraction that uses the *true* posterior $\Pr(Y\mid X)$. Naive Bayes is a specific generative model with the conditional-independence assumption, used to *estimate* a posterior.
- **Forgetting that "naive" is a technical word.** Refers to the conditional-independence assumption, not to the classifier being simple.
- **Treating naive Bayes as a strict subset of LDA.** Strictly: with Gaussian marginals + pooled $\Sigma$, naive Bayes ⊂ LDA (LDA with diagonal $\Sigma$). With Gaussian marginals + class-specific $\Sigma_k$, naive Bayes ⊂ QDA. With non-Gaussian marginals, naive Bayes is its own thing.
- **Standardization concerns.** Same as for LDA/QDA, if predictors are on wildly different scales, the marginals' sds get badly estimated. Standardize.

## Scope vs ISLR

- **In scope:** The conditional-independence assumption, $f_k(x) = \prod_j f_{kj}(x_j)$, why it's used (large $p$, parameter-count win), bias-variance trade-off justification.
- **Look up in ISLR:** §4.4.4, pp. 156–158. The toy example ($p = 3$, $K = 2$) illustrates the multiplicative posterior. §4.5.1 shows the formal connection to LDA (eq. 4.34), useful for the "naive Bayes ⊂ LDA with diagonal Σ" insight.
- **Skip in ISLR:**
  - Detailed mixed-predictor naive-Bayes implementations (§4.4.4 final paragraphs), concept matters, mechanics don't.
  - Smoothing parameter / Laplace correction for zero-frequency categorical cells, never covered.

## Exercise instances

None, naive Bayes has no recommended-exercise or compulsory-exercise problem in module 4. The slide deck mentions it briefly; the prof's lecture covers it in maybe 2 minutes. Kept as a thin atom (per manifest note 5) because it's a named method on the slide curriculum and could plausibly appear as an MCQ.

## How it might appear on the exam

- **MCQ:** "Naive Bayes assumes which of the following?" → predictors are conditionally independent given the class.
- **Method-comparison T/F:** "Naive Bayes is preferred when $p$ is large" → true. "Naive Bayes assumes the predictors are unconditionally independent" → false (only conditionally, given class). "Naive Bayes is a special case of QDA" → true (if Gaussian marginals + class-specific variances).
- **Parameter-count question:** "How many parameters does Gaussian naive Bayes estimate for $K$ classes and $p$ predictors?" → $2pK + (K - 1)$, much fewer than LDA or QDA.
- **Bias-variance argument:** Why naive Bayes might out-perform QDA when $n$ is small relative to $p^2$.

## Related

- [[linear-discriminant-analysis]]: same Bayesian-discriminant machinery; LDA with diagonal $\Sigma$ ≈ Gaussian naive Bayes (with shared variances).
- [[quadratic-discriminant-analysis]]: naive Bayes is QDA with diagonal $\Sigma_k$.
- [[multivariate-normal]]: the multivariate Gaussian whose diagonal-restriction gives Gaussian naive Bayes.
- [[diagnostic-vs-sampling-paradigm]]: naive Bayes is on the sampling/generative side.
- [[bias-variance-tradeoff]]: the standard justification for using a more-restricted model when $p$ is large.
- [[curse-of-dimensionality]]: naive Bayes is one of the standard answers for "what to do in high $p$."
