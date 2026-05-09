---
concept: linear-discriminant-analysis
module: 04-classif
lectures: [L08, L09]
isl-ref: 4.4.1-4.4.2
exercises:
  - Exercise4.2a - pooled covariance estimator across two groups
  - Exercise4.2b - write the LDA classification rule for new observation
  - Exercise4.2c - classify a length=214/diagonal=140.4 bank note via LDA
  - Exercise4.6e - repeat lag-based classification with LDA on Weekly
  - CE1 problem 3d - interpret π_k, μ_k, Σ, f_k(x) in LDA
  - CE1 problem 3e - derive δ_k(x), solve for class boundary, plot
  - CE1 problem 3f - perform LDA in R, confusion matrix, sensitivity/specificity
related: [quadratic-discriminant-analysis, discriminant-score-and-decision-boundary, naive-bayes, logistic-regression, multivariate-normal, diagnostic-vs-sampling-paradigm, dimensionality-reduction, bias-variance-tradeoff]
tags:
  - concept
  - module/04-classif
aliases:
  - LDA
---

# Linear discriminant analysis (LDA)

The prof's canonical generative classifier: model the **class-conditional density** $f_k(x) = \Pr(X \mid Y = k)$ as Gaussian with class-specific means $\mu_k$ and a **pooled** covariance $\Sigma$, plus a class prior $\pi_k$, then flip with Bayes. The resulting [[discriminant-score-and-decision-boundary|discriminant score]] is **linear in $x$**, that's the "L." Decision boundary derivation is exam-flagged twice.

## Definition (prof's framing)

> "We're not trying to model $P(Y \mid X)$ directly. We're trying to get the other distributions to then use Bayes to get the classification." - [[L09-classif-3]]

> "We're flipping it around. So now, instead of modeling this probability of Y given X directly, we want to make a model of what is the prior distribution of Y, like how likely is this class A or class B, and the probability of X given Y." - [[L09-classif-3]]

The two assumptions that make it work:

1. **Within each class, $X \sim \mathcal{N}(\mu_k, \Sigma)$**, Gaussian class-conditional.
2. **The covariance $\Sigma$ is shared across classes** ("pooled"). This is the **L** in LDA. Relax it → [[quadratic-discriminant-analysis|QDA]].

Plus a class prior $\pi_k = \Pr(Y = k)$ for each class.

## Notation & setup

- $K$ classes labeled $k = 1, \ldots, K$; binary case: $K = 2$.
- $\pi_k = \Pr(Y = k)$, with $\sum_k \pi_k = 1$.
- $\mu_k \in \mathbb{R}^p$, class-$k$ mean vector.
- $\Sigma \in \mathbb{R}^{p \times p}$, **shared** covariance matrix (positive definite).
- $f_k(x)$, multivariate normal density with $(\mu_k, \Sigma)$.

For $p = 1$, replace $\Sigma$ with shared scalar $\sigma^2$.

## Formula(s) to know cold

**Bayes' rule (the engine):**

$$\Pr(Y = k \mid X = x) = \frac{\pi_k f_k(x)}{\sum_\ell \pi_\ell f_\ell(x)}$$

**1D Gaussian discriminant score** (linear in $x$, drop terms not depending on $k$, take logs):

$$\delta_k(x) = x \cdot \frac{\mu_k}{\sigma^2} - \frac{\mu_k^2}{2\sigma^2} + \log \pi_k$$

**Multivariate Gaussian discriminant score:**

$$\delta_k(x) = x^\top \Sigma^{-1} \mu_k - \tfrac{1}{2} \mu_k^\top \Sigma^{-1} \mu_k + \log \pi_k$$

Classify by $\hat y = \arg\max_k \delta_k(x)$.

**Posterior recovery (softmax over discriminants):**

$$\hat\Pr(Y = k \mid X = x) = \frac{e^{\delta_k(x)}}{\sum_\ell e^{\delta_\ell(x)}}$$

### Parameter estimators (plug-in MLEs)

- $\hat\pi_k = n_k / n$ (class frequency).
- $\hat\mu_k = \dfrac{1}{n_k} \sum_{i: y_i = k} x_i$.
- **Pooled covariance**: the formula to know:

$$\hat\Sigma = \sum_{k=1}^K \frac{n_k - 1}{n - K} \cdot \hat\Sigma_k$$

where $\hat\Sigma_k = \dfrac{1}{n_k - 1} \sum_{i: y_i = k}(x_i - \hat\mu_k)(x_i - \hat\mu_k)^\top$.

In 1D: $\hat\sigma^2 = \dfrac{1}{n - K} \sum_k \sum_{i: y_i = k}(x_i - \hat\mu_k)^2$.

## Why the discriminant is linear

Take the log of $\pi_k f_k(x)$ with $f_k$ multivariate Gaussian. The exponent has $-\tfrac{1}{2}(x - \mu_k)^\top \Sigma^{-1}(x - \mu_k)$, which expands as

$$-\tfrac{1}{2} x^\top \Sigma^{-1} x + x^\top \Sigma^{-1} \mu_k - \tfrac{1}{2} \mu_k^\top \Sigma^{-1} \mu_k$$

The $x^\top \Sigma^{-1} x$ term has **no $k$** (because $\Sigma$ is shared) → it cancels in $\arg\max_k$ → drop. What's left is **linear in $x$**.

> "The $x^2$ term has no $k$, gone… [in QDA we don't pool], so the coefficient becomes $k$-dependent, the $x^2$ term survives, and the discriminant becomes quadratic." - [[L09-classif-3]]

## Insights & mental models

- **The decision boundary is a *consequence* of the assumed model, not a fitted parameter.** Contrast with logistic regression, where the boundary's slope *is* a parameter. In LDA, boundaries fall out from equating $\delta_k = \delta_\ell$. - [[L09-classif-3]]
- **Priors literally move the boundary.** Increasing $\pi_k$ shifts the boundary away from class $k$ (more things classified as $k$). The product-curve picture is the prof's go-to visualization.
- **Equal priors, 2 classes, equal $\sigma$, 1D**: boundary at $(\mu_1 + \mu_2)/2$. Solve $\delta_1(x) = \delta_2(x)$ to verify.
- **LDA = dimensionality reduction.** $K$ classes → $K - 1$ discriminant scores. "Mapping it down to a dimension specifically to separate out these categories." - [[L09-classif-3]]. Useful in high $p$ where KNN is cursed.
- **In-sample LDA can beat the Bayes-optimal classifier on training data.** Sound paradoxical; it's just overfitting. The Bayes classifier uses true parameters; the fitted LDA chases noise. - [[L09-classif-3]]
- **LDA ↔ logistic regression** are very close. Same linear log-odds form (slide deck note: $\log(p_1(x) / (1 - p_1(x))) = c_0 + c_1 x$ for both). Different parameter-estimation routes, Gaussian-plug-in vs MLE.

## Worked 2D example, the prof's recurring template

Setup: $\mu_A = (1, 1)$, $\mu_B = (3, 3)$, $\Sigma = 2I$, equal priors. Equate $\delta_A(x) = \delta_B(x)$:

$$x^\top \Sigma^{-1}(\mu_A - \mu_B) - \tfrac{1}{2}\mu_A^\top \Sigma^{-1} \mu_A + \tfrac{1}{2} \mu_B^\top \Sigma^{-1} \mu_B + \log\pi_A - \log\pi_B = 0$$

With $\Sigma^{-1} = \tfrac{1}{2}I$, this collapses to $-x_1 - x_2 + 4 = 0$, i.e. **$x_2 = 4 - x_1$**.

> "I didn't ask for a line. I just equated the two things and then I solved for them and then it became a line. I didn't tell the math to give me a line." - [[L09-classif-3]]

The prof did the algebra live, slipped on a cancellation, came back from break with the corrected version. Worth knowing because **this is the exam pattern** (next section).

## Exam signals

> "And that's often an exam question. Or that would be a typical exam question. So you would, given an LDA setting and here are the values for the parameters, you have the pies, you have the mu's, and you'd have a value for the standard deviation, and then you would solve for where is the decision point." - [[L09-classif-3]]

> "This would be another kind of question that you could ask on an exam, like find the equation for the decision boundary between these two categories. Then you solve for the equation for the line or the plane or whatever it happens to be depending on the dimension of the X's." - [[L09-classif-3]]

> "That would be another question one could ask if I was so inspired, show that this leads to this thing." - [[L09-classif-3]] (re: deriving $\delta_k$ from the Gaussian)

The prof flagged **decision-boundary derivation** twice, in two different lectures' worth of material, see [[discriminant-score-and-decision-boundary]] for the standalone procedural atom.

## Pitfalls

- **Forgetting that $\Sigma$ is pooled.** If you use class-specific $\Sigma_k$, you've done QDA, not LDA. Boundary becomes quadratic.
- **The $\Sigma^{-1}$ matrix gets confused with $\Sigma$ in the $\delta_k$ formula.** It's $\Sigma^{-1}$, the precision matrix. Don't drop the inverse.
- **Wrong direction of "decision boundary moves with prior."** Increasing $\pi_k$ moves the boundary **away** from class $k$ (more area classified as $k$). Easy to flip in a hurry.
- **Treating in-sample confusion matrix as the truth.** "You're not looking at how well you classified out of sample. This is in-sample, on the data you actually train on. So you're going to do really well because you have all the noise in the data." - [[L09-classif-3]]
- **Gaussian assumption violated.** "Maybe it's not a good idea to pretend that the X's are well modeled by a Gaussian, that's a good way to break a model." - [[L09-classif-3]]
- **Pooling assumption violated** when class covariances genuinely differ, bias goes up.

## Scope vs ISLP

- **In scope:** Bayes' rule for class probabilities, Gaussian class-conditionals, pooled covariance, derivation of $\delta_k(x)$ (1D and multivariate), decision-boundary derivation, parameter estimation (plug-in MLEs), softmax recovery of posterior, comparison with logistic regression, LDA-as-dimensionality-reduction.
- **Look up in ISLP:** §4.4.1 (LDA for $p = 1$), §4.4.2 (LDA for $p > 1$), pp. 145–155. Equations (4.18) and (4.24) are the canonical $\delta_k$ formulas; Figure 4.6 is the 3-class decision-boundary picture.
- **Skip in ISLP:**
  - **Fisher's discriminant derivation** (within-class vs between-class variance ratio, eigenvectors of $\Sigma^{-1} B$), slide deck section is marked "Optional" and the prof never lectured on it.
  - **Multinomial-logistic-vs-LDA detailed mapping**: prof skipped multinomial logistic.

## Exercise instances

- **Exercise4.2a**: write the pooled covariance estimator across two groups (genuine vs fake bank notes); plug in $\hat\Sigma_G, \hat\Sigma_F$ and equal sample sizes.
- **Exercise4.2b**: state the LDA assumptions; write the classification rule for a new observation (need to assume $\pi_G = \pi_F$, normality, equal $\Sigma$).
- **Exercise4.2c**: classify a bank note with length 214 / diagonal 140.4 using LDA. R-friendly matrix calc.
- **Exercise4.6e**: `lda(Direction ~ Lag2)` on the `Weekly` data; held-out confusion matrix.
- **CE1 problem 3d**: explain $\pi_k$ (prior), $\mu_k$ (class mean vector), $\Sigma$ (pooled covariance), $f_k(x)$ (multivariate Gaussian density) in words.
- **CE1 problem 3e**: derive $\delta_k(x)$ from Bayes' rule, solve $\delta_0(x) = \delta_1(x)$ for the boundary in the form $ax_1 + bx_2 + c = 0$, plot it.
- **CE1 problem 3f**: `lda()` in R, confusion matrix, sensitivity/specificity on the tennis test set.

## How it might appear on the exam

- **Decision-boundary derivation (the prof's flagged pattern):** Given $\pi_k$, $\mu_k$, $\Sigma$ (or $\sigma$ in 1D), solve $\delta_1 = \delta_2$ for $x$. 1D → a point; 2D → a line. The prof said this twice.
- **Discriminant-score derivation:** Show how $\delta_k$ comes from $\log(\pi_k f_k(x))$ by dropping $k$-independent terms. Show *why* the $x^2$ drops out (the $\Sigma$ is pooled).
- **Pooled-covariance computation:** Given per-class $\hat\Sigma_k$ and $n_k$'s, compute $\hat\Sigma$.
- **Output interpretation:** Given an `lda()` output (means, prior, scaling), classify a new observation; or read off whether a sample is closer to class A or B in the discriminant space.
- **Method comparison:** "When would you prefer LDA to logistic regression?" → Gaussian holds, well-separated classes, small $n$, multi-class. Or "to QDA?" → small $n$, equal-covariance assumption defensible (bias-variance argument).
- **Confusion-matrix companion:** sensitivity, specificity from an LDA confusion matrix.

## Related

- [[discriminant-score-and-decision-boundary]]: the standalone procedural atom for deriving $\delta_k$ and solving for the boundary.
- [[quadratic-discriminant-analysis]]: drop the pooled-$\Sigma$ assumption, get quadratic boundaries.
- [[naive-bayes]]: assume $\Sigma$ is diagonal (predictors conditionally independent given class).
- [[logistic-regression]]: same linear log-odds form, different fitting route.
- [[multivariate-normal]]: the class-conditional density assumption.
- [[diagnostic-vs-sampling-paradigm]]: LDA is the canonical *sampling* method.
- [[dimensionality-reduction]]: LDA as $K-1$-D projection.
- [[bias-variance-tradeoff]]: pooling reduces variance (fewer parameters) at the cost of bias if $\Sigma_k$ differ.
- [[confusion-matrix]], [[sensitivity-specificity]], [[roc-auc]]: performance metrics.
