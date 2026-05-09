---
concept: classification-setup
module: 04-classif
lectures: [L07, L08, L09]
isl-ref: 4.1-4.2
exercises: []
related: [logistic-regression, knn-classification, linear-discriminant-analysis, diagnostic-vs-sampling-paradigm, bias-variance-tradeoff]
tags:
  - concept
  - module/04-classif
aliases:
  - Bayes classifier
  - Bayes error rate
---

# Classification setup, Bayes classifier, and Bayes error rate

The conceptual frame the prof opens module 4 with: response $Y$ is now categorical, we estimate $\Pr(Y=k\mid X)$, and assign the most likely class. The **Bayes classifier** is provably optimal , its irreducible error floor (the **Bayes error rate**) is the classification analogue of $\sigma^2$ from the regression bias-variance story.

## Definition (prof's framing)

> "Build probabilities of a new sample belonging to each of the different classes" - [[L07-classif-1]]

The training data is $\{(x_1, y_1), \ldots, (x_n, y_n)\}$ with $y_i \in \mathcal{C} = \{c_1, \ldots, c_K\}$. We build a classifier $f: \mathbb{R}^p \to \mathcal{C}$, ideally also returning a probability per class.

> "It ends up being like a true/false kind of setting and you want to make a decision. Yes, no. And the question is, how do you draw this line? Should it be a line? Should it be a curve? … Should it be something very squiggly?" - [[L08-classif-2]]

## Notation & setup

- $Y \in \{1, 2, \ldots, K\}$: categorical response, $K$ classes (binary: $K = 2$).
- $p_k(x) = \Pr(Y = k \mid X = x)$: posterior probability of class $k$ given the covariates.
- **0/1 loss** for classifier $\hat f$: $\mathbf{1}(y \neq \hat y)$ (1 on mismatch, 0 on match).

### Why not just OLS on a 0/1 response?

The prof gives two reasons:

1. Nothing constrains $\hat y \in [0, 1]$ , predictions can leave the probability scale.
2. **Class imbalance breaks it.** If one class dominates, the OLS line may "never pass the 0.5 threshold" , the classifier predicts the majority class everywhere unless you extrapolate way past the data range.

For multi-class, OLS on the codes 1, 2, 3 imposes an artificial ordering: "It's not like if you have two times the amount of stroke you suddenly get a drug overdose." - [[L07-classif-1]]

> "It's not super nice that it's not bounded between zero and one, as probability should be." - [[L07-classif-1]]

## Formula(s) to know cold

**Training / test error rate (0/1 loss):**

$$\text{err} = \frac{1}{n}\sum_{i=1}^n \mathbf{1}(y_i \neq \hat y_i)$$

**Bayes classifier:**

$$\hat y(x_0) = \arg\max_{k} \Pr(Y = k \mid X = x_0)$$

For binary, that's $\hat y = 1 \iff \Pr(Y=1 \mid X = x_0) > 0.5$.

**Bayes error rate (over the population):**

$$1 - E\left[\max_k \Pr(Y = k \mid X)\right]$$

## Insights & mental models

- **Bayes classifier ↔ irreducible error.** "Best case performance… directly analogous to irreducible error" - [[L07-classif-1]]. Even with the *true* posterior, you still misclassify when classes overlap , that's the noise floor of the problem.
- **Optimal but unattainable.** "It's optimal if you're right, but you're probably wrong." - [[L08-classif-2]]. Real procedures estimate $\Pr(Y \mid X)$, and the estimates are "probably bullshit." The Bayes classifier remains a useful theoretical benchmark.
- **The Bayes-optimal classifier can do *worse* on training data than a fitted LDA.** Counter-intuitive, but it's because the fitted classifier chases the training noise: in-sample comparisons flatter the model. - [[L09-classif-3]]
- **The Bayes decision boundary** is the locus where two posteriors cross , for binary, where $p(x) = 0.5$.

## Three methods covered in module 4

The prof's roster:

- [[logistic-regression]]: binary case, models $\Pr(Y\mid X)$ directly.
- [[knn-classification]]: non-parametric, any $K$ classes.
- [[linear-discriminant-analysis]] / [[quadratic-discriminant-analysis]] , generative, Gaussian class densities.

Plus [[naive-bayes]] as a variant of LDA/QDA when $p$ is large.

## Exam signals

> "The Bayes error rate is comparable to the *irreducible error* in the regression setting." - [[L07-classif-1]]

> "We usually don't know the true conditional distribution $\Pr(Y \mid X)$ for real data." - [[L07-classif-1]]

> "It's optimal if you're right, but you're probably wrong." - [[L08-classif-2]]

## Pitfalls

- **Don't use OLS for binary classification with imbalanced classes.** Predictions can stick at zero forever.
- **Multi-class OLS on numeric labels imposes an ordering**: categorical means dummy code, not 1/2/3.
- **Confusion matrices on training data are misleading.** "In-sample on the data you actually train on. So you're going to do really well because you have all the noise in the data. That doesn't mean you're going to do well out of sample." - [[L09-classif-3]]

## Scope vs ISLR

- **In scope:** Categorical response, 0/1 loss, training vs test misclassification rate, Bayes classifier definition + optimality, Bayes error rate as the irreducible-error analogue.
- **Look up in ISLR:** §4.1, §4.2, §4.4 (Bayes classifier definition); §2.2.3 for the original Bayes-classifier presentation.
- **Skip in ISLR (book-only, prof excluded):** Multi-class logistic regression beyond the high-level "use LDA instead" comment , [[L07-classif-1]] / slide deck. Probit / complementary-log-log link functions , [[L07-classif-1]]: "outside the scope of this course."

## Exercise instances

The classification setup itself (Bayes classifier, error rate, paradigm framing) appears as scaffolding inside almost every module-4 exercise , no exercise tests *just* the setup in isolation, so no direct exercise instances. See [[logistic-regression]], [[linear-discriminant-analysis]], etc., for the per-method exercise drill.

## How it might appear on the exam

- **MC/T-F:** Definition of Bayes error rate; relation to irreducible error; whether the Bayes classifier minimizes test error (yes) or training error (no , fitted classifiers do).
- **Conceptual:** "Why is the Bayes classifier provably optimal but not practically usable?"
- **Output interpretation:** Given a fitted $\hat p(x)$ table or curve, identify the classification threshold and the decision boundary.
- **Trap:** Confusing the Bayes classifier with naive Bayes (different things , one is the optimal-classifier abstraction, the other is a generative model with the conditional-independence assumption).

## Related

- [[diagnostic-vs-sampling-paradigm]]: the conceptual divide between modeling $\Pr(Y\mid X)$ directly vs via Bayes' theorem.
- [[logistic-regression]], [[knn-classification]], [[linear-discriminant-analysis]], [[quadratic-discriminant-analysis]], [[naive-bayes]] , the five method-specific atoms.
- [[confusion-matrix]]: how to read mistakes from a classifier output.
- [[sensitivity-specificity]], [[roc-auc]] , refined performance metrics for binary problems.
- [[bias-variance-tradeoff]]: same story for classification (under 0/1 loss), tunes K in KNN, drives LDA-vs-QDA preference.
- [[multivariate-normal]]: class-conditional density assumption for LDA/QDA.
