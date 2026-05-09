---
concept: knn-classification
module: 02-statlearn
lectures: [L03, L07, L08]
isl-ref: 2.2.3
exercises:
  - Exercise4.1a - compute Euclidean distances by hand from each of 7 training points to the test point (1,2)
  - Exercise4.1b - predict class for K=1, K=4, K=7; explain why K=7 is bad
  - Exercise4.1c - relate the best K to whether the Bayes decision boundary is highly nonlinear
  - Exercise4.6g - fit KNN on Weekly stock data with K=1, build confusion matrix
  - Exercise4.6h - sweep K=1..30 on Weekly, pick best K from error curve, report confusion matrix at best K
  - Exercise4.6j - compute KNN probabilities for ROC / AUC alongside glm/lda/qda
related: [knn-regression, parametric-vs-nonparametric, flexibility-overfitting-underfitting, bias-variance-tradeoff, classification-setup, curse-of-dimensionality, diagnostic-vs-sampling-paradigm, logistic-regression, linear-discriminant-analysis, cross-validation]
tags:
  - concept
  - module/02-statlearn
aliases:
  - KNN
  - K-nearest neighbors
  - K-nearest neighbours
---

# K-nearest neighbors (classification)

The course's running [[parametric-vs-nonparametric|nonparametric]] classifier and the prof's go-to demo for the [[bias-variance-tradeoff]]. Take a new point $x_0$, find its $K$ nearest training points by Euclidean distance, take a majority vote (or class-proportion estimate) over their labels. $K$ is the flexibility knob: $K = 1$ wiggly islands (overfit, high variance), large $K$ over-smoothed (underfit, high bias). Killed by the [[curse-of-dimensionality]]; the prof's "statisticians don't really love KNN" classifier.

## Definition (prof's framing)

Introduced in module 2 as the canonical nonparametric example, then formalized in module 4:

> "The general idea is that there's no parameters … a non-parametric model tries to have a very loose form" - [[L03-statlearn-2]]

> "To classify a new point, look at its K nearest training points (Euclidean distance) and take a majority vote. The class probability is just the fraction of the K neighbours of each class." - [[L08-classif-2]]

Formally, for a test point $x_0$ and a positive integer $K$, let $\mathcal N_0$ be the index set of the $K$ training points closest to $x_0$ in Euclidean distance. Then:

$$\hat P(Y = j \mid X = x_0) = \frac{1}{K} \sum_{i \in \mathcal N_0} I(y_i = j)$$

Predict the class with the largest $\hat P$.

## Notation & setup

- Training data $(x_1, y_1), \dots, (x_n, y_n)$ with $x_i \in \mathbb R^p$, $y_i \in \{1, \dots, K_{\text{classes}}\}$ (note: $K$ in KNN is **number of neighbors**, not number of classes, flagged in [[L07-classif-1]]).
- Distance: Euclidean by default, $d(x_0, x_i) = \|x_0 - x_i\|_2$. Other metrics exist; not on this exam.
- Use **odd $K$** for binary classification to avoid ties - [[L03-statlearn-2]].
- Probability output: the *fraction* of neighbors of each class. This is the score you'd threshold for ROC / AUC (Exercise 4.6j).

## Formula(s) to know cold

$$\boxed{\;\hat P(Y = j \mid X = x_0) = \frac{1}{K} \sum_{i \in \mathcal N_0} I(y_i = j); \qquad \hat y(x_0) = \arg\max_j \hat P(Y = j \mid X = x_0)\;}$$

Hand-computation skill the exercises drill: for a small training set, compute Euclidean distances $d(x_0, x_i) = \sqrt{(x_{i1} - x_{01})^2 + (x_{i2} - x_{02})^2}$ to every training point, sort, take the top $K$, vote. Exercise 4.1 is the canonical pen-and-paper drill.

## Insights & mental models

**$K$ is exactly a flexibility knob** ([[L03-statlearn-2]] and [[L07-classif-1]]):

- $K = 1$: each training point is its own little decision island. Boundary jumps at every point. **High variance**, low bias. "Little islands of red, which probably doesn't make any sense."
- $K = $ large (e.g. K = 150 out of 200): boundary becomes nearly straight. **High bias**, low variance. In the limit $K = n$: KNN returns the majority class everywhere.
- Optimal $K$ is intermediate, found by [[cross-validation]] (Exercise 4.6h: sweep $K = 1, \dots, 30$ and pick the elbow / minimum on a CV / test-error curve).

The boundary-shape interpretation:

> "If the Bayes decision boundary is highly non-linear, would we expect the best $K$ to be large or small?", small., Exercise 4.1c

Small $K$ tracks a wiggly Bayes boundary; large $K$ smooths it away.

**KNN is the diagnostic-paradigm classifier**: it estimates $P(Y \mid X)$ directly, by an empirical count over neighbors. Same paradigm as logistic regression. Contrast with the **sampling / generative paradigm** ([[diagnostic-vs-sampling-paradigm]]) of LDA / QDA / Naive Bayes, which model class-conditional densities $f_k(x)$ and flip via Bayes' rule. ([[L07-classif-1]], [[L09-classif-3]].)

**Decision regions can be wildly complicated**, even with a trivial model:

> "KNN can produce extremely complicated decision regions, including disconnected 'islands', even though the model itself is trivial." - [[L08-classif-2]]

That's the price of nonparametric flexibility, and the reason it sometimes beats LDA / QDA on weird boundaries. It's also why interpretability is essentially zero, there's no per-feature coefficient to read off.

**The killer pitfall**, [[curse-of-dimensionality]] ([[L08-classif-2]]):

> "The dimensionality is a curse, and in this case, it's a curse for K-nearest neighbours … makes it useless."

In high $p$, all pairwise distances become nearly equal, so "nearest" stops meaning anything. Simulate it: sample uniform points in $d$ dimensions, plot the distribution of pairwise distances as $d$ grows, the spread collapses. Fix is dimensionality reduction (e.g. PCA), but it's "a bit hacky still."

**Why statisticians don't love KNN** ([[L08-classif-2]]): it has no interpretable parameters, no closed-form theory, no per-feature inference, no extrapolation. It's a "lookup-table" classifier. Useful as a baseline and as a teaching device for bias-variance, but rarely the model anyone would deploy.

### Practical recipe for picking K

The exercise pattern (Exercise 4.6h):

1. For $K = 1, 2, \dots, K_{\max}$, fit KNN on training data.
2. Predict on validation / test set.
3. Compute misclassification rate at each $K$.
4. Plot error vs $K$, pick the $K$ at the minimum (or one-SE rule, see [[one-standard-error-rule]]).

In the 2026 exam form, you wouldn't write the R code; you'd interpret the resulting curve.

### Standardization

Because KNN uses Euclidean distance, predictors on different scales bias the metric: a feature in millimeters dominates a feature in years just because of unit. **Standardize predictors first** (z-score) before fitting KNN. The prof's [[standardization]] atom flags KNN as a method that is *not* scale-invariant; this is mandatory preprocessing in practice. (Not heavily emphasized in M2, but it's a real exam-trap-shaped fact.)

## Pitfalls

- **Tied votes**: use odd $K$ for binary; for multi-class, the standard library breaks ties at random (warned in Exercise 4.6g R-hint).
- **Curse of dimensionality**: KNN degrades fast as $p$ grows; not the right tool when $p$ is big.
- **Small K = high flexibility** (the inverted-knob trap): large $K$ → smoother → less flexible → more bias. Easy T/F mistake.
- **No standardization**: distances dominated by large-scale features; results garbage. Standardize first.
- **No extrapolation**: at a $x_0$ far from any training point, KNN's "nearest neighbors" are still its nearest neighbors, but they're far. Predictions are essentially the majority class of distant points; not meaningful.
- **K = 7 with 7 training points and $K=7$** (Exercise 4.1b): you're using *every* training point, the prediction is just the global majority class regardless of $x_0$. Pure underfit. The exam-trap framing of "why is K=7 bad?" in that exercise.

## Scope vs ISLP

- **In scope:** the algorithm, $K$ as flexibility knob, hand-computation of distances and votes, contrast with the parametric / generative paradigms, curse of dimensionality, picking $K$ by CV.
- **Look up in ISLP:** §2.2.3 (Bayes / KNN setup, Figs 2.14–2.17 for the geometry and the U-shape in $1/K$); §4.7.6 (KNN as classifier alongside logistic / LDA / QDA in the comparison setting).
- **Skip in ISLP:** weighted KNN ([[scope|out of scope]]: name-checked only), distance metrics other than Euclidean (mentioned in M10 [[distance-metrics]] but not exam-tested for KNN).

## Exercise instances

- **Exercise 4.1a**: compute Euclidean distances from a 7-point training set to test point $(1,2)$ by hand. Pure pen-and-paper drill.
- **Exercise 4.1b**: predict class for $K = 1, 4, 7$ from those distances. Explain why $K=7$ is bad (uses all training points → just the majority class everywhere).
- **Exercise 4.1c**: relate optimal $K$ to whether the Bayes decision boundary is highly nonlinear (small $K$ for wiggly boundary).
- **Exercise 4.6g**: fit KNN with $K = 1$ on Weekly stock data via `knn()` from the `class` library; produce confusion matrix on test set.
- **Exercise 4.6h**: sweep $K = 1, \dots, 30$, plot test misclassification, pick best $K$, report confusion matrix at that $K$.
- **Exercise 4.6j**: extract KNN probabilities (via `attributes(model)$prob`) and use them in ROC / AUC alongside glm, LDA, QDA. The prob extraction is library-specific R hassle, not exam material, but the *idea* (KNN gives a probability estimate, not just a class label) is.

## How it might appear on the exam

- **Hand calculation** (the 2026 procedural format): given a small training set and a test point, compute distances, identify the $K$ nearest, vote. The 2024 / 2025 exams have variants of Exercise 4.1.
- **Direction-of-effect T/F.** "Increasing $K$ increases the flexibility of KNN" → false. "Small $K$ has low bias" → true. "Large $K$ has low variance" → true.
- **Boundary-and-K matching.** Given two KNN decision-boundary plots (one wiggly, one smooth), match them to "K = 1" and "K = 50."
- **Method comparison.** "When would you prefer KNN to LDA?" → highly nonlinear Bayes boundary, low $p$, lots of data, no need for interpretation. (Or vice versa, prefer LDA when assumptions hold and you want coefficients to interpret.)
- **Concept recall.** "What is the curse of dimensionality and how does it affect KNN?", pairwise distances become nearly equal in high $p$, neighbors stop being meaningfully near, KNN's mechanism breaks.

## Related

- [[knn-regression]]: the same algorithm with $\hat f(x_0) = \frac{1}{K}\sum_{\mathcal N_0} y_i$ instead of a vote
- [[parametric-vs-nonparametric]]: the methodological frame KNN sits inside
- [[flexibility-overfitting-underfitting]]: $K$ is the flexibility knob; KNN provides the U-shape demo for classification
- [[bias-variance-tradeoff]]: small $K$ → high variance / low bias; large $K$ the reverse
- [[classification-setup]]: KNN is one estimator of $P(Y \mid X)$; the Bayes classifier is the gold standard it tries to approximate
- [[curse-of-dimensionality]]: KNN's headline failure mode in high $p$
- [[diagnostic-vs-sampling-paradigm]]: KNN sits in the diagnostic camp alongside logistic regression
- [[logistic-regression]], [[linear-discriminant-analysis]], [[quadratic-discriminant-analysis]]: the classifiers KNN is benchmarked against in M4
- [[cross-validation]]: how you'd pick $K$ in practice
- [[standardization]]: required preprocessing for KNN
