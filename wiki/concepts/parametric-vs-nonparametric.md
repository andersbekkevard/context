---
concept: parametric-vs-nonparametric
module: 02-statlearn
lectures: [L03]
isl-ref: 2.1.2
exercises: []
related: [knn-classification, knn-regression, linear-regression, flexibility-overfitting-underfitting, bias-variance-tradeoff, curse-of-dimensionality]
tags:
  - concept
  - module/02-statlearn
aliases:
  - parametric methods
  - nonparametric methods
---

# Parametric vs nonparametric methods

The prof's framing for the first big methodological cut in the course: do you assume a functional form for $f$ and estimate a finite set of parameters, or do you let the data dictate the shape with effectively no parametric form? Linear regression vs KNN is the canonical contrast, flexibility traded against assumption-freedom.

## Definition (prof's framing)

> "[Parametric methods] select a form for $f$ … then estimate the parameters using a training set." - [[L03-statlearn-2]]

> "Non-parametric: the general idea is that there's no parameters … a non-parametric model tries to have a very loose form." - [[L03-statlearn-2]]

In practice "no parameters" is a slight overstatement: KNN still has the hyperparameter $K$. The prof's point is that $K$ is a **flexibility knob**, not a structural parameter of an assumed model. The assumed *form* of $f$ is what's missing.

## Notation & setup

- Parametric: assume $f(X) = g(X; \theta)$ for some known $g$, finite-dimensional $\theta \in \mathbb R^d$. Estimate $\hat\theta$ from training data, predict $\hat f(x_0) = g(x_0; \hat\theta)$.
- Canonical parametric form: linear model $f(X) = \beta_0 + \beta_1 X_1 + \dots + \beta_p X_p$; fit by least squares.
- Nonparametric: no fixed-dimensional $\theta$. KNN's "model" is the training data itself plus the rule "average / vote over the $K$ nearest neighbors." Smoothing splines and LOESS work the same way later.

## Insights & mental models

**Two-step parametric recipe** ([[L03-statlearn-2]]):

1. Select a form for $f$.
2. Estimate the parameters using a training set ("fit data").
3. (Implicit step 3: assume $\varepsilon = 0$ when reading $\hat Y$ off the fitted curve.)

**Nonparametric is essentially interpolation**:

> "It's really just kind of interpolating your data. So if you don't have data in some place then you're going to get a bad interpolation." - [[L03-statlearn-2]]

A parametric model can extrapolate using its assumed form (for better or worse). KNN cannot fill empty regions with anything sensible, it has no global structure to lean on.

**Even nonparametric methods have flexibility knobs.** $K$ in KNN behaves exactly like polynomial degree in parametric models: small $K$ → wiggly islands (high variance), large $K$ → over-smoothed flat decision (high bias). The prof drives this home with the K=1 vs K=150 contrast on the same data ([[L03-statlearn-2]]).

### The trade-offs (verbatim flavor)

**Parametric pros**: simple, interpretable, requires little data, computationally cheap. The prof downgrades the last one:

> "Would you really use a different method simply because it took your computer two seconds less to use when the other one is, you know, better in a way or makes fewer assumptions?" - [[L03-statlearn-2]]

**Parametric cons**: $f$ is constrained to a specific form ("can be worse than we think"); the assumed form generally won't match the truth → poor estimate; limited flexibility. Most dangerously:

> "It makes assumptions about what happens outside of your data which can often lead to very bad things." - [[L03-statlearn-2]]

That's the extrapolation point, a fitted line keeps going beyond the data range whether or not the truth does.

**Nonparametric pros**: flexible, no strong assumptions about $f$, naturally captures complicated boundaries.

**Nonparametric cons**: easy to overfit, needs *lots* of data (to cover the input space), no extrapolation, killed by the [[curse-of-dimensionality]] in high $p$.

## Inflexible vs flexible: where the methods sit

The prof maps the course's methods onto this axis ([[L03-statlearn-2]]):

- **Inflexible (parametric, structured)**: linear regression (M3), LDA (M4), subset selection / lasso (M6).
- **Flexible (often nonparametric)**: [[knn-classification]] / [[knn-regression]] (M4), smoothing splines (M7), bagging and boosting (M8/M9), neural networks (M11).

Then the seed for the rest of the course:

> "Something that is sort of underappreciated is that while we have flexible models, we also have good ways of making sure that the flexible models don't completely go crazy … we do have often good ways of restraining the flexible models." - [[L03-statlearn-2]]

This is foreshadowing of [[regularization]], ridge / lasso / shrinkage (M6), and the broader theme of *regularized flexible models*. It's the prof's main reason for resisting the "trade-off" framing of [[bias-variance-tradeoff]]: with regularization you can build a flexible model whose variance doesn't explode.

## Pitfalls

- **"Nonparametric" $\neq$ "no hyperparameters."** $K$ in KNN, $\lambda$ in smoothing splines, etc., still need to be picked, typically by [[cross-validation]].
- **Parametric extrapolation is a trap.** A linear model will happily predict at $x_0$ far outside the training range, with no warning that it's nonsense. Nonparametric methods at least fail loudly there (they have nothing to interpolate from).
- **Choice depends on your goal** (the [[prediction-vs-inference]] split): parametric usually wins when you need an interpretable coefficient story; nonparametric wins when you only care about prediction accuracy and have enough data.

## Scope vs ISLR

- **In scope:** the parametric-vs-nonparametric distinction, the two-step parametric recipe, the K-as-flexibility story for KNN, the prediction-vs-extrapolation contrast, the inflexible / flexible classification of course methods.
- **Look up in ISLR:** §2.1.2 (Parametric vs Non-Parametric Methods) and §2.1.3 (The Trade-Off Between Prediction Accuracy and Model Interpretability). Figures 2.4–2.6 (linear vs thin-plate spline on the Income data) are the canonical visual.
- **Skip in ISLR:** thin-plate spline mechanics, covered conceptually only, splines proper come back in M7.

## How it might appear on the exam

- **MC / true-false on which methods are parametric.** Linear regression, LDA, lasso = parametric. KNN, smoothing splines, GAMs, bagging, boosting, NNs = nonparametric or semi-parametric. Easy concept-check.
- **Direction-of-effect question.** "If $K$ in KNN decreases, does flexibility increase or decrease?" → increases. Same question type as polynomial degree.
- **Method-comparison interpretation.** Given output from two models on the same data, identify which is parametric / which is more flexible, and explain the bias-variance implications.
- **Extrapolation trap.** "Why is KNN bad at predicting outside the training range?", because it has no global form to extrapolate; it averages neighbors that don't exist out there.

## Related

- [[knn-classification]]: the canonical nonparametric example used to teach the contrast
- [[knn-regression]]: the same idea applied to continuous $Y$
- [[linear-regression]]: the canonical parametric example
- [[flexibility-overfitting-underfitting]]: the U-shape is the bias-variance consequence of changing the flexibility knob, parametric or not
- [[bias-variance-tradeoff]]: the formal decomposition behind "flexibility costs you variance"
- [[curse-of-dimensionality]]: the technical reason nonparametric methods break in high $p$
- [[regularization]]: the prof's preferred way to make a flexible model behave
