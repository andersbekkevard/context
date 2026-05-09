---
concept: flexibility-overfitting-underfitting
module: 02-statlearn
lectures: [L03, L04, L17]
isl-ref: 2.2.1
exercises:
  - Exercise2.2 - flexible vs rigid methods, test error, overfit/underfit relation to bias-variance
  - Exercise2.5 - full polynomial-regression simulation (training MSE always falls, test MSE is U-shaped)
related: [bias-variance-tradeoff, reducible-vs-irreducible-error, parametric-vs-nonparametric, knn-classification, knn-regression, polynomial-regression, double-descent, regularization, cross-validation]
tags:
  - concept
  - module/02-statlearn
aliases:
  - overfitting
  - underfitting
  - U-shape
  - training vs test MSE
---

# Flexibility, overfitting, underfitting, and the train/test MSE U-shape

The prof's first big working theorem of the course: **training MSE always decreases** as you add flexibility (more flexible models contain less flexible ones, the new fit can never do worse on the data it saw). **Test MSE is U-shaped**, it drops at first as bias falls, then rises as variance / overfitting takes over. The polynomial-degree (or KNN-K) sweep is the canonical demo. The minimum of the U is the model you want; you find it via held-out data or [[cross-validation]] (M5).

## Definition (prof's framing)

> "Every model, if you make it more flexible, it will necessarily fit the data better." - [[L03-statlearn-2]]

That's training MSE. The test side:

> "We don't want to predict last week's stock price. We want to predict the stock price of next week." - [[L03-statlearn-2]]

Test MSE measures how the fit generalizes to new $(x_0, y_0)$, and unlike training MSE, it's U-shaped in flexibility. The minimum of the U is the **bias-variance sweet spot**.

> **Overfitting**: every training point sits on the curve, the fit gets wiggly, doesn't generalize. KNN with $K=1$ is the prototype.
> **Underfitting**: model too rigid / too few parameters. Real curve in the data, but you fit a line, missing structure.

## Notation & setup

- **Training MSE**: $\mathrm{MSE}_{\text{train}} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat f(x_i))^2$, evaluated on the same data the model was fit to.
- **Test MSE**: $\mathrm{MSE}_{\text{test}} = \frac{1}{n_{\text{test}}} \sum_j (y_{0j} - \hat f(x_{0j}))^2$, evaluated on independent $(x_0, y_0)$.
- **Flexibility knob** depends on the method: polynomial degree, $K$ in KNN (smaller $K$ = more flexible, note the inversion), spline df, tree depth, $\lambda^{-1}$ for ridge/lasso, hidden-layer width.

## Formula(s) to know cold

The U-shape *itself* is the takeaway, not a formula. It's the consequence of decomposing test MSE via the [[bias-variance-tradeoff]]:
$$\mathbb{E}[(y_0 - \hat f(x_0))^2] = \underbrace{\mathrm{Var}(\varepsilon)}_{\text{floor}} + \underbrace{(f(x_0) - E[\hat f(x_0)])^2}_{\text{bias}^2 \,\downarrow \text{ in flexibility}} + \underbrace{\mathrm{Var}(\hat f(x_0))}_{\text{variance} \,\uparrow \text{ in flexibility}}$$

Bias falls and variance rises with flexibility → sum bottoms out at some intermediate complexity → that's the U.

## Insights & mental models

**Why training MSE monotonically decreases**: each more-flexible model contains the previous as a special case. A degree-10 polynomial contains degree-2 (set the higher coefficients to zero). On training data you literally cannot do worse, only equal or better. ([[L03-statlearn-2]].)

**Why test MSE is U-shaped**: at low flexibility, $\hat f$ is too rigid to express the truth → high bias. At high flexibility, $\hat f$ chases noise in the specific training set → high variance, generalizes poorly. The polynomial sim ([[L03-statlearn-2]] / Exercise 2.5) makes this concrete: poly2 is the truth and wins; poly10/20 *contain* the truth but get pulled around by noise.

> "This does not happen, right? This is what people want to happen. This doesn't happen. We never get the right model … But it's good for the math because then everything works out." - [[L03-statlearn-2]]

I.e. the polynomial example is rigged so the truth is in the function class. In practice it never is. The U-shape persists anyway.

**KNN as the same story with the knob inverted**:

- $K = 1$: extremely wiggly decision boundary, "little islands of red", overfit. High variance.
- $K = 150$ (out of 200): boundary nearly straight, underfit. High bias.
- Optimal $K$ is intermediate, found via test/CV error.

> "How well you fit the data versus how well it generalizes is a common theme in the course." - [[L03-statlearn-2]]

This is the *exact same picture* as polynomial degree, just with $K$ as the flexibility knob (and $1/K$ if you want flexibility increasing left-to-right on the x-axis, ISLR Fig 2.17 plots it that way).

**Why training error doesn't catch overfitting**: a low training error can be a sign of overfitting that *increases* test error. Training error doesn't account for model complexity, that's what AIC/BIC/Cp try to fix (M5/M6, conceptually only, see [[aic-bic-conceptual]]) and what cross-validation fixes properly.

**The fix when you don't have a test set**: split your data, e.g. 80/20. The systematic version is [[cross-validation]] in M5. The polynomial simulation in Exercise 2.5 shows you can recover the U-shape using a held-out test set drawn from the same DGP.

**Sensitivity to which dataset you got** (the variance picture): KNN with small $K$ is very sensitive to *which* training set you happened to draw. Re-run the experiment, the boundary changes a lot. Large $K$ is much more stable across resamples. ([[L03-statlearn-2]].) This sensitivity *is* variance, it's exactly $\mathrm{Var}(\hat f(x_0))$ from the bias-variance decomposition.

### Connection to other modules

The U-shape is the spine of model selection across the whole course:
- **M5 ([[cross-validation]])**: how you find the U's minimum without a test set.
- **M6 ([[ridge-regression]], [[lasso]])**: how you keep a flexible model from blowing up its variance, flatten the right side of the U.
- **M8 (regression / classification trees, [[cost-complexity-pruning]])**: prune a deep tree to avoid the right side of the U.
- **M9 ([[boosting]] / [[gradient-boosting]])**: weak learners + many of them, with $\nu$ controlling how fast you walk along the U.
- **M11 ([[nn-regularization]])**: dropout, weight decay, early stopping all attack overfitting in NN.
- And: in extreme over-parameterization, the U keeps going down. That's [[double-descent]], the prof's favorite hobby-horse from L04.

The prof was explicit ([[L03-statlearn-2]]) that *flexible-but-restrained* is the modern strategy:

> "Something that is sort of underappreciated is that while we have flexible models, we also have good ways of making sure that the flexible models don't completely go crazy."

That's [[regularization]], and it's why he keeps resisting the "trade-off" framing of bias-variance. See the dedicated atom.

## Pitfalls

- **KNN K is inverted: small K = high flexibility.** Easy T/F trap. Use $1/K$ on the x-axis if you want flexibility to increase left-to-right.
- **Training error declining doesn't mean the model is good.** It just means the model has more parameters. Always check on held-out data.
- **The U exists for almost every flexibility knob.** Polynomial degree, K, df, $\lambda^{-1}$, tree depth, NN width, the same story with different axes. (The exception is the over-parameterized regime, see [[double-descent]].)
- **"Fits the truth on average" $\neq$ "good prediction on this dataset."** A high-flexibility model can have low *bias* (correct on average across resamples) but huge *variance* on any single fit. The U is that trade.
- **Overfitting is method-dependent.** Some models (with built-in regularization, or well-chosen flexibility) overfit much less than others ([[L03-statlearn-2]]: "Some models are *designed* so overfitting isn't such a problem; others have a strong tendency to overfit.").

## Scope vs ISLR

- **In scope:** the U-shape of test MSE, monotonic decrease of training MSE, what each side of the U represents (bias / variance), polynomial degree and KNN K as flexibility knobs, the train/test discrepancy as the diagnostic for overfitting.
- **Look up in ISLR:** §2.2.1 (Measuring the Quality of Fit), §2.2.2 (The Bias-Variance Trade-Off), Figures 2.9–2.12 (the U-shape and its bias/variance decomposition for three datasets), Figure 2.17 (KNN training/test error vs $1/K$). For [[double-descent]], ISLR §6.4 has the high-dimensional discussion but the prof's L04 simulation is the better source.
- **Skip in ISLR:** the *book-only* claim that "more flexible models always have higher variance" is the prof's quoted hobby-horse, true on average but regularization can flatten the variance curve. He'll grumble about this in M6.

## Exercise instances

- **Exercise 2.2**: given ISL Fig 2.9, identify which methods (flexible vs rigid) have the highest test error, and whether that's always the case; relate over-/underfitting to bias-variance. Pure conceptual T/F-style drill.
- **Exercise 2.5**: full polynomial-regression simulation. Generate $Y = X^2 + \varepsilon$ with $\varepsilon \sim N(0, 4)$, fit polynomials of degree 1-20, plot training MSE (always falls) and test MSE (U-shape), then decompose test MSE into bias², variance, irreducible. The canonical exercise that nails everything in this atom plus the [[bias-variance-tradeoff]] derivation.

## How it might appear on the exam

- **Direction-of-effect T/F.** "As polynomial degree increases, training MSE decreases" → **true** (always). "Test MSE decreases" → **false** (U-shaped). "Bias decreases" → **true** (typically). "Variance decreases" → **false** (typically rises).
- **Identify overfitting from a plot.** Given a training-MSE-low / test-MSE-high gap as a function of complexity, point to where overfitting starts. Or given two models' fit + test error, diagnose which is over- and which is underfit.
- **KNN K direction trap.** "Increasing K in KNN increases flexibility" → **false**, it decreases flexibility (smoother fit).
- **Procedural** (prof's preferred 2026 question style): given a CV plot or a candidate set of polynomial degrees with their test MSE, pick the model and justify with bias-variance language.
- **Method comparison.** "Two models give the same training MSE but very different test MSE, what's going on, and which would you trust?" → the higher-test-MSE one is overfit; trust the other (or a regularized version).

## Related

- [[bias-variance-tradeoff]]: the formal decomposition that *explains* why the U exists
- [[reducible-vs-irreducible-error]]: the floor under the U
- [[parametric-vs-nonparametric]]: flexibility is the axis underlying both
- [[knn-classification]] / [[knn-regression]]: K is the canonical nonparametric flexibility knob
- [[polynomial-regression]]: degree is the canonical parametric flexibility knob (and the classroom simulation playground)
- [[double-descent]]: what happens *past* the U if you go ridiculously over-parameterized
- [[regularization]]: the modern toolkit for keeping the right side of the U flat
- [[cross-validation]]: how you find the U's minimum without a test set
