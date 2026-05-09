---
lecture: 3
date: 2026-01-13
module: 02-statlearn
title: Statistical Learning 2
slides: modules/2StatLearn/2StatLearn.1.md
topics:
  - supervised-learning
  - reducible-vs-irreducible-error
  - prediction-vs-inference
  - parametric-vs-nonparametric
  - linear-regression
  - knn-classification
  - flexibility
  - overfitting
  - underfitting
  - training-vs-test-mse
  - bias-variance-tradeoff
  - regularization
tags:
  - lecture
  - module/02-statlearn
aliases:
  - Lecture 3
---

# L03 — Statistical Learning 2

The prof recaps the supervised setup Y = f(X) + ε, derives the [[reducible-vs-irreducible-error]] split, then walks parametric vs non-parametric estimation (linear regression vs [[knn-classification]]). The back half is the toy polynomial-regression simulation that motivates the [[training-vs-test-mse]] U-shape, and a full board derivation of the [[bias-variance-tradeoff]] — which he insists on calling a **decomposition**, not a trade-off, and flags as very likely to appear on the exam.

## Key takeaways

- Y = f(X) + ε with ε mean-zero and independent of X. ε represents "stuff that has nothing to do with the thing you're trying to model" — measurement noise and unobserved fluctuations.
- Two reasons to estimate f: **prediction** (don't care about the shape, just about Ŷ) vs **inference** (care about the form). Different goals → different models. Finance vs scientists is the standard contrast.
- Expected squared error decomposes into a **reducible** and an **irreducible** part. The cross term vanishes because E[ε] = 0.
- Parametric (linear regression) trades flexibility for assumptions; non-parametric ([[knn-classification]]) trades assumption-freedom for needing lots of data.
- For polynomial regression: train MSE always decreases with degree; test MSE is **U-shaped**. Minimum (in this rigged simulation) is at the true degree, poly2.
- [[bias-variance-tradeoff]] decomposition: E[(y₀ − f̂(x₀))²] = Var(ε) + Var(f̂(x₀)) + (f(x₀) − E[f̂(x₀)])². Three terms: irreducible + variance + squared bias. Derive it by hand.
- The prof keeps pushing back on calling it a "trade-off" — modern methods (e.g. regularized fits) can reduce variance *without* paying full bias cost. The decomposition is exact; the trade-off framing is misleading.

> [!important] Likely exam question
> "I most likely will put an exam question about bias variance. Maybe something like, why am I critical of the word trade-off? Even though it's not wrong."

## Recap: the supervised setup

We're in the supervised setting: a Y we want to predict / model / explain, decomposed as

$$Y = f(X) + \varepsilon$$

where X are the predictors / inputs and ε is a random error term. The two non-negotiable properties of ε:

- **mean zero** on average,
- **independent of X**.

> "Imagine X is actually the true thing… then epsilon represents other stuff. Measurement noise. Noise due to fluctuations that have nothing to do with the thing that you have and are independent of what you're studying."

He emphasizes ε is "really tough to deal with because you don't really know anything about it, right? It's orthogonal to everything you're looking at."

The standard regression picture: X on the x-axis, Y on the y-axis, ε is the **vertical distance** from points to a fit line. The advertising example (sales vs TV/radio/newspaper spend, all in dollars) is the canonical ISL Figure 2.1 motivation — fit a line and ask how good a fit is, what the trade-offs are if you go more complex, what's just hopeless because of noise.

### Two reasons to estimate f

- **Prediction** — care only about Ŷ. The shape of f doesn't matter ("black box"). Finance: predict tomorrow's stock price; sacrifice all interpretability for a better number.
- **Inference** — care about the *form* of f. Which X's go in, what shape the relationship takes (linear, quadratic, oscillating). Scientists: which risk factors lead to death.

The same problem can be approached either way and gives "very different solutions often" in modern ML.

LLMs as an aside: ~trillion parameters, purely a prediction model — "we don't really care what's in it." But you may still care about *controlling* predictions, e.g. avoiding hallucinations / pathological outputs, even if you don't care about the parameters per se.

## Reducible vs irreducible error

For Ŷ = f̂(X) (no ε term — best guess of the noise is zero):

$$\mathbb{E}[(Y - \hat Y)^2] = \underbrace{(f(X) - \hat f(X))^2}_{\text{reducible}} + \underbrace{\mathrm{Var}(\varepsilon)}_{\text{irreducible}}$$

He works the derivation on the board. Substitute Y = f(X) + ε, expand the square (a−b)²-style:

- a² term → (f(X) − f̂(X))²
- b² term → ε²
- cross term → −2 (f(X) − f̂(X)) · ε

The cross term has expectation zero because E[ε] = 0 and ε is independent of X (and of f̂). The leftover ε² becomes Var(ε) — if we'd assumed ε ~ N(0, σ²) this is just σ², "but you can think of it as how big the air is."

> "We can do something about this f of x… largely dependent on choosing a good x and choosing a good f."

### Aside: deterministic relationships

> "Q: If there were a deterministic relationship between the response and a set of predictors, would there then be both reducible and irreducible errors?"

If the relationship fully determines Y (e.g. unit conversion feet→cm), there's no noise → no irreducible error. Almost never the case in practice.

## Parametric vs non-parametric

The book carves estimation methods into two classes.

### Parametric: linear regression

$$f(X) = \beta_0 + \beta_1 X_1 + \dots + \beta_p X_p$$

β₀ is the **bias term / intercept** ("the thing that doesn't vary with respect to the other variables"). The other βs are slope terms. With one predictor: a line, with intercept β₀ and slope β₁; ε is "how much off of the line it is."

Two-step recipe:

1. Select a form for f.
2. Estimate the parameters using a **training set** (a.k.a. "fit data").

Once you have β̂s you assume ε = 0 and read off Ŷ — the point estimate sits on the line.

### Non-parametric: KNN classification

> "The general idea is that there's no parameters." A non-parametric model "tries to have a very loose form" — but in practice you still pick a hyperparameter, here **k**.

[[knn-classification]] (specifically classification — there's also a regression variant covered later): given a new point, find its k nearest neighbors in feature space, take a majority vote.

- k = 4: P(blue) = (#blue in 4 nearest) / 4 → assign winner.
- Use **odd k** to avoid ties.
- k = 1: extremely wiggly decision boundary — "little islands of red, which probably doesn't make any sense."
- k = 150: very smooth boundary — maybe too smooth.

So even in the "non-parametric" model you have a flexibility knob (k), and changing it gives the same kind of trade-off you get from changing model complexity elsewhere.

> "How well you fit the data versus how well it generalizes is a common theme in the course."

He notes: KNN with small k is very sensitive to *which* data set you got — re-run the experiment, the boundary changes a lot. Large k is much more stable across resamples. (Foreshadowing variance.)

### Pros and cons table (verbatim flavor)

**Parametric**: simple, easy to understand, often interpretable, requires little data, "computationally cheap" — though he downgrades the computational-cost argument:

> "I would say in general this argument of computationally cheap is no longer as relevant as it used to be… would you really use a different method simply because it took your computer two seconds less to use when the other one is, you know, better in a way or makes fewer assumptions?"

Disadvantages: **f is constrained to a specific form** ("can be worse than we think"); the assumed form generally won't match the truth → poor estimate; limited flexibility; and dangerously, "it makes assumptions about what happens outside of your data which can often lead to very bad things."

**Non-parametric**: flexible, no strong assumptions about f. But: easy to overfit; need *lots* of data because

> "It's really just kind of interpolating your data. So if you don't have data in some place then you're going to get a bad interpolation."

KNN can't fill empty regions with anything; a parametric model can extrapolate using its assumed form (for better or worse).

## Inflexible vs flexible methods

Inflexible (rigid, structured):

- linear regression (M3)
- linear discriminant analysis (M4)
- subset selection, lasso (M6)

Flexible:

- KNN classification (M4), KNN regression
- smoothing splines (M7)
- bagging and boosting (M8/M9)
- neural networks (M11)

> "Something that is sort of underappreciated is that while we have flexible models, we also have good ways of making sure that the flexible models don't completely go crazy… we do have often good ways of restraining the flexible models."

This is the seed for [[regularization]] (ridge / lasso / shrinkage in M6) and is the prof's main reason for resisting "trade-off" framing — see below.

### Overfitting and underfitting

- **[[overfitting]]**: every training point sits on the curve, the fit gets wiggly, doesn't generalize. Like KNN with k = 1.
- **[[underfitting]]**: model too rigid / too few parameters. Curve in the data but you fit a line — missing structure.

Some models are *designed* so overfitting isn't such a problem; others have a strong tendency to overfit. The degree depends on the model.

## Polynomial regression simulation

The toy example. Generate n = 61 points from

$$Y = X^2 + \varepsilon, \quad \varepsilon \sim N(0, 4), \quad x \in [-2, 4]$$

Then *pretend you don't know* the truth and fit polynomials of degree 1, 2, 10, 20:

- **poly1**: linear → underfits. Captures the trend in [−2, 4] partially; over [−4, 4] it'd be a U and the line would average flat.
- **poly2**: quadratic → fits perfectly. *Includes* the true model. Too good, because every assumption is correct.
- **poly10**, **poly20**: include the truth as a special case but have many more terms — get pulled around by noise, wiggly.

> "This does not happen, right? This is what people want to happen. This doesn't happen. We never get the right model… But it's good for the math because then everything works out."

He really hammers that the "true-model is in your class" assumption almost never holds in practice. This polynomial setup is a teaching device, not a realistic situation.

### Train MSE vs test MSE

Training MSE:

$$\mathrm{MSE}_{\text{train}} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat f(x_i))^2$$

Always **monotonically decreases** as you add polynomial degree, because each more-flexible model contains the previous ones (poly10 contains poly2 — set the higher coefficients to 0). On training data you can never do worse.

> "Every model, if you make it more flexible, it will necessarily fit the data better."

But what we *want* is performance on **unseen data** — the test MSE, evaluated on (x₀ⱼ, y₀ⱼ) the model was not fit to. The test MSE is U-shaped: rapid decrease early, then increase as the fit starts chasing noise.

> "We don't want to predict last week's stock price. We want to predict the stock price of next week."

For this rigged simulation poly2 wins on test MSE.

What if you don't have a test set? **Make one** — split your data, e.g. 80/20. Cross-validation in **module 5** is the systematic version.

Why not just use train MSE? Because a low train error can be a sign of overfitting that *increases* test error. The training error doesn't account for model complexity.

## Bias-variance decomposition

The "U-shape is the result of two competing properties," which can be derived by manipulating the test MSE expression. The prof works it on the board (says "I don't think it's good to show math in slide form").

Start from

$$\mathbb{E}\!\left[(y_0 - \hat f(x_0))^2\right]$$

Substitute y₀ = f(x₀) + ε. We already showed this gives:

$$\mathbb{E}\!\left[\big(f(x_0) - \hat f(x_0)\big)^2\right] + \mathrm{Var}(\varepsilon)$$

(The reducible + irreducible split.) Now decompose the reducible part further.

**The trick**: add and subtract E[f̂(x₀)] inside the squared term:

$$f(x_0) - \hat f(x_0) = \big(f(x_0) - \mathbb{E}[\hat f(x_0)]\big) + \big(\mathbb{E}[\hat f(x_0)] - \hat f(x_0)\big)$$

Square this (a + b)² = a² + 2ab + b². The cross term

$$2 \cdot \big(f(x_0) - \mathbb{E}[\hat f(x_0)]\big) \cdot \mathbb{E}\!\left[\mathbb{E}[\hat f(x_0)] - \hat f(x_0)\right]$$

is **zero**, because E[E[f̂(x₀)] − f̂(x₀)] = E[f̂(x₀)] − E[f̂(x₀)] = 0.

> "We don't like cross terms — it's too much work to keep track of them. So conveniently we now have what's going on, we now have just those first two terms."

What's left:

$$\mathbb{E}\!\left[(y_0 - \hat f(x_0))^2\right] = \underbrace{\mathrm{Var}(\varepsilon)}_{\text{irreducible}} + \underbrace{\big(f(x_0) - \mathbb{E}[\hat f(x_0)]\big)^2}_{\text{squared bias}} + \underbrace{\mathrm{Var}(\hat f(x_0))}_{\text{variance}}$$

### What each term means

- **Irreducible** Var(ε): noise inherent to the data — you can't touch it.
- **Squared bias** (f(x₀) − E[f̂(x₀)])²: how far off the model's *expected* prediction is from the truth. The "truth minus the expected value of the [model's] truth."
- **Variance** Var(f̂(x₀)) = E[(E[f̂(x₀)] − f̂(x₀))²]: how much the prediction *varies* across different training sets — sensitivity to which data points you happened to draw.

> "Different models will have different sensitivity to how the bias and the variance will change depending on how you change your model."

He recommends going through this derivation yourself — important to understand what each expectation is over.

> "I think this is, of the theoretical stuff that we talk about [in] the course… a particularly interesting one."

### Reading the canonical bias-variance plot

(ISL Figure 2.12.) X-axis = flexibility, Y-axis = error. Three lines:

- dark red = test MSE,
- blue = squared bias,
- orange = variance,
- dotted horizontal = irreducible Var(ε).

Test MSE = irreducible + bias² + variance. As flexibility increases:

- bias decreases (better at fitting the truth on average),
- variance increases (more sensitive to specific samples — fitting noise),
- test MSE bottoms out somewhere in between.

At low flexibility: high bias (bad fit), low variance (model can't move much regardless of data). At high flexibility: opposite.

> "Initially our error will go down until it goes to some minimum, and then it goes up again."

Different datasets give differently-shaped curves but the same qualitative story.

## Why "trade-off" is misleading

This is the prof's hobby-horse and recurring closing point. The bias-variance **decomposition** is mathematically exact. The "trade-off" framing implies that to lower variance you must raise bias and vice versa. That's *not always true*.

> "What you can do is you can change how your model behaves and how much variance it has to give up in order to reduce the bias or how much bias it has to give up to reduce the variance. Because ultimately, the goal is to minimize both of these terms. And one way to do that is actually change what model you're fitting to your data."

Concrete example: keep degree-20 polynomial regression but **penalize the size of the βs** (shrinkage), or force most of them to zero (sparsity). Same flexibility ceiling, but the bias-variance curve flattens — variance stops exploding because you've removed the wiggle, while bias stays low because the truth is still expressible.

> "If I made this same plot, you wouldn't see the variance go up as much… the variance term would be more like that or something. The variance wouldn't explode because it's a different model."

This is foreshadowing **module 6** (ridge / lasso) and the broader theme of regularized flexible models — "a lot of machine learning in general… is how to make models that both fit the data well in terms of reducing the bias and also reducing this variance."

The prof flagged this directly as exam-relevant: an exam question asking *why* he resists "trade-off" is plausible.

## Closing notes

Out of time mid-deck (about polynomial-regression worked example). Next session (Monday → L04) continues with module 2 part 2 — slight deviation from the published schedule, which had module 3 starting then.

> "Yeah, I'm going slower in this theme — this module two — because I like it. I think it's interesting. Because I know we're already supposed to be on a different set of slides, but I will catch up later. So it'll all work out."
