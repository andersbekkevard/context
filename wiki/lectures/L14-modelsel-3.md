---
lecture: 14
date: 2026-03-02
module: 06-modelsel
title: Model Selection and Regularization 3 (PCR/PCA)
slides: modules/6ModelSel/selection_regularization_presentation_lecture2.md
topics:
  - regularization
  - lasso
  - ridge-regression
  - implicit-regularization
  - bayesian-interpretation-of-regularization
  - dimensionality-reduction
  - principal-component-regression
  - principal-component-analysis
  - partial-least-squares
  - collinearity
  - cross-validation
  - bias-variance-tradeoff
tags:
  - lecture
  - module/06-modelsel
aliases:
  - Lecture 14
---

# L14: Model Selection and Regularization 3 (PCR/PCA)

The prof recaps L1/L2 [[regularization]] as the third leg of model selection (after best-subset and forward/backward), introduces **implicit regularization** as a teaser, gives the Bayesian-prior view of ridge and lasso, then pivots to **dimensionality reduction** (specifically [[principal-component-regression]] via [[principal-component-analysis|PCA]]) as a different way to tame too-many-correlated-predictors. Ends early (doctor's appointment) with [[partial-least-squares]] flagged for next time.

## Key takeaways

- **Regularization is the most important variant of model selection** he'll teach. *"This is the most important one that we talk about throughout, the most important form of this type of parameter selection."* It accomplishes a lot with little: sparsity, uniqueness when $p > n$, variance reduction.
- **Lasso vs Ridge as personalities**: lasso "encourages winners and losers" (capitalist), ridge "encourages ties / everyone has a vote" (socialist). Geometric picture (ISL Fig 6.7) shows why: *"worth learning how to interpret this geometric interpretation."*
- **Implicit regularization** (teaser, not on the test now): SGD's noisy mini-batch gradients act as an L2 regularizer. *"Those tricks that are so critical to modern tools can often be mapped back to a role of L2 normalization."*
- **Bayesian view**: ridge ≡ Gaussian prior on $\beta$; lasso ≡ Laplace prior. Prof: *"I really don't think I'd put this on the test"*, assumes Bayes background students don't have. Cool framing only.
- **PCR pipeline**: standardize $X$ → run PCA → keep first $M$ PCs as new predictors $Z_m$ → fit $y$ on $Z$ → back out $\beta$'s from the $\phi$'s. PCs are constructed greedily: max variance, $\sum \phi^2 = 1$, orthogonal to previous PCs.
- **PCA is not scale invariant, standardize first.** Otherwise the variable with the biggest unit dominates. *"PCA is not scale invariant"* (verbatim slide).
- **PCA is unsupervised**: the PCs are chosen using only $X$, not $y$. That's the core difference from PLS, which uses $y$ to guide the directions (next lecture).

## Recap: model selection so far

Module name is *model selection and regularization*. Three big families covered so far:

1. **Best subset**: try all $2^p$ combinations. Optimal but explodes combinatorially.
2. **Forward / backward / hybrid**: greedy approximations. Add one predictor at a time, or start full and prune. Much cheaper, not optimal.
3. **Regularization (shrinkage)**: penalize $\|\beta\|$ to shrink unimportant coefficients toward (or to) zero.

The framing throughout: more predictors → better training fit but worse out-of-sample. The error decomposes into [[bias-variance-tradeoff|bias and variance]]; regularization and selection both attack the **variance** term. *"We'll lose a little bit of prediction accuracy on our training data, but we'll do better out of sample."*

> "Picking your parameters correctly, really important, really good."

[[cross-validation]] is how we pick those parameters honestly. The prof emphasizes how easily you can get great training error and terrible generalization: *"this happens much more subtly all the time. It ends up in publications. It ends up in making the wrong decision related to engineering stuff. I just happens everywhere."*

### Why regularization is the headliner

Subset selection breaks at scale. *"If you had to go through and try to figure out which of those trillion parameters matter by removing one and then training a whole new model… that's a complete waste of time."* Regularization gets you the same goal (variance reduction, effective parameter shrinkage) without retraining for every subset.

> "I would argue this is the most important one that we talk about throughout, the most important form of this type of parameter selection."

## Regularization recap: explicit and implicit

The combined optimization:

$$\min_\beta \;\|y - X\beta\|^2 + \lambda \|\beta\|_p^p$$

- $p=1$: lasso (L1), sum of $|\beta_j|$
- $p=2$: ridge (L2), sum of $\beta_j^2$

You're optimizing two objectives simultaneously: fit the data, and don't let the $\beta$'s get big. The penalty restricts the solution space: *"makes it more of a restrictive problem and specifically restrictive in terms of getting solutions that have small values of beta, have more zeros."*

Why this is powerful:

- **Allows $p > n$ (wide matrices).** OLS alone is non-convex (infinite solutions); add the penalty and you get a unique solution.
- **Restricts model space in modern ML** even when there's no unique solution; keeps the search tractable.
- **Reduces variance** in the bias-variance decomposition. *"Trivially true if lambda is super big — it just makes everything zero. But when it's not super big, while not obvious, it does actually have that effect, and you can work it out mathematically."*

### L0 aside (not on the exam)

Cool name only: L0 regularization (counting nonzero $\beta$'s) was originally called **"Optimal Brain Damage"** in the 60s. *"It's actually related to the model selection stuff, but again, we won't go into it because it's not used in practice."*

### Lasso vs Ridge: different personalities

> "[Lasso] encourages winners and losers, right? It's the capitalist regularization method. Whereas the ridge one… encourages ties, encourages that everyone has a vote and no one has zeros."

Geometric picture (ISL §6.2.2 / Fig 6.7): the level sets of the data objective intersect the L1 diamond at corners (zeros), but intersect the L2 ball typically at interior points (nonzero, shrunk). The prof gestures at the figure rather than re-derive:

> "It's worth learning how to interpret this geometric interpretation of ridge and lasso. You know, because it really shows how the two objectives combined… you can see why lasso is more likely to choose winners and losers, and ridge regression is more likely to give you averages or find compromises, do this socialist thing where everyone counts."

### Implicit regularization (teaser)

By contrast with **explicit** regularization (where you write the penalty into the objective), **implicit** regularization shows up "by accident", meaning the original method isn't framed around regularizing, but it ends up doing it.

The example he gives: stochastic gradient descent. Instead of using all data per update, you use a mini-batch. *"What happens is that you have this noise in the gradient of the optimizer. And so what that noise does is it actually acts as an L2 regularization."*

> "There have been very nice proofs showing that by optimizing with just a subset of data every time, you're implicitly regularizing the data."

Especially relevant when $p > n$ (multiple solutions). The implicit regularizer steers SGD toward the **min-norm solution** (in the L2 sense). *"Those tricks that are so critical to modern tools can often be mapped back to a role of L2 normalization. So it's like, oh, we use this fancy thing, it's so cool, and the reason it works so well is often something as simple as, oh, that's actually an L2."*

Out of scope for now ("we'll talk about it later"), but important context.

## Bayesian interpretation of ridge and lasso

Take $P(Y|X) \cdot P(X)$: Bayesian model. Take logs:

$$\log P(Y|X) + \log P(X)$$

The first term, **assuming Gaussian likelihood**, is equivalent to maximizing $-\|y - X\beta\|^2$ (least squares). The second, the **log prior**, is whatever assumption you make on $\beta$.

- **Ridge** ≡ Gaussian prior on $\beta$. Penalty $\beta^2$ matches the log of a Gaussian density.
- **Lasso** ≡ **Laplace** prior on $\beta$. Penalty $|\beta|$ matches the log of a Laplace.

Why this is a useful perspective:

- **Gaussian prior** has thin tails: very unlikely to draw $\beta$ values far from 0, but also unlikely to draw exactly 0. So ridge gives you small-but-nonzero $\beta$'s. Matches the math.
- **Laplace prior** has more mass at zero AND heavier tails, so lasso can give you exact zeros AND occasionally larger values when warranted.

> "Ridge just chops it off, which you can also see just in the form of the equation — that $x^2$ term, it's penalizing very big numbers but not giving you much penalization at zero."

> [!note] Exam call
> "I'm going to mention this just as a cool way of interpreting it, but I really don't think I'd put this on the test, just because it kind of assumes a lot of knowledge that maybe you don't have." (Bayes prerequisites.)

## Picking $\lambda$ and refitting

Standard recipe:

1. **Cross-validate** over a grid of $\lambda$. Pick the $\lambda$ minimizing CV error.
2. **Refit on all the data** (or all the data minus your held-out test set) with the chosen $\lambda$.
3. *Optionally* (especially with **lasso**): note which coefficients went exactly to zero, drop those variables, then **refit** without the penalty on the surviving ones. *"Then you've essentially done model selection, kind of like in the sense that we've talked about last time."*

Standardize $X$ first so the betas are on comparable scales (and the penalty hits them comparably). *"Yeah, also that way the betas all have the same amplitude or it would have a similar amplitude. So it is typically standardized."*

## Pivot: dimensionality reduction as a third strategy

We've now seen two ways to handle too-many-predictors: **subset selection** (drop variables) and **shrinkage/regularization** (penalize them). Now a third: **transform** them.

Set-up: $y = X\beta + \varepsilon$ where $X$ is $n \times p$, $p$ too large or columns too correlated. Idea: build a new matrix $Z$ that's $n \times M$ with $M < p$, where each column is a linear combination of the $X$ columns:

$$Z_m = \sum_{j=1}^p \phi_{jm} X_j$$

Then fit

$$y = \sum_{m=1}^M \theta_m Z_m + \varepsilon$$

a smaller, easier regression. **Then back out the implied $\beta$'s** by composing the linear maps:

$$\beta_j = \sum_{m=1}^M \theta_m \phi_{jm}$$

> "You're taking the X, you're squishing it down to fit your model, and then you go backwards to the original model again."

This **constrains the coefficients of the standard linear regression**: you've forced $\beta$ to live in an $M$-dimensional subspace defined by the $\phi$'s. ISL §6.3.

### Why bother: the multicollinearity angle

If two $X_j$'s are nearly identical (or strongly correlated more generally, **[[collinearity|multicollinearity]]**), the OLS fit can't tell them apart. *"Parameters can trade off of each other, which is bad."* You get many near-equivalent solutions, high coefficient variance, terrible generalization.

> "Even if you only have three [predictors], if two of them are identical, the model's not going to fit. They're not identical, but very close — which is this issue called multicollinearity."

Squishing to orthogonal $Z$'s **removes the redundancy**. Each $Z_m$ is independent of the others by construction. Cleaner fit, lower variance.

### Two methods

We'll cover two ways to choose the $\phi$'s, both **linear** combinations:

- **PCR: Principal Components Regression** (today): use [[principal-component-analysis|PCA]] to pick the $\phi$'s. **Unsupervised**, uses only $X$.
- **PLS: Partial Least Squares** (next lecture): supervised, uses $y$ to guide the directions.

Local color on PLS: *"developed by a Swede"* (Herman Wold). *"Locally relevant because… commonly used in this field called chemometrics, and one guy who's fairly prominent in chemometrics and also in the development of these methods is actually Harold Martens, if anyone's heard of him. He was big in this partial least squares stuff. He wrote some of the early papers in the 70s."* Used widely on wide chemometrics data.

## Principal Component Analysis (PCA)

Will get a fuller treatment in **Chapter 10 (unsupervised learning)**. Today: just enough to use it inside PCR.

### The construction

Start with $X$ (standardized: mean 0, variance 1 per column; see scale-invariance note below).

The $m$-th principal component $Z_m = \sum_j \phi_{jm} X_j$ is chosen to satisfy three conditions:

1. **Maximize $\text{Var}(Z_m)$**: find the direction of maximal variance in the data.
2. **Constraint $\sum_j \phi_{jm}^2 = 1$**: otherwise variance can be inflated trivially by scaling the $\phi$'s. *"You don't want the variance to increase just because the fees are bigger. That's dumb."*
3. **Orthogonality**: $Z_m \perp Z_{m'}$ for all $m' < m$, i.e., uncorrelated with all previous PCs.

So $Z_1$ is the direction of maximum variance, $Z_2$ is the direction of maximum *remaining* variance orthogonal to $Z_1$, and so on.

> "There's a lot of things that are perpendicular to the original set of weights, and you just want to find of those many ones that are orthogonal to the previous set, which one maximizes the variance and keeps that norm constant."

### Why it's computationally easy

> "PCA sounds like it would be a difficult optimization problem, but actually it's not, which is one of the reasons it's used very often."

It reduces to an **eigendecomposition / SVD** of the (sample) covariance matrix of $X$. Pearson published the technique over 100 years ago; it became popular precisely because you could solve it by hand (well, with linear algebra) before computers. The full mechanics come in Chapter 10.

### Full vs reduced

If you compute **all $p$ PCs**, you've explained **100% of the variance**: you've just rotated $X$ into a new orthogonal basis. Useless on its own; a one-to-one mapping with no compression.

The point is the **ordering**: $Z_1$ has the highest variance, $Z_2$ the next, etc. You truncate at some $M < p$ and discard the low-variance directions. Common stopping rule: **cumulative variance explained**, with thresholds typically **90%, 95%, or 99%**.

### Scree-style plots

The prof shows scree-style cumulative-variance plots. Y-axis: cumulative proportion of variance explained. X-axis: number of PCs.

- The first PC contributes a lot, the next less, the next less still: *"you're going to be explaining less and less variance."*
- You eyeball where the curve flattens and the marginal gain becomes uninteresting.
- E.g. on a $p=13$ example: 90% threshold lands at ~7 PCs. *"Remember, with model selection, 7 is a lot better than 13 when you're fitting a model."*
- On a $p=300$ example: 95% might give you 75–80 PCs, still a huge reduction.

(A student asks how the variance is calculated. Prof: it's variance captured from the per-PC contributions; can be expressed via the covariance/sum matrix of the $Z$'s built so far. *"Wow, that's a good question. So it would be nice if I had an equation here, wouldn't it… we'll define it better later."* Promised a cleaner definition next time.)

### Worked example: Ad spending vs Population

ISL §6.3.1 ad-data figure (slides §"PCA Example - Ad spending"). Two predictors: population and ad spending. They're correlated: bigger populations get more ad spend. That correlation is bad for OLS.

PCA on these two:

- **First PC** (green line in the slide): points along the direction of maximal joint variance, the diagonal of the cloud. Captures both predictors at once.
- **Second PC** (blue dashed): perpendicular, captures the remaining (much smaller) variance.

Project the data onto these two new axes and they're now **uncorrelated by construction**. The first PC alone correlates strongly with both pop and ad, so a single regressor on PC1 captures most of what you'd get from both originals: *"you have one variable that correlates with both of them, and that's what you're going to use to fit."*

> "By reducing the redundancy in the data and squishing it down to fewer parameters, you can remove this issue. You can make sure that every one of these new X values, these Zs, are no longer — where they really are independent or orthogonal."

### PCA also as visualization

Side use: project to first 2 (or 3) PCs to **visualize** high-dimensional data. *"It makes no sense to plot something in 50 dimensions. But if you take the first two principal components, that would be the first two directions of maximal variance, sometimes that actually does something for you. Sometimes it actually does show you something interesting about the data. Not always."*

### Standardize first: PCA is not scale invariant

> "PCA is not scale invariant. So if you don't standardize them so that their mean is zero and their variance is one, then if one had a standard deviation of like a million, then that will be your strongest variable. That will be the thing that gives you the highest variance, which is annoying because you don't want it to just be that the scale of the variable is bigger."

This shows up on the slide as a bullet too. **Standardize, then PCA.**

## PCA's relationship to multicollinearity

Three ways the course has now offered to deal with [[collinearity|multicollinearity]]:

- **L1 (lasso)**: pick one of the correlated variables, zero out the others.
- **L2 (ridge)**: hold both back, share the load. *"Creates like a tug of war between the two, neither one, it penalizes both of them getting bigger, so it kind of holds them back."*
- **PCA / PCR**: rotate to an orthogonal basis where the correlation is gone by construction.

Caveat: PCA only handles **linear** correlation. *"If your issue is actually not a linear correlation but some sort of complicated thing, then it might not find it. But linear works well."*

## Wrap-up and what's next

He had to cut early for a doctor's appointment. Stops mid-PCR. PCA construction covered, but PCR cross-validation tuning of $M$ (the number of components to keep) and **PLS** are deferred to the next lecture.

Suggested exercise: *"A good exercise is to go back to this credit data set and then see how many principal components you should use, make that same figure where you look at the explained variance, and you'll see it saturates at some point, and then you're like, oh, well, chop here."*

> "Tomorrow we'll pick up from here and continue and finish up the section on model selection."

No exam-flag drops in this lecture beyond the explicit Bayes-is-not-on-the-test note. Treat the geometric ridge/lasso interpretation as worth fluency in (his "worth learning how to interpret"), and PCR's pipeline (standardize → PCA → fit → back-transform) as the canonical procedure.
