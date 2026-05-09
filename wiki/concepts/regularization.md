---
concept: regularization
modules: [05-resample, 06-modelsel, 07-beyondlinear, 09-boosting, 11-nnet]
lectures: [L11, L12, L13, L14, L15, L16, L20, L23, L24, L26]
isl-ref: 6.2
exercises:
  - Exercise6.5 — apply ridge to Credit, compare with OLS
  - Exercise6.6 — apply lasso to Credit, compare with ridge and OLS
  - Exercise11.1d — how can a 10000-weight NN fit on 1000 obs (regularization)
  - Exercise11.4.2 — apply rotation/shift/flip data augmentation to CIFAR-10 CNN
related: [bias-variance-tradeoff, ridge-regression, lasso, elastic-net, cross-validation, smoothing-splines, cost-complexity-pruning, nn-regularization, gradient-descent-and-sgd, bagging, boosting, double-descent, standardization]
tags:
  - concept
  - specials
aliases:
  - shrinkage
  - penalization
  - L1 regularization
  - L2 regularization
---

# Regularization

The prof's framing: **the most important variant of model selection** in the course, and "the central trick of modern statistical learning." Add a constraint to the optimization that shrinks parameters (or trees, or NN weights) toward simpler / smaller / sparser solutions. Why it works is always the same answer: **trade a tiny bit of bias for a lot of variance reduction** ([[bias-variance-tradeoff]]). Comes in **explicit** flavors (L1/L2 weight penalties, smoothing-spline curvature penalty, tree cost-complexity, dropout, weight decay) and **implicit** flavors (mini-batch SGD's noise, bagging's averaging, model ensembling) — and the prof returns to it in every module from 6 onward.

## Definition (prof's framing)

> "Regularization is really a way of constraining the model. And constraining it in such a way that what comes out is good." — [[L12-modelsel-1]]

> "The constraint that we're putting on this system is not 'build parking lots,' but 'push your parameters towards zero.' And the result is you have a model that will generalize better on data that it hasn't seen." — [[L12-modelsel-1]]

> "I would argue this is the most important one that we talk about throughout, the most important form of this type of parameter selection." — [[L14-modelsel-3]]

The general optimization template:

$$\min_\theta \;\underbrace{L(\text{data}, \theta)}_{\text{fit}} \;+\; \lambda \cdot \underbrace{\Omega(\theta)}_{\text{penalty}}$$

The first term wants $\theta$ to fit the data; the second pulls $\theta$ toward "simpler." The hyperparameter $\lambda$ slides between them — chosen by [[cross-validation]]. The fit term can be any loss (squared error, log-likelihood, deviance, hinge); the penalty can be any complexity proxy ($L_2$ norm, $L_1$ norm, $\int g''(t)^2 dt$, $|T|$ for tree size, dropout rate, …).

> "If you think of it in terms of [GLMs], the first term could be the log likelihood of anything — Poisson log-likelihood, whatever distribution your data is. So the first term is very flexible, and you can stick this regularization term on any of those log-likelihoods." — [[L13-modelsel-2]]

## Returns in other modules

- [[L11-resample-2]] — first explicit appearance: [[bagging]] is framed as "implicit regularization" via averaging across bootstrap samples. *"By doing that, even though you're always just resampling the same data, you can actually remove bias from your model"* — and reduce variance. Ensemble averaging is the precursor of every later "stack many models" trick.
- [[L12-modelsel-1]] — the "central trick" framing introduced. **Ridge regression (L2)** as the first explicit shrinkage method. RSS + $\lambda \sum \beta_j^2$. Standardize first. $\lambda = 0$ → OLS, $\lambda = \infty$ → all betas zero.
- [[L13-modelsel-2]] — full module-6 treatment. **Lasso (L1)** as the second explicit method: RSS + $\lambda \sum |\beta_j|$. The L2 ball has no corners, the L1 diamond does → sparsity for free. Ridge "averages over correlated parameters (socialist)"; lasso "picks one (capitalist)." [[elastic-net]] = both penalties together.
- [[L14-modelsel-3]] — Bayesian view (ridge ≡ Gaussian prior on $\beta$, lasso ≡ Laplace prior — *"I really don't think I'd put this on the test"*). **Implicit regularization** teaser: SGD's noisy mini-batch gradients act as L2 ("min-norm solution among interpolators"). Out-of-scope-as-derivation, in-scope-as-concept.
- [[L15-modelsel-4]] — PCR ≈ a *discretized* ridge regression: ridge shrinks small-eigenvalue directions smoothly, PCR truncates them abruptly. Both are forms of regularization. [[partial-least-squares|PLS]] same family.
- [[L16-beyondlinear-1]] — **smoothing splines** introduce a *new kind* of penalty: $\lambda \int g''(t)^2 dt$ on the integrated squared second derivative. $\lambda = 0$ → arbitrarily wiggly fit; $\lambda \to \infty$ → straight line. Same "two objectives, one tuning parameter" pattern. **Effective df = trace($S$)** is the non-integer version of model complexity.
- [[L17-trees-1]] / [[L18-trees-2]] — **cost-complexity pruning** $C_\alpha(T) = \text{RSS} + \alpha |T|$ regularizes tree size. Same "loss + $\lambda$·penalty" template; CV picks $\alpha$.
- [[L20-boosting-2]] — boosting itself uses **multiple regularization levers**: shrinkage rate $\nu \le 0.1$ ("forces weak learners by not letting them have a strong vote"), [[stochastic-gradient-boosting|stochastic subsampling]], early stopping, and (for XGBoost) explicit L1/L2 on leaf weights, pruning rate $\gamma$, and [[dropout]].
- [[L23-nnet-1]] — **mini-batch SGD's implicit L2** is the prof's headline NN regularization fact: *"when there's an infinite number of exact solutions, it will find the solution where the L2 norm is minimized."* Powers-of-2 batch sizes for hardware, but the regularization effect is what matters for generalization.
- [[L24-nnet-2]] — full NN regularization menu: **L1/L2 weight decay, data augmentation (rotate/flip/shift/noise), label smoothing, early stopping, dropout (zero 20–50% of nodes per training step — never 50%), transfer learning**. *"You don't want to train a neural network without regularization."* Dropout is the prof's favorite ("almost no hyperparameters, trivial to use") and is conceptually identical to bagging inside a single network.
- [[L26-nnet-3]] — *"I can't think of an example where you'd ever want to train a neural network without a regularization."* The Hitters comparison is contrived because the NN was unregularized. Then **[[double-descent]]**: past the interpolation point, the optimization itself becomes a regularizer (minimum-norm solution among infinite interpolators).

## Notation & setup

- $\theta$ = parameters being regularized (regression $\beta$'s, NN weights, leaf values, tree size, …).
- $\lambda$ (sometimes $\alpha$, $\nu$, $\gamma$, dropout rate $p$) = the regularization strength. Hyperparameter, chosen by [[cross-validation]].
- $\Omega(\theta)$ = the penalty function. The "shape" of $\Omega$ determines what "simpler" means:
  - $\Omega(\theta) = \sum \theta_j^2$ → L2 / ridge / weight decay → smooth shrinkage.
  - $\Omega(\theta) = \sum |\theta_j|$ → L1 / lasso → sparsity.
  - $\Omega(\theta) = \int g''(t)^2 dt$ → smoothing spline → low curvature.
  - $\Omega(T) = |T|$ → cost-complexity pruning → fewer terminal nodes.
  - $\Omega(W) = $ "drop a random 20–50% of nodes" → dropout → ensemble inside a network.

## Formula(s) to know cold

### Ridge (L2)

$$\boxed{\;\hat\beta_\text{ridge} = \arg\min_\beta \left\{ \sum_i (y_i - x_i^\top \beta)^2 + \lambda \sum_{j=1}^p \beta_j^2 \right\}\;}$$

Closed-form solution; works when $p > n$ (where OLS doesn't); never sets $\beta_j$ exactly to zero.

### Lasso (L1)

$$\boxed{\;\hat\beta_\text{lasso} = \arg\min_\beta \left\{ \sum_i (y_i - x_i^\top \beta)^2 + \lambda \sum_{j=1}^p |\beta_j| \right\}\;}$$

Sets some $\beta_j$ exactly to zero (variable selection for free).

### Elastic net

$$\hat\beta_\text{enet} = \arg\min_\beta \left\{ \text{RSS} + \lambda \sum \beta_j^2 + \gamma \sum |\beta_j| \right\}$$

Combines L2's correlated-variable averaging with L1's sparsity.

### Smoothing spline

$$\hat g = \arg\min_g \sum_i (y_i - g(x_i))^2 + \lambda \int g''(t)^2 \, dt$$

### Cost-complexity pruning

$$C_\alpha(T) = \sum_{m=1}^{|T|} \sum_{i \in R_m} (y_i - \hat y_{R_m})^2 + \alpha |T|$$

### XGBoost leaf regularization

$$\Omega(f) = \gamma |T| + \tfrac{1}{2} \lambda \|w\|^2 + \alpha \|w\|_1$$

(L1 + L2 on leaf weights $w$, plus pruning by tree size $|T|$.)

### Mini-batch SGD update (implicit L2)

$$\theta^{(t+1)} = \theta^{(t)} - \eta \, \nabla_\theta L_\text{batch}, \qquad |\text{batch}| \ll N$$

Noise from the batch sampling biases the optimizer toward the **minimum-L2-norm** solution among all training-error minimizers ([[L23-nnet-1]]).

## Insights & mental models

### "Reduce variance" is the universal answer

> "Often we can substantially reduce the variance at the cost of a negligible increase in bias. I think that's been always surprising to me — just how little bias you need to get a lot of reduction in variance." — [[L12-modelsel-1]]

Every regularizer in the course operates by accepting a small bias bump for a big variance reduction. That's the *only* sales pitch — and it's the [[bias-variance-tradeoff]] hammer applied everywhere.

### Why the L2 ball doesn't sparsify but the L1 diamond does

> "It's the same idea that both this [ellipse] and this [diamond] have constant value of the objective — and where they intersect is one of these corners. It doesn't have to be, but they tend to." — [[L13-modelsel-2]]

Geometric story (ISL Fig 6.7): in $\beta_1$-$\beta_2$ space, the RSS contours are ellipses; the L2 constraint set is a circle (smooth boundary, no axes-aligned corners), the L1 constraint set is a diamond (sharp corners on the axes). Ellipses tend to first touch a circle at a generic interior point ($\beta_1 \neq 0$ AND $\beta_2 \neq 0$); they tend to first touch a diamond at one of its corners ($\beta_1 = 0$ OR $\beta_2 = 0$).

In the function-form: the squared penalty is *flat* at zero ($d\beta^2/d\beta = 2\beta = 0$ at $\beta=0$) — no gradient pushing things to *exactly* zero. The absolute value is *kinked* at zero (constant gradient $\pm 1$ on either side) — keeps pushing toward zero all the way up to it.

### Capitalist vs socialist

> "Lasso is like very capitalist. Just shoot everyone, let all the poor people die, let the one rich guy win. And L2 is more socialist. Honestly, if you go to the extremes of either one — like where the US is tending — and if you went to the extreme of socialism — probably both are too bad. Which is why we have something called the elastic net." — [[L13-modelsel-2]]

Operational meaning: when two predictors are highly correlated, lasso picks one and zeros the other (sparse, but arbitrary on which one). Ridge averages them — both keep moderate values, no winner. Elastic net does both depending on the global pattern.

### Explicit vs implicit

**Explicit** = penalty in the loss function (ridge, lasso, smoothing splines, cost-complexity, XGBoost L1/L2 on leaves, NN weight decay, dropout in the loss).

**Implicit** = the algorithm's behavior happens to act like a regularizer even though no penalty is written down. The course's three implicit examples:

1. **Mini-batch SGD** ([[L14-modelsel-3]], [[L23-nnet-1]]): the gradient noise biases the optimizer toward minimum-L2-norm solutions. *"Whenever you use this and you're in a problem setting where there's an infinite number of exact solutions, it will find the solution where the L2 norm is minimized."*
2. **Bagging** ([[L11-resample-2]]): averaging $B$ bootstrap-trained models reduces variance.
3. **The minimum-norm interpolator in the over-parameterized regime** ([[double-descent]], [[L26-nnet-3]]): when $p \gg n$ and the model has infinitely many zero-training-error solutions, fitting via the pseudo-inverse picks the smallest-norm one. **This is implicit ridge regression.** It's what makes [[double-descent]] work.

### Pre-step: standardize

[[standardization]] is mandatory before ridge, lasso, elastic net, PCA/PCR, k-means, hierarchical clustering, KNN, and almost every NN. The penalty $\sum \beta_j^2$ or $\sum |\beta_j|$ treats all coordinates symmetrically — so if one predictor is in $10^{-4}$ units and another in $10^3$, the small-scale one's $\beta$ is enormous and gets crushed by the penalty for no good reason. Z-score everyone first.

### Picking $\lambda$ — and "refit on selection"

The standard recipe ([[L14-modelsel-3]]):

1. Cross-validate over a $\lambda$ grid → pick the $\lambda$ minimizing CV error (or one-SE simpler).
2. Refit the chosen model on **all** training data with that $\lambda$.
3. **For lasso specifically**: optionally drop the variables that lasso zeroed, then refit unpenalized on the survivors. *"Then you've essentially done model selection."*

### The shape of "more regularization = simpler"

Direction-of-effect cheat sheet (these are common T/F traps):

| Knob | Direction increases → |
|---|---|
| Ridge $\lambda$ ↑ | All $\beta_j$ shrink toward zero (never reach zero) |
| Lasso $\lambda$ ↑ | $\beta_j$'s hit zero one by one (sparsity grows) |
| Smoothing spline $\lambda$ ↑ | Function becomes **smoother** (straight line at $\infty$) — opposite direction from polynomial degree, classic T/F trap |
| Tree pruning $\alpha$ ↑ | Tree gets **smaller** (fewer terminal nodes) |
| Boosting shrinkage $\nu$ ↓ | Each tree contributes less → need larger $M$, generalizes better |
| RF mtry ↓ | Trees more decorrelated → variance reduction kicks in |
| NN dropout rate ↑ | More aggressive ensembling; 20% common, never 50% |
| Mini-batch size ↑ | Less noise → less implicit L2 regularization |

### "Why doesn't the $\sigma^2$ term shrink?"

Regularization attacks the *variance* and (a tiny bit of) bias. Irreducible noise stays the same. CV's U-shape moves down by the variance reduction; the floor stays at $\sigma^2$.

## Exam signals

> "I would argue this is the most important one that we talk about throughout, the most important form of this type of parameter selection." — [[L14-modelsel-3]]

> "The most important one… regularization, it really goes at the core of what a lot of statistical learning, machine learning is." — [[L14-modelsel-3]]

> "Importantly, you can have many situations where for example p is bigger than n. In which case, you have a model that you cannot fit. In standard regression, the matrix is non-invertible. … But if you actually have a regularizer, then it can be made unique." — [[L14-modelsel-3]]

> "I can't think of an example where you'd ever want to train a neural network without a regularization." — [[L26-nnet-3]]

> "Mini-batch stochastic gradient descent actually gives you an implicit L2 regularization, which is super weird." — [[L23-nnet-1]]

> "Worth learning how to interpret this geometric interpretation of ridge and lasso. You know, because it really shows how the two objectives combined." — [[L14-modelsel-3]]

The 2025 Q3a (lasso vs least squares) was solved via direct bias-variance reasoning ([[L27-summary]]) — *"less flexible than LS, improved accuracy when the increase in bias is less than the decrease in variance."* Q6b (lasso CV on Hitters) is the inverse: when the variance-reduction is *not* enough to offset the bias bump, you don't regularize.

## Pitfalls

- **Forgetting to standardize** before ridge/lasso/PCA/k-means → small-scale predictors get over-penalized and dominate distance computations.
- **Ridge vs lasso direction trap**: ridge **never** sets coefficients to zero; lasso **does**. Don't confuse them. ([[L13-modelsel-2]] geometric reason: L2 ball has no corners.)
- **Smoothing spline direction trap**: $\lambda$ ↑ → **smoother**, not wigglier. Opposite direction from polynomial degree. Easy T/F flip.
- **Refitting after lasso to "clean up" doesn't do CV again** — you've already used CV to pick the variables; refitting unpenalized just removes the shrinkage bias on the survivors.
- **AIC/BIC/Cp are NOT regularization in the optimization sense** — they pick among models that were each fit by unregularized OLS. Regularization changes the *fit*; AIC/BIC/Cp change the *selection* across already-fit candidates.
- **L2 ≠ L1 + L1**: elastic net needs *both* penalty terms; the L2 part doesn't sparsify on its own, so you can't get sparsity without the L1 component.
- **PCR / PLS are dimensionality reduction, not regularization in the strict sense** — but [[L15-modelsel-4]]: *"PCR can be seen as a discretized version of ridge regression."* Both push down on the small-eigenvalue directions.
- **"Implicit regularization" doesn't mean you don't need explicit too** — the prof's L24 menu (L1/L2/dropout/augmentation/early stopping) is layered on top of mini-batch SGD's implicit L2.
- **Dropout rate 50% is too aggressive** — *"20% is very common, never use 50%."* ([[L24-nnet-2]])
- **Early stopping** can over-tune your validation set: each epoch you peek you've effectively used validation as training. Use a separate test set, or [[nested-cv-and-cv-pitfalls|nested CV]].
- **Bayesian interpretation of ridge/lasso (Gaussian/Laplace priors) is OUT of scope** — *"I really don't think I'd put this on the test, just because it kind of assumes a lot of knowledge that maybe you don't have."* ([[L14-modelsel-3]]). Cool framing only.

## Scope vs ISLR

- **In scope:** L1/L2 formulation, geometric picture (Fig 6.7), how to choose $\lambda$ via CV, why standardize first, $p > n$ as the headline use case, the shape of CV vs $\lambda$ curve, ridge ↔ PCR analogy, smoothing spline penalty $\int g''^2$, cost-complexity pruning $|T|$, NN regularization menu (L1/L2/dropout/augmentation/early stopping/transfer learning), implicit regularization via mini-batch SGD (concept).
- **Look up in ISLR:** §6.2 (full ridge + lasso treatment, including Fig 6.7 geometry); §6.3 (PCR + PLS as dimensionality-reduction regularizers); §7.5.2 (smoothing-spline $\lambda$ selection); §8.1.1 / §8.2.5 (cost-complexity pruning); §10.7 (regularization for NNs).
- **Skip in ISLR (book-only, prof excluded):**
  - **Bayesian interpretation of ridge/lasso (Gaussian / Laplace priors)** — [[L14-modelsel-3]]: *"I really don't think I'd put this on the test."*
  - **L0 norm / "Optimal Brain Damage"** — [[L14-modelsel-3]]: *"we won't go into it because it's not used in practice."*
  - **Detailed elastic-net tuning** — concept only, no worked example expected.
  - **Vanishing/exploding gradients, batch normalization, weight initialization (Xavier/He), Adam optimizer internals** — [[L23-nnet-1]] / [[L24-nnet-2]]: not discussed in depth, not on the exam.
  - **Formal proof that mini-batch SGD = implicit L2** — [[L24-nnet-2]]: *"It has been proven, which is nice. The math is there, but it's not super short."* Concept in scope; proof out.

## Exercise instances

- **Exercise6.5** — apply ridge to Credit data, compare with OLS. Standardize first.
- **Exercise6.6** — apply lasso to Credit, compare with ridge and OLS. Note which $\beta_j$'s go exactly to zero.
- **Exercise11.1d** — *"how can a 10000-weight model fit on 1000 obs?"* Answer: regularization (any form).
- **Exercise11.4.2** — apply rotation/shift/flip data augmentation to CIFAR-10 CNN. Concrete example of implicit/explicit regularization for NNs.

## How it might appear on the exam

- **T/F or fill-in on direction-of-effect**: $\lambda$ ↑ in ridge → coefficients shrink (TRUE), become exactly zero (FALSE — that's lasso); $\lambda$ ↑ in smoothing spline → wigglier (FALSE — smoother); etc.
- **Why lasso produces sparsity but ridge doesn't** — geometric answer (corners on axes), or function-form answer (kink at zero vs flat at zero).
- **Capitalist vs socialist comparison** — given two correlated predictors, what does each regularizer do? Lasso picks one; ridge averages them.
- **Why regularization helps when $p > n$** — OLS has no unique solution (singular $X^\top X$); regularization makes the optimization convex with a unique minimum.
- **2025 Q3a-style: lasso vs LS** — answered via bias-variance: lasso is less flexible, accepts a small bias bump in exchange for variance reduction, wins when the variance reduction dominates.
- **2025 Q6b-style: lasso CV vs unregularized linear** — given that the lasso-CV MSE is roughly equal to or worse than unregularized, conclude that all parameters matter and the variance reduction isn't enough to offset the bias.
- **NN regularization menu** — name three forms and explain each. Bonus: explain why mini-batch SGD is *also* a regularizer.
- **Smoothing spline $\lambda$**: $\lambda = 0$ → wiggly; $\lambda = \infty$ → straight line. Effective df = trace($S$).
- **Cost-complexity pruning** — write the objective $\text{RSS} + \alpha|T|$ and explain how CV picks $\alpha$.
- **"Why does dropout work?"** — same logic as bagging, applied inside a single network. Forces redundancy, prevents any single neuron from being critical.
- **Mathy adjacent**: explain in words why the squared loss can't drive coefficients to zero (gradient is $2\beta_j$, vanishes at zero) but the absolute-value loss can (gradient is $\pm 1$, constant push toward zero).

## Related

- [[bias-variance-tradeoff]] — the lever; every regularizer trades a tiny bit of bias for big variance reduction
- [[ridge-regression]] — canonical L2; closed-form, smooth shrinkage, never zero
- [[lasso]] — L1; sparsifies for free
- [[elastic-net]] — both penalties; centrist compromise
- [[ridge-vs-lasso-geometry]] — the ellipse-meets-diamond/circle picture
- [[principal-component-regression]] — discretized ridge (truncate small-eigenvalue directions instead of shrinking them)
- [[smoothing-splines]] — regularization on functional smoothness via $\int g''^2$
- [[cost-complexity-pruning]] — regularization on tree size via $\alpha |T|$
- [[boosting]] — uses shrinkage $\nu$, weak learners, early stopping, [[stochastic-gradient-boosting|subsampling]] — multiple regularization levers
- [[xgboost]] — adds explicit L1/L2 on leaf weights plus dropout
- [[nn-regularization]] — the full NN menu (L1/L2/dropout/augmentation/label-smoothing/early-stopping/transfer-learning)
- [[gradient-descent-and-sgd]] — mini-batch SGD as implicit L2 ("min-norm solution among interpolators")
- [[bagging]] — implicit regularization via averaging
- [[dropout]] — random ensembling inside a network
- [[double-descent]] — past the interpolation point, the optimization itself becomes a regularizer
- [[standardization]] — mandatory pre-step before any regularizer
- [[cross-validation]] — how you pick $\lambda$
