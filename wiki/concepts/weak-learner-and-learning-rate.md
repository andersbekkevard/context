---
concept: weak-learner-and-learning-rate
module: 09-boosting
lectures: [L19, L20]
isl-ref: 8.2.3
exercises:
  - Exercise9.3 - interpret ν, how to choose it, where it enters the gradient tree boosting algorithm
related: [boosting, gradient-boosting, adaboost, stochastic-gradient-boosting, xgboost, regularization, bias-variance-tradeoff, cross-validation]
tags:
  - concept
  - module/09-boosting
aliases:
  - shrinkage
  - learning rate
  - eta
  - nu
  - weak learner
---

# Weak learners and the learning rate $\nu$

The two "force everything to be small" hyperparameters that make boosting work. **Weak learner** = a deliberately shallow / dumb base model (typically a tree with few splits) that individually does barely better than random. **Learning rate $\nu$** = a scalar in $(0, 1]$ multiplying each tree's contribution before it's added to the ensemble, so each step is a *small* step. Together they are why boosting reduces bias without exploding variance.

## Definition (prof's framing)

> "We want weak learners, and this way we're kind of forcing them to be weak by not letting them have a strong vote." - [[L19-boosting-1]]

Restated more carefully a lecture later, in the steepest-descent picture:

> "Weak learners sound like a weird thing to want, but if you remember, the thing we really don't want to do is overfit. So weak learners won't overfit. And also, in the gradient descent picture, we don't want to make that huge jump." - [[L20-boosting-2]]

## Notation & setup

- **Weak learner** = the per-iteration base model. For tree boosting, a small tree characterized by:
  - **Tree depth** $d$ (or equivalently number of leaves $J = d+1$): typically $4 \le J \le 8$ in practice. $J = 2$ → a stump (one split). $J = 3$ → captures pairwise interactions. Rarely $J > 6$.
  - **Minimum observations per terminal node**: secondary, less important for big data.
- **Learning rate** $\nu$ (also $\eta$, the prof switches mid-lecture): the scalar multiplying each new tree before it's added to the running model:
  $$f_m(x) = f_{m-1}(x) + \nu \cdot \sum_j \gamma_{jm}\,\mathbb{1}(x \in R_{jm}).$$
  Empirical rule: **$\nu \le 0.1$**. Common values: 0.1, 0.05, 0.01, 0.001.

## Formula(s) to know cold

Where $\nu$ enters the gradient tree boosting algorithm, modify step 2(d) of Algorithm 10.3:

$$f_m(x) \;=\; f_{m-1}(x) \;+\; \nu \cdot \sum_{j=1}^{J_m} \gamma_{jm}\, \mathbb{1}(x \in R_{jm}).$$

Equivalently for the regression-tree special case (Algorithm 8.2 from the slides):

$$\hat f(x) \leftarrow \hat f(x) + \nu \cdot \hat f^b(x), \qquad r_i \leftarrow r_i - \nu \cdot \hat f^b(x_i).$$

The **$M$–$\nu$ trade-off** ([[L19-boosting-1]] / [[L20-boosting-2]]):

> "If you had a very weak eta… like 0.00001 then you'd need to do this a whole bunch of times before you get anywhere… whereas if it was bigger then you'd need fewer." - [[L20-boosting-2]]

Operational consequence: smaller $\nu$ ⇒ larger $M$ needed to reach the same fit. Standard recipe: **fix $\nu$ small (≈ 0.1), then tune $M$ via early stopping** on a validation set or CV.

## Insights & mental models

- **The steepest-descent analogy.** Each tree is one step on the loss landscape. $\nu$ is the step size, too big you overshoot the minimum or oscillate, too small you take forever. The same intuition you have for vanilla gradient descent, only here the "step direction" is itself an estimated tree.
- **Forcing weakness is a feature, not a workaround.** Strong individual learners would each over-explain the residual at their step, leaving little for subsequent trees → poor diversity → worse generalization. Weak learners + many of them = the bias gets aggregated down without inviting variance.
  > "We want weak learners… we don't want overly precise learners. We want learners that make good progress and a step in the right direction, but not ones that are going to throw you out into the, you know, too far or overfit. We're specifically trying to reduce variance in this setting." - [[L20-boosting-2]]
- **Tree depth controls interaction order.** $J - 1$ splits ⇒ at most $J - 1$ variables in any path ⇒ at most $(J-1)$-way interactions are modeled. So $J = 2$ (stumps) → purely additive; $J = 3$ → up to 2-way interactions; $J = 6$ → up to 5-way. The Elements rule of thumb $4 \le J \le 8$ corresponds to "allow up to 3- to 7-way interactions."
- **Same depth across all trees is the modern norm.** Historically people grew big trees and pruned, but
  > "they realized it didn't help as much as they wanted." - [[L20-boosting-2]]
- **$\nu$ is regularization**: it's the boosting analog of ridge's $\lambda$, the lasso's $\lambda$, the smoothing spline's $\lambda$, and the NN learning rate (where it doubles as both optimization step size and implicit regularizer). Smaller $\nu$ ⇒ more regularization ⇒ more trees needed.
- **Why this is different from random forests.** With RF, you can keep adding trees almost for free, variance reduction asymptotes but doesn't go negative. With boosting, residuals eventually become noise; additional trees overfit. So $M$ and $\nu$ are real, coupled tuning parameters here, not "use enough."

## Worked example: Boston via gbm() (slide deck)

Two runs with $M = 5000$ trees: one at default $\nu = 0.001$, one at $\nu = 0.2$. Test MSE comparable on Boston ("doesn't make a big difference") because $M$ is plenty big in both cases. The point of the comparison: with enough $M$, the practical effect of $\nu$ on the final fit is small; the *training-time* and the *minimum-CV-error iteration* shift dramatically.

## Exam signals

> "We need to specify the number of trees, the depth, and the shrinkage. How do you determine good values?" - [[L27-summary]]

The prof's answer: cross-validation. Fix $\nu$ small (e.g. 0.1), pick $M$ at the CV minimum, sanity-check depth against bias-variance reasoning.

> "You should be prepared to think about and discuss in like an exam setting for example: like why would we want the trees not to be too deep?" - [[L20-boosting-2]]

The bias-variance answer the prof gave himself: shallow trees are weak learners that take small, precise steps; deep trees overfit per-step and break the additive correction logic.

> "It's funny when you can only get the result that the people show if you set the seed equal to like 1, 2, 3 because it shouldn't matter. You should get something very similar if you don't set the seed." - [[L20-boosting-2]]

(Honest skepticism: be wary of cherry-picked $M$ / $\nu$ choices that work only with a specific random seed.)

## Pitfalls

- **Treating $M$ and $\nu$ independently.** They're coupled. The standard recipe is fix $\nu$ then tune $M$, not the other way around.
- **Setting $\nu = 1$.** That kills the "small step" property, each tree contributes its full predictions, ensemble jumps to a near-perfect training fit instantly, no diversity, no benefit over a single bigger tree.
- **Growing deep trees in boosting.** This is the random-forest reflex transferred to the wrong setting. RF wants high-variance/low-bias trees because it averages them. Boosting wants high-bias/low-variance weak learners because it sums their corrections.
- **Forgetting to do early stopping.** With small $\nu$, you don't know in advance what $M$ should be; running for fixed $M = 1000$ may overfit or underfit. Use `cv.folds` (in `gbm()`) or `early_stopping_rounds` (in xgboost) to find the optimal $M$.
- **Confusing $\nu$ in boosting with $\eta$ in stochastic GBM.** The prof uses $\eta$ for both the learning rate (in some lectures) and for the row-subsample fraction (Friedman 2002 notation). Here $\nu$ = learning rate, $\eta$ depends on context; check the deck.
- **Ignoring the seed.** Stochastic-flavored boosting (subsampled rows / cols) can give visibly different results across seeds, set one and report it.

## Scope vs ISLP

- **In scope:** the *concept* of weak learners (small trees), the role of tree depth as interaction-order control, the empirical rule $4 \le J \le 8$, the role of $\nu$ as a step-size / shrinkage regularizer, the rule $\nu \le 0.1$, the $M$–$\nu$ coupling, the early-stopping recipe, the connection to gradient descent.
- **Look up in ISLP:** §8.2.3 spells out shrinkage $\lambda$ (book's notation for what the slides call $\nu$) and tree depth $d$, pp. 343–347. The empirical rule and the deeper "interaction order = depth" framing live in Elements ch. 10 (reference, not exam material).
- **Skip in ISLP (book-only, prof excluded):**
  - **Detailed pseudocode of where exactly $\nu$ multiplies in**: [[L27-summary]] / [[L20-boosting-2]]: concept matters, line-by-line doesn't.
  - **Heavy theory of step-size choice / convergence rates**: out per the prof's "no fancy proofs" comment.

## Exercise instances

- Exercise9.3: explain the learning rate $\nu$: what does it mean, how would one choose it, where does it enter Algorithm 10.3 (modify step 2(d)). The expected discussion: $\nu$ shrinks each tree's contribution → forces weak / slow learning → smaller $\nu$ requires larger $M$ → typically pick $\nu \le 0.1$ and tune $M$ by early stopping on a validation set.

## How it might appear on the exam

- **Conceptual short-answer**: "why do we want trees in boosting to be weak?" Expected answer: weak learners take small steps that don't overshoot the gradient direction; many weak learners summed reduces bias without overfitting; deep trees individually overfit and break the additive-correction logic.
- **Hyperparameter explanation (à la 2025 Q6c)**: "how would you choose the learning rate and the number of trees?" Expected answer: fix $\nu$ small (≈ 0.1), then choose $M$ at the CV minimum / via early stopping; never tune $M$ alone with $\nu = 1$.
- **True/false**: "in boosting, increasing $\nu$ requires fewer trees" (true), "$\nu$ controls the depth of each tree" (false), "the trees in gradient boosting should be deep so they don't underfit" (false, opposite).
- **Where-does-it-enter**: given Algorithm 10.3 without $\nu$, modify the update step to incorporate $\nu$ (Exercise 9.3 verbatim).
- **Bias-variance reasoning**: given a training-error / CV-error vs. $M$ plot, identify the optimal $M$ and explain the U using "additional trees fit residual noise = variance increases past the minimum."

## Related

- [[boosting]]: the parent concept; weak learners and $\nu$ are the two regularizers that make boosting work.
- [[gradient-boosting]]: where $\nu$ enters explicitly (Algorithm 10.3 step 2(d)).
- [[adaboost]]: historically used full-strength stumps with no $\nu$; modern AdaBoost variants add a learning rate.
- [[stochastic-gradient-boosting]]: the third regularizer (subsampling), on top of weak learners + $\nu$.
- [[xgboost]]: adds L1/L2 leaf-weight penalties and dropout to the regularization stack.
- [[regularization]]: $\nu$ is just shrinkage in another guise; the prof groups it with ridge/lasso $\lambda$, smoothing-spline $\lambda$, NN weight decay.
- [[bias-variance-tradeoff]]: weak learners reduce bias via accumulation; $\nu$ trades a bit of training-fit speed for variance reduction.
- [[cross-validation]]: the standard tool for picking $M$ given a fixed $\nu$.
