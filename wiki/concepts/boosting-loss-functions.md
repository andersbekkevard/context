---
concept: boosting-loss-functions
module: 09-boosting
lectures: [L20]
isl-ref: 8.2.3
exercises: []
related: [boosting, gradient-boosting, adaboost, logistic-regression, residual-diagnostics, regression-tree, classification-tree, bias-variance-tradeoff]
tags:
  - concept
  - module/09-boosting
aliases:
  - boosting losses
  - quadratic loss
  - absolute loss
  - Huber loss
  - binomial deviance
  - multinomial deviance
  - exponential loss
---

# Boosting loss functions (quadratic / absolute / Huber / deviance)

The whole point of [[gradient-boosting|gradient boosting]] generalizing beyond the regression-residuals special case: **any differentiable loss plugs into Algorithm 10.3**. The prof walks through the standard menu , quadratic (default but outlier-sensitive), absolute (robust but discontinuous), Huber (the smooth compromise), binomial deviance (binary classification), multinomial deviance ($K$-class , fits $K$ trees per round) , and the exponential loss that AdaBoost implicitly minimizes.

## Definition (prof's framing)

> "A lot of this kind of algorithmic work, which is kind of where machine learning fits in, is coming up with different loss functions that have the behaviors that you want… there's a lot of optimization theory to it and in many ways a lot of art." - [[L20-boosting-2]]

Headline: pick a loss whose gradient describes the kind of error you care about, plug it into Algorithm 10.3 step 2(a) , that's it. The same boosting machinery now optimizes whatever loss you wrote down.

## The menu (in scope)

### Quadratic / squared-error loss (regression default)

$$L(y, f) = \tfrac12 (y - f)^2.$$

- **Negative gradient:** $-\partial L / \partial f = y - f$ , the **residual**. This is why fitting residuals = gradient boosting in the squared-error case.
- **Pro:** smooth, simple, what `distribution = "gaussian"` means in `gbm()`.
- **Con:** very sensitive to outliers (squaring a big residual blows up).
- **Con:** doesn't sparsify , small residuals stay small forever.

### Absolute loss (robust regression)

$$L(y, f) = |y - f|.$$

- **Negative gradient:** $-\partial L / \partial f = \operatorname{sign}(y - f) \in \{-1, +1\}$ , the residual *sign*, not magnitude.
- **Pro:** outliers hurt much less (linear penalty, not quadratic).
- **Pro/con:** V-shape at 0 → pushes small residuals to *exactly* zero → sparse fits. Could be good (clean) or bad (kills the gradient signal that boosting wants).
- **Con:** gradient discontinuous at 0 , "it's annoying" ([[L20-boosting-2]]).

### Huber loss (the smooth compromise)

$$L(y, f) = \begin{cases} (y - f)^2 & |y - f| \le \delta, \\ 2\delta|y - f| - \delta^2 & \text{otherwise.} \end{cases}$$

- Quadratic near zero (smooth, no V, doesn't aggressively zero things).
- Linear far out (doesn't blow up on outliers).
- Smooth pieces stitched together (continuous gradient if $\delta$ chosen well).
  > "What it combines is it eliminates the quadratic thing for big numbers." - [[L20-boosting-2]]
- Practical reading from the slide: Huber (green curve) is "the part of the squared loss here where it doesn't push everything to zero" + linear tails.

### Exponential loss (binary classification , implicitly used by AdaBoost)

$$L(y, f) = \exp(-y\,f(x)), \qquad y \in \{-1, +1\}.$$

- Same sign $y$ and $f$ → $-yf$ very negative → $\exp(-yf) \approx 0$ , good.
- Opposite sign → $-yf$ large positive → $\exp(-yf)$ blows up , heavy penalty.
  > "It's really going to penalize this shit out of misclassified points." - [[L20-boosting-2]]
- Plugging exponential loss into the forward stagewise additive modeling scheme **gives AdaBoost** , discovered five years after the fact. See [[adaboost]].

### Binomial deviance (binary classification , used by gradient boosting)

For $y \in \{0, 1\}$:
$$L(y, f) = -\mathbb{1}(y = 1)\log p(x) - \mathbb{1}(y = 0)\log(1 - p(x)),$$
with $p(x) = e^{f(x)}/(1 + e^{f(x)})$ , the same logistic / sigmoid as [[logistic-regression]].

- Same shape as logistic regression's negative log-likelihood.
- Plug into Algorithm 10.3, take the gradient, you have a binary-classifier boosting algorithm.
- This is what `distribution = "bernoulli"` means in `gbm()`.

### Multinomial deviance ($K$-class classification)

$$L(y, f) = -\sum_{k=1}^K \mathbb{1}(y = k) \log p_k(x), \qquad p_k(x) = \frac{e^{f_k(x)}}{\sum_{l=1}^K e^{f_l(x)}}$$

, the latter is the **softmax**. The $f_k(x)$ are per-class predictor functions, each represented by its own tree ensemble.

- **Negative gradient (per class $k$):**
  $$-g_{ikm} = \mathbb{1}(y_i = k) - p_k(x_i).$$
  > "Indicator minus the class probability prediction, which ends up sounding very similar to residuals, right? Conceptually similar, which it should be because it is related to the gradient of the whole thing." - [[L20-boosting-2]]
- **Critical practical detail:** for $K$ classes, you build **$K$ trees per boosting round** , one per class, each fit to its own class-specific gradient $\mathbf{g}_{km}$. In step 3 the class probabilities are aggregated across the trees.

## Insights & mental models

- **Gradient = "residual generalized."** For squared error, the gradient is the residual. For binomial deviance, it's "indicator minus probability." For exponential loss, it's a re-weighting of the data. Same algorithm, same step structure , just different gradients.
- **Loss design is engineering.**
  > "A lot of this kind of algorithmic work… is coming up with different loss functions that have the behaviors that you want… in many ways a lot of art." - [[L20-boosting-2]]
- **Robustness vs. sparsity trade-off in the regression menu.** Squared = smooth + outlier-sensitive + non-sparse. Absolute = outlier-robust + sparse + nondifferentiable at 0. Huber = the engineered middle path.
- **Multi-class costs $K$× per round.** Often forgotten on exams , the gradient per class $k$ is $\mathbb{1}(y = k) - p_k(x)$, you fit $K$ separate trees per iteration, not 1.
- **Binomial deviance ≈ logistic regression's loss.** Same negative log-likelihood, same sigmoid link. The boosting algorithm just replaces "linear in $\beta$" with "sum of trees" in the predictor function.

## Exam signals

> "If it was covered either in the slides or in the exercises, then I would say fair game." - [[L27-summary]]

Boosting losses are in the slides + lecture L20, so they're in scope. The prof did not promise a dedicated exam question on losses, but the conceptual menu (when to pick which loss) is exactly the kind of multi-select / fill-in-the-blank he likes.

## Pitfalls

- **Forgetting that for $K$-class classification you fit $K$ trees per round.** The most likely T/F trap.
- **Using squared-error loss with heavy outliers.** Squared loss "really penalizes things that are far away" , your fit bends to chase the noise.
  > "If you have a number that's like 100, 100 squared is really big. So it really penalizes things that are far away." - [[L20-boosting-2]]
- **Coding $y \in \{0, 1\}$ for exponential loss / AdaBoost.** Exponential loss assumes $y \in \{-1, +1\}$. Logistic / binomial deviance assumes $y \in \{0, 1\}$. Don't mix the conventions.
- **Confusing "deviance" with "exponential loss".** Binomial deviance = logistic NLL (gradient boosting). Exponential loss = AdaBoost's implicit objective. Both are binary-classification losses, but they're not the same function.
- **Forgetting Huber's $\delta$ as a hyperparameter.** Pick badly and you lose the smooth-stitch property; the slides note "stitches together smoothly if you pick $\delta$ well."

## Scope vs ISLP

- **In scope:** the menu (quadratic, absolute, Huber, binomial deviance, multinomial deviance, exponential loss); the gradient for squared error = residual; the gradient for multinomial deviance = $\mathbb{1}(y_i = k) - p_k(x_i)$; "fits $K$ trees per round for $K$-class"; the robust-vs-non-robust intuition; the connection to AdaBoost via exponential loss; the connection to logistic regression via binomial deviance.
- **Look up in ISLP:** §8.2.3 covers the squared-error case only; for the deeper general loss treatment, the slide deck refers to **Elements of Statistical Learning ch. 10** (reference, not exam material). Anders does **not** need this for the exam.
- **Skip in ISLP (book-only / out of scope):**
  - **Detailed Hessian computations for each loss**: out per the prof's "no fancy proofs."
  - **Advanced robust losses (Tukey biweight, etc.)**: never lectured.
  - **Quantile loss for quantile regression boosting**: not lectured.

## Exercise instances

None. No Exercise 9 problem is built directly around loss-function selection , the recommended exercises drill `gbm()` and `xgboost()` mechanics with the default `distribution = "gaussian"`. The losses themselves are pure lecture / slide content.

## How it might appear on the exam

- **Multiple choice**: "the negative gradient of the squared-error loss equals __________" (the residual, $y - f(x)$).
- **True/false**: "for $K$-class gradient boosting you fit one tree per round" (false , $K$ trees per round), "Huber loss is more robust to outliers than squared error" (true), "absolute loss has a continuous gradient at zero" (false).
- **Loss-selection short-answer**: "you have a regression dataset with three obvious outliers. Which boosting loss would you choose?" Expected: Huber (or absolute) , squared loss would over-weight the outliers.
- **Conceptual link-back**: "what loss does AdaBoost implicitly minimize?" (Exponential, $L(y, f) = \exp(-yf)$.) "What loss does `distribution = 'bernoulli'` in `gbm()` use?" (Binomial deviance , same as logistic regression.)
- **Mathy-ish**: derive the negative gradient of binomial deviance and observe it equals $y - p(x)$.

## Related

- [[boosting]]: the parent framework; loss is one of the three ingredients.
- [[gradient-boosting]]: the algorithm into which any of these losses plugs.
- [[adaboost]]: the special case for exponential loss (binary $\pm 1$).
- [[logistic-regression]]: same binomial deviance / sigmoid story, but predictor function is linear instead of a tree ensemble.
- [[residual-diagnostics]]: squared-error's outlier-sensitivity is what the residual-vs-fitted plot warns you about in OLS-land; same intuition transfers.
- [[regression-tree]] / [[classification-tree]] , the building blocks that get fit to each gradient.
- [[bias-variance-tradeoff]]: robust losses (Huber, absolute) trade a bit of efficiency for robustness; same trade-off lens.
