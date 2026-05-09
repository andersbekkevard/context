---
concept: adaboost
module: 09-boosting
lectures: [L19, L20]
isl-ref: 8.2.3
exercises: []
related: [boosting, gradient-boosting, weak-learner-and-learning-rate, boosting-loss-functions, classification-tree, bias-variance-tradeoff]
tags:
  - concept
  - module/09-boosting
aliases:
  - AdaBoost.M1
  - adaptive boosting
---

# AdaBoost

The first and historically founding boosting algorithm: a sequence of weak binary classifiers fit on **re-weighted** training data, with misclassified points getting boosted weight in the next round. The prof uses it as the on-ramp into the general gradient-boosting story , it turned out (five years after the fact) to be the special case of forward stagewise additive modeling under exponential loss.

## Definition (prof's framing)

> "Each of these little bad predictors is biased towards whatever the worst shit is , wherever the model sucks the most, it says, OK, I'm going to try there." - [[L19-boosting-1]]

The prof's higher-level summary on the second pass:

> "Adaboost used the idea of bagging trees and the benefit of using multiple trees and taking these things and averaging them together , but now improving on the randomization that random forest tries to do, to generate a more diverse set of models that will generalize better because they're individually very simple." - [[L19-boosting-1]]

Setting: binary classification with $y \in \{-1, +1\}$. Each weak classifier $G_m(x)$ is a small tree (often a stump, a one-split tree) , though the recipe works for any classifier. Final prediction = sign of an $\alpha$-weighted vote.

## Notation & setup

- $y_i \in \{-1, +1\}$: labels coded ±1 (not 0/1, unlike logistic regression).
- $G_m(x)$: the $m$-th weak classifier, output ±1.
- $w_i$: weight on training observation $i$. Initialized to $1/N$.
- $\text{err}_m$: weighted misclassification rate of $G_m$.
- $\alpha_m$: classifier weight, computed from $\text{err}_m$.
- $M$: number of boosting rounds.

## Formula(s) to know cold

The full algorithm (Algorithm 10.1 in Elements; the prof's full board-derivation in [[L19-boosting-1]]):

Initialize $w_i = 1/N$. For $m = 1, \dots, M$:

1. Fit $G_m(x)$ to the data with weights $w_i$.
2. Compute weighted error rate
   $$\text{err}_m = \frac{\sum_i w_i \cdot \mathbb{1}[y_i \neq G_m(x_i)]}{\sum_i w_i}.$$
3. Compute classifier weight
   $$\alpha_m = \log\frac{1 - \text{err}_m}{\text{err}_m}.$$
4. Update sample weights:
   $$w_i \leftarrow w_i \cdot \exp\big(\alpha_m \cdot \mathbb{1}[y_i \neq G_m(x_i)]\big).$$

Final classifier:
$$G(x) = \operatorname{sign}\!\Big(\sum_{m=1}^M \alpha_m G_m(x)\Big).$$

Two structural details to memorize:

- **Misclassified** ($\mathbb{1}=1$): $w_i$ is multiplied by $e^{\alpha_m}$, weight goes up.
- **Correct** ($\mathbb{1}=0$): $w_i$ is multiplied by $e^0 = 1$, unchanged.

So persistently-misclassified points keep accumulating weight, dominating later fits.

The exponential-loss connection (the punchline that makes AdaBoost a special case of gradient boosting):
$$L(y, f) = \exp(-y\,f(x)).$$
Same sign → $-yf$ very negative → $\exp(-yf) \approx 0$ (good). Opposite sign → $-yf$ large → $\exp(-yf)$ blows up.

> "It's really going to penalize this shit out of misclassified points." - [[L20-boosting-2]]

## Insights & mental models

- **Re-weighting samples is the key trick**, distinct from gradient boosting (which fits the gradient of a loss instead of re-weighting samples). Two very different mechanical recipes that turn out to be the same general scheme.
- **Why the exponential update.**
  > "If it was just the indicator function, it would just be a step function , that's boring. So this gives us nice smooth things." - [[L19-boosting-1]]
- **Why** $\alpha_m = \log\frac{1-\text{err}_m}{\text{err}_m}$. When $\text{err}_m < 0.5$, $\alpha_m > 0$, good classifiers get positive weight. When $\text{err}_m > 0.5$, $\alpha_m < 0$, the classifier is *worse* than random and gets a *negative* weight, effectively flipping its prediction in the final vote (as it should).
- **Stumps work surprisingly well.** The trees are "essentially a bias term, very, very small" ([[L19-boosting-1]]). The bias-reduction-via-many-weak-learners angle.
- **Discovered as gradient boosting under exponential loss only later.**
  > "These two models came out… in different articles and no one really realized that it was the same thing until like five years later. So they kind of kicked themselves for not realizing it sooner. Most things are very obvious after the fact." - [[L20-boosting-2]]

## Worked example , synthetic 10-D Gaussian (slide 18 of the deck)

10-dim multivariate-normal features, label $= +1$ if $\sum X_j^2 > 9.34$, else $-1$. ~50/50 split. AdaBoost test error stays bad through ~5 iterations (the average is dominated by a handful of stumps), then drops sharply, plateauing around iteration 90.

## Worked example , Boston-data classification

> "Random forest on the Boston data: error rate 0.17. AdaBoost: 0.1. Quite a striking difference." - [[L19-boosting-1]]

So AdaBoost beats the previous winner ([[random-forest]]) by a meaningful margin on this dataset.

## Exam signals

> "I won't have you memorize the names of the R functions, of course, but you should know what tree boosting is." - [[L27-summary]]

The prof did **not** flag AdaBoost as having its own dedicated exam question, but it's covered in slides + lectures (so in scope per the rule), and it's the canonical illustration of the bagging-vs-boosting contrast.

## Pitfalls

- **Coding $y$ as $\{0,1\}$ instead of $\{-1,+1\}$.** AdaBoost's exponential-loss formulation uses ±1; the update rules and the final $\operatorname{sign}(\cdot)$ depend on this.
- **Confusing weight-of-classifier ($\alpha_m$) with weight-of-observation ($w_i$).** Both are called "weights" but live on opposite sides, observations get re-weighted across iterations; classifiers get weighted in the final vote.
- **Forgetting to normalize weights.** In practice, weights are renormalized to sum to 1 after each update; the algorithm-as-stated drops this step but `ada()` does it under the hood.
- **Treating AdaBoost as parallel.** It's sequential , round $m+1$ depends on which points $G_m$ missed.

## Scope vs ISLR

- **In scope:** the algorithm structure (sequential weak classifiers, sample re-weighting, $\alpha$-weighted voting), the role of ±1 labels and the sign of the sum, the exponential-loss connection, why stumps work, the Boston-data result vs. RF.
- **Look up in ISLR:** §8.2.3 covers boosting generally but does **not** spell out AdaBoost as its own algorithm. For the explicit AdaBoost.M1 algorithm and the equivalence-with-exponential-loss derivation, the prof points to **Elements of Statistical Learning ch. 10.1** (not on the exam table , this is reference, not study material).
- **Skip in ISLR (book-only):**
  - The full proof that AdaBoost = forward stagewise additive modeling under exponential loss , [[L20-boosting-2]] notes the result, doesn't derive it.

## Exercise instances

None. AdaBoost has no dedicated exercise problem , the recommended Exercise 9 set is built around gradient boosting (`gbm()`, h2o, xgboost). Anders's exposure is purely lectures + slides.

## How it might appear on the exam

- **Multiple choice / fill-in-the-blank**: "AdaBoost re-weights __________ in each iteration" (the data / observations).
- **True/false**: "AdaBoost is a special case of gradient boosting under exponential loss" (true), "AdaBoost trees are grown independently" (false).
- **Conceptual short-answer**: explain what happens to misclassified points across rounds (their weight grows, pulling later trees' attention onto them).
- **Method-comparison**: given AdaBoost test error vs. random forest test error, which performed better and which mechanism explains the gap (re-weighting + sequential fitting → bias reduction beyond what RF's variance reduction offers).

## Related

- [[boosting]]: the general framework AdaBoost is a special case of.
- [[gradient-boosting]]: the modern unified algorithm; AdaBoost falls out under exponential loss.
- [[weak-learner-and-learning-rate]]: stumps as the canonical weak learner.
- [[boosting-loss-functions]]: exponential loss is what AdaBoost minimizes implicitly.
- [[classification-tree]]: the building block for AdaBoost when applied to trees.
- [[bias-variance-tradeoff]]: many weak learners → low bias via summation, low variance because each is too small to overfit.
