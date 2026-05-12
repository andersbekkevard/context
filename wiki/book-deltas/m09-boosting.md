---
title: "M09: Boosting and Additive Trees — Book delta"
module: 09-boosting
isl-ch: 8
lectures: [L19, L20, L21]
---

# Module 09: Boosting and Additive Trees — Book delta

Module 09 maps to **ISLP §8.2.3 Boosting** in `wiki/book/08-trees.md`. That section is **deliberately thin**: ISLP gives only the *regression-trees with squared-error residuals* recipe (Algorithm 8.2), three tuning parameters ($B$, $\lambda$, $d$), a gene-expression worked example, and the explicit disclaimer that *"boosting classification trees proceeds in a similar but slightly more complex way, and the details are omitted here."* Everything boosting-specific the prof teaches **beyond** that 2-page treatment is delta and reproduced here.

> "If it was covered either in the slides or in the exercises, then I would say fair game." — [[L27-summary]]

What ISLP §8.2.3 contains (NOT delta):

- Algorithm 8.2, the squared-error residual-fitting recipe with shrinkage $\lambda$ for regression trees.
- The three tuning parameters $B$, $\lambda$, $d$ at high level.
- The gene-expression worked example.
- The "learn slowly" / overfit-if-$B$-too-large warning.
- ISLP §8.2.1 covers variable importance for *bagging* (RSS / Gini decrease); §8.2.3 does not extend it to boosting.
- BART (§8.2.4) is not lectured.

What ISLP §8.2.3 does NOT contain (= delta below):

1. **AdaBoost.M1** as a stand-alone algorithm.
2. The **exponential-loss / forward-stagewise-additive-modeling** equivalence (AdaBoost-as-special-case).
3. The **general gradient boosting algorithm** (any differentiable loss).
4. The **functional-gradient-descent / steepest-descent in function space** framing.
5. The **loss menu** (quadratic, absolute, Huber, exponential, binomial deviance, multinomial deviance) and their gradients.
6. **Classification gradient boosting** ($K$-trees-per-round multinomial extension).
7. **XGBoost** — second-order gradients, leaf-weight L1/L2, $\gamma$-pruning, dropout, parallelization.
8. **Stochastic gradient boosting** — row and column subsampling.
9. **Boosting-specific variable importance** (extension of §8.2.1).
10. **Partial dependence plots** as a tree-ensemble interpretability tool.
11. The **"three ingredients / four hyperparameters"** rubric.
12. The **$M$–$\nu$ coupling recipe**.
13. The **interaction-order = depth** rule for $J$.

---

## 1. AdaBoost.M1 (binary classification)

[[L19-boosting-1|L19]], [[adaboost]]

ISLP §8.2.3 mentions classification only to say "the details are omitted here." [[adaboost|AdaBoost]] itself is not in ISLP. The prof teaches it as the **historical on-ramp** into the [[boosting]] story and as the canonical illustration of [[bagging]]-vs-boosting contrast.

### 1.1 Setting

- Binary classification with $y_i \in \{-1, +1\}$ (**not** $\{0,1\}$ — the sign-rule story depends on $\pm 1$ coding).
- Each weak classifier $G_m(x) \in \{-1, +1\}$ — typically a tree stump (one split).
- Sample weights $w_i$ initialized uniformly.

### 1.2 Algorithm (the prof's full statement)

Initialize $w_i = 1/N$ for $i = 1, \ldots, N$. For $m = 1, 2, \ldots, M$:

1. **Fit** classifier $G_m(x)$ to the training data using weights $w_i$.
2. **Weighted error**
$$
\mathrm{err}_m \;=\; \frac{\sum_{i=1}^{N} w_i \cdot \mathbb{1}[y_i \neq G_m(x_i)]}{\sum_{i=1}^{N} w_i}.
$$
3. **Classifier weight**
$$
\alpha_m \;=\; \log \frac{1 - \mathrm{err}_m}{\mathrm{err}_m}.
$$
4. **Update sample weights**
$$
w_i \;\leftarrow\; w_i \cdot \exp\!\big( \alpha_m \cdot \mathbb{1}[y_i \neq G_m(x_i)] \big), \qquad i = 1, \ldots, N.
$$

Output the final classifier
$$
G(x) \;=\; \operatorname{sign}\!\left( \sum_{m=1}^{M} \alpha_m \, G_m(x) \right).
$$

### 1.3 Structural facts to memorize

These are the exam-flagged consequences of the algorithm form:

- **Misclassified** ($\mathbb{1}=1$): $w_i$ multiplied by $e^{\alpha_m}$ → weight increases.
- **Correctly classified** ($\mathbb{1}=0$): $w_i$ multiplied by $e^0 = 1$ → weight unchanged.
- Persistently-missed points accumulate weight, dominating the next fit.
- When $\mathrm{err}_m < 1/2$: $\alpha_m > 0$, classifier votes "with" the ensemble.
- When $\mathrm{err}_m > 1/2$: $\alpha_m < 0$, classifier vote is **flipped** in the final sum — a worse-than-random classifier is sanely auto-inverted.
- When $\mathrm{err}_m = 1/2$: $\alpha_m = 0$, the classifier contributes nothing.
- In implementations, the $w_i$ are typically renormalized to sum to 1 after each step; the algorithm-as-stated drops this step but it doesn't change the output (only the scale).

> "Each of these little bad predictors is biased towards whatever the worst shit is. Wherever the model sucks the most, it says, OK, I'm going to try there." — [[L19-boosting-1|L19]]

> "If it was just the indicator function, it would just be a step function — that's boring. So this gives us nice smooth things." — [[L19-boosting-1|L19]]

### 1.4 Why stumps work

The "[[weak-learner-and-learning-rate|weak learners]] with no overfitting room" angle: stumps are individually so dumb they have near-zero variance, but their **sum** captures the structure via the re-weighting machinery — many shallow corrections, each focused on the previous round's failures.

> "Adaboost used the idea of bagging trees and the benefit of using multiple trees and taking these things and averaging them together — but now improving on the randomization that [[random-forest|random forest]] tries to do, to generate a more diverse set of models that will generalize better because they're individually very simple." — [[L19-boosting-1|L19]]

---

## 2. AdaBoost ≡ forward-stagewise additive modeling under exponential loss

[[L20-boosting-2|L20]], [[adaboost]], [[boosting-loss-functions]]

This is the unification result that ISLP completely skips. The prof states it but does not derive it on the board; he flags the historical "kicked themselves for not realizing sooner" framing. The reproduction below is the standard outline Anders should be able to recognize at the exam — full algebra is in ESL §10.4, not required.

### 2.1 The forward-stagewise additive modeling (FSAM) scheme

Seek a model of the form
$$
f(x) \;=\; \sum_{m=1}^{M} \beta_m \, b(x; \gamma_m),
$$
i.e. an expansion in [[basis-functions|basis functions]] $b(\cdot;\gamma)$ (for trees, $b(x;\Theta) = T(x;\Theta) = \sum_j \gamma_j \, \mathbb{1}(x \in R_j)$). Fit it **one term at a time**, never revisiting earlier terms: at step $m$ solve
$$
(\beta_m, \gamma_m) \;=\; \arg\min_{\beta,\gamma} \sum_{i=1}^{N} L\!\left(y_i,\; f_{m-1}(x_i) + \beta\, b(x_i;\gamma)\right),
$$
and update $f_m(x) = f_{m-1}(x) + \beta_m\, b(x;\gamma_m)$.

### 2.2 The equivalence statement

**Claim.** Running FSAM with the **exponential loss**
$$
L(y, f) \;=\; \exp\!\big( -y \, f(x) \big), \qquad y \in \{-1, +1\}
$$
**produces exactly AdaBoost.M1**: the per-step minimization separates into (i) fitting a weighted classifier and (ii) computing a classifier weight, and the resulting weight is $\beta_m = \tfrac12 \log\frac{1-\mathrm{err}_m}{\mathrm{err}_m}$; the implicit re-weighting that emerges from the residual $\exp(-y_i f_{m-1}(x_i))$ is exactly the AdaBoost update with $\alpha_m = 2\beta_m$.

(The factor-of-2 discrepancy is purely notational: the final classifier is $\operatorname{sign}(\sum_m \beta_m G_m)$, and scaling all $\beta_m$ by 2 doesn't change the sign. The prof's lecture / Algorithm 10.1 writes $\alpha_m = \log\frac{1-\mathrm{err}_m}{\mathrm{err}_m}$ directly, absorbing the factor.)

### 2.3 Why exponential loss

Geometric reading of $L(y,f) = e^{-yf}$:

- $y, f$ same sign → $-yf$ large negative → $e^{-yf} \approx 0$ (good, no penalty).
- $y, f$ opposite sign → $-yf$ large positive → $e^{-yf}$ blows up (heavy penalty).
- Smooth and differentiable everywhere (unlike $\mathbb{1}[y \neq \hat y]$).

> "It's really going to penalize this shit out of misclassified points." — [[L20-boosting-2|L20]]

> "These two models came out… in different articles and no one really realized that it was the same thing until like five years later. So they kind of kicked themselves for not realizing it sooner." — [[L20-boosting-2|L20]]

### 2.4 The classifier-weight derivation (the part the prof did write down)

The mini-derivation in [[L20-boosting-2|L20]]: at iteration $m$, fix $f_{m-1}$. Define weights $w_i^{(m)} = \exp(-y_i f_{m-1}(x_i))$. The FSAM objective separates:
$$
\sum_i w_i^{(m)} \exp(-y_i \beta G_m(x_i))
\;=\; e^{-\beta} \sum_{y_i = G_m(x_i)} w_i^{(m)} \;+\; e^{\beta} \sum_{y_i \neq G_m(x_i)} w_i^{(m)}.
$$
Minimizing over $\beta$ (set derivative to zero):
$$
\beta_m \;=\; \tfrac{1}{2}\log \frac{1 - \mathrm{err}_m}{\mathrm{err}_m},
$$
which is AdaBoost's $\alpha_m$ up to the factor of 2. The re-weighting $w_i^{(m+1)} = w_i^{(m)} \exp(-y_i \beta_m G_m(x_i))$ recovers the AdaBoost weight update.

---

## 3. The general gradient tree boosting algorithm (Algorithm 10.3)

[[L20-boosting-2|L20]], [[gradient-boosting]]

ISLP §8.2.3 gives only Algorithm 8.2 (squared-error residuals). The general algorithm for any differentiable loss is **not** in ISLP. The prof presents Algorithm 10.3 of ESL as the unified scheme and uses it as the framework for the entire loss menu in §4.

### 3.1 Setup

- Training data $\{(x_i, y_i)\}_{i=1}^{N}$.
- Loss $L(y, f)$, **differentiable** in its second argument.
- Tree size $J$ (number of terminal nodes), learning rate $\nu \in (0,1]$, number of iterations $M$.

### 3.2 Algorithm

**Step 1 (initialize).** Choose the constant minimizer
$$
f_0(x) \;=\; \arg\min_{\gamma} \sum_{i=1}^{N} L(y_i, \gamma).
$$

**Step 2.** For $m = 1, 2, \ldots, M$:

- **(a)** Compute the **negative-gradient pseudo-residuals**
$$
r_{im} \;=\; -\!\left[ \frac{\partial L\!\big(y_i, f(x_i)\big)}{\partial f(x_i)} \right]_{f = f_{m-1}}, \qquad i = 1, \ldots, N.
$$
- **(b)** Fit a regression tree to the targets $\{r_{im}\}_{i=1}^N$ **by least squares**, producing terminal regions $R_{jm}$ for $j = 1, \ldots, J_m$.
- **(c)** For each region $j = 1, \ldots, J_m$, compute the leaf value from the **original** loss
$$
\gamma_{jm} \;=\; \arg\min_{\gamma} \sum_{x_i \in R_{jm}} L\!\big( y_i,\; f_{m-1}(x_i) + \gamma \big).
$$
- **(d)** Update the model with shrinkage
$$
f_m(x) \;=\; f_{m-1}(x) \;+\; \nu \cdot \sum_{j=1}^{J_m} \gamma_{jm}\, \mathbb{1}(x \in R_{jm}).
$$

**Step 3 (output).** $\hat f(x) = f_M(x)$.

### 3.3 The "two sub-problems decoupled" framing

The prof's load-bearing structural point: step 2(b) and step 2(c) solve **different** problems.

> "We're going to find the cute function we can minimize to find these regions, right, these R-tildes, and then we're going to use those to find the gammas." — [[L20-boosting-2|L20]]

- **Regions** $R_{jm}$ are found via the **smooth surrogate** problem: minimize squared error to the negative gradient. This is "the cute function we can minimize" — a least-squares [[regression-tree|regression tree]] fit, which is computationally tractable.
- **Leaf values** $\gamma_{jm}$ are then computed from the **original** loss $L$, per region. For squared error this is the leaf mean; for binomial deviance it's a logit-shrunk quantity, etc.

This is why the algorithm is called "gradient tree boosting" rather than "gradient boosting on the original loss" — gradients are used only to *steer the tree partition*, then the original loss is restored for the leaf values.

### 3.4 Functional gradient descent (the steepest-descent framing)

ISLP gives the residual-fitting recipe with no theoretical scaffold. The prof's framing is **[[gradient-boosting|gradient boosting]] = steepest descent in function space**:

- In parameter space, [[gradient-descent-and-sgd|gradient descent]] says: $\theta \leftarrow \theta - \rho \, \nabla_\theta L$.
- In function space, the analog is: $f \leftarrow f - \rho \, g$ where $g_{im} = \partial L / \partial f(x_i)$ is the "gradient of the loss at observation $i$ with respect to the function value at $x_i$."
- But $f$ must live in the space of sums-of-trees, so we **project** the descent direction onto that space by fitting a tree to $-g_m$ via least squares.

$$
\tilde \Theta_m \;=\; \arg\min_{\Theta} \sum_{i=1}^{N} \big(\, -g_{im} - T(x_i; \Theta) \,\big)^{2}.
$$

> "By updating according to the residuals, what really we were doing is actually fitting the gradient of the function that we're optimizing. We're estimating the direction we have to go in." — [[L19-boosting-1|L19]]

> "The main conceptual idea is that basically when we were boosting the stuff for regression we were using the residuals which seemed like a hack that someone figured out. It was actually estimating the gradient. And so more generally you can just say, I let R be the gradient or the negative gradient, and then fit a tree to that negative gradient." — [[L20-boosting-2|L20]]

### 3.5 The squared-error sanity check (delta)

ISLP gives Algorithm 8.2 (fit a tree to the residuals) as if by direct definition. The prof's mini-derivation that **the residual is the negative gradient** is the bridge to the general algorithm. Take squared-error loss
$$
L(y, f) \;=\; \tfrac{1}{2} (y - f)^2.
$$
Differentiate with respect to $f$ at the current ensemble:
$$
\frac{\partial L}{\partial f(x_i)} \;=\; -(y_i - f(x_i)) \;=\; -r_i.
$$
So
$$
-\left[\frac{\partial L}{\partial f(x_i)}\right]_{f_{m-1}} \;=\; y_i - f_{m-1}(x_i) \;=\; r_i,
$$
**the residual itself.** ISLP's Algorithm 8.2 is Algorithm 10.3 with squared-error plugged into step 2(a) and a regression-tree-mean closed-form for step 2(c).

---

## 4. The loss menu

[[L20-boosting-2|L20]], [[boosting-loss-functions]]

Everything in §4 is delta — ISLP §8.2.3 only treats squared error. The prof gives a full menu and the gradient for each loss is exam-relevant.

### 4.1 Squared-error / Gaussian loss (regression)

$$
L(y, f) \;=\; \tfrac{1}{2}(y - f)^2, \qquad -\frac{\partial L}{\partial f} \;=\; y - f \;=\; r.
$$

- Smooth, differentiable everywhere.
- Outlier-sensitive (squaring big residuals blows up).
- Doesn't sparsify — small residuals stay small.
- The `distribution = "gaussian"` option in `gbm()`.

### 4.2 Absolute / $L_1$ loss (robust regression)

$$
L(y, f) \;=\; |y - f|, \qquad -\frac{\partial L}{\partial f} \;=\; \operatorname{sign}(y - f) \in \{-1, +1\}.
$$

- Gradient is the **sign** of the residual, not its magnitude.
- Outliers hurt linearly, not quadratically → robust.
- V-shape at 0 → discontinuous gradient at 0 ("annoying", [[L20-boosting-2|L20]]).
- Pushes small residuals exactly to zero → sparse fits.

### 4.3 Huber loss

$$
L(y, f) \;=\;
\begin{cases}
(y - f)^2 & |y - f| \le \delta, \\
2 \delta |y - f| - \delta^2 & \text{otherwise.}
\end{cases}
$$

- Quadratic near 0 (smooth, no V, doesn't aggressively zero things).
- Linear in the tails (doesn't blow up on outliers).
- Smooth at the knot if $\delta$ is chosen well.
- The "engineered middle path" between squared and absolute.

### 4.4 Exponential loss (binary classification)

$$
L(y, f) \;=\; \exp(-y\,f), \qquad y \in \{-1, +1\}.
$$

- The implicit loss minimized by AdaBoost (§2).
- Heavy penalty on misclassified points; near-zero on correctly classified.
- Smooth everywhere.

### 4.5 Binomial deviance (binary classification)

For $y \in \{0, 1\}$ with sigmoid link $p(x) = e^{f(x)} / (1 + e^{f(x)})$:
$$
L(y, f) \;=\; -\mathbb{1}(y = 1) \log p(x) \;-\; \mathbb{1}(y = 0) \log(1 - p(x)).
$$

Same negative log-likelihood as [[logistic-regression|logistic regression]]. The negative gradient is the **logistic residual**:
$$
-\frac{\partial L}{\partial f(x_i)} \;=\; y_i - p(x_i).
$$
The `distribution = "bernoulli"` option in `gbm()`. The prof flagged the analogy:

> "Indicator minus the class probability prediction, which ends up sounding very similar to residuals, right? Conceptually similar, which it should be because it is related to the gradient of the whole thing." — [[L20-boosting-2|L20]]

### 4.6 Multinomial deviance ($K$-class classification)

Per-class logit $f_k(x)$, softmax link
$$
p_k(x) \;=\; \frac{e^{f_k(x)}}{\sum_{\ell = 1}^K e^{f_\ell(x)}}.
$$
Loss:
$$
L(y, f) \;=\; -\sum_{k=1}^{K} \mathbb{1}(y = k) \log p_k(x).
$$

**Per-class negative gradient:**
$$
-g_{ikm} \;=\; \mathbb{1}(y_i = k) \;-\; p_k(x_i).
$$

**Critical structural fact (exam trap):** for $K$-class gradient boosting, you fit **$K$ trees per iteration** $m$ — one per class, each fit to its own gradient $\{-g_{ikm}\}_{i=1}^N$. The class probabilities are aggregated by softmax across the $K$ tree ensembles. This is what ISLP's "the details are omitted here" was hiding.

### 4.7 Insight: gradient = "residual generalized"

The unifying observation: for squared error the negative gradient is the residual $y - f$; for binomial deviance it's $y - p$ (the "indicator minus probability"); for exponential loss it's a re-weighting; for multinomial deviance it's the per-class indicator-minus-probability. Same algorithm, same step structure, different gradients.

---

## 5. Hyperparameters: the three-ingredients / four-hyperparameters rubric

[[L20-boosting-2|L20]], [[gradient-boosting]], [[weak-learner-and-learning-rate]]; Exercise 9.2, 9.3

ISLP §8.2.3 lists three tuning parameters ($B$, $\lambda$, $d$) at high level. The prof reorganizes this as **the** explicit exam-flagged rubric. Exercise 9.2 literally asks Anders to explain $L$, $f_m$, $M$, $J_m$ in Algorithm 10.3 — the abstract objects that don't appear in Algorithm 8.2.

### 5.1 Three ingredients of gradient boosting

1. **[[weak-learner-and-learning-rate|Weak learners]]** — small trees with $4 \le J \le 8$ leaves typically; each does barely better than random.
2. **A loss function** $L(y, f)$ — any differentiable choice (§4).
3. **An additive way to combine them** — sum of trees with shrinkage:
$$
f_M(x) \;=\; \sum_{m=1}^{M} \nu \cdot T(x; \Theta_m).
$$

### 5.2 Four hyperparameters

**Tree-level (weak-learner) hyperparameters:**

(a) **Tree depth** $d$ or equivalently number of leaves $J = d + 1$. ISLP's `interaction.depth = d` parameter; off-by-one with $J$. Typical $4 \le J \le 8$. Same depth across all trees in the modern norm; historically people grew big and pruned but "they realized it didn't help as much as they wanted" — [[L20-boosting-2|L20]].

(b) **Minimum observations per terminal node.** So each leaf has enough samples for a stable mean / majority vote. Less critical when $N$ is large.

**Boosting-level (additive-model) hyperparameters:**

(c) **Number of trees** $M$ (= ISLP's $B$). **A real tuning parameter here**, unlike in [[random-forest|RF]]. Too small → underfit; too large → ensemble fits noise.

(d) **Learning rate** $\nu$ (= ISLP's $\lambda$; also written $\eta$). Empirical rule $\nu \le 0.1$. Enters Algorithm 10.3 step 2(d) by multiplying the per-tree update.

### 5.3 The interaction-order rule (depth → interaction order)

ISLP §8.2.3 says briefly: "$d$ is the *interaction depth*, and controls the interaction order of the boosted model, since $d$ splits can involve at most $d$ variables." The prof develops this into a usable rule:

> A tree with $J$ leaves has $J - 1$ splits, so each path from root to leaf can involve at most $J - 1$ distinct variables → the boosted ensemble models interactions of order up to $J - 1$.

Reading:
- $J = 2$ (stumps): purely **additive** model (one variable per tree).
- $J = 3$: up to 2-way interactions.
- $J = 6$: up to 5-way interactions.

This is why stumps fit an additive model (each tree depends on one $x_j$), and why the prof's empirical rule $4 \le J \le 8$ corresponds to "allow up to 3- to 7-way interactions."

### 5.4 The $M$–$\nu$ coupling recipe

ISLP says "very small $\lambda$ can require using a very large value of $B$ in order to achieve good performance" but stops there. The prof turns this into an **explicit procedural recipe**, flagged for Exercise 9.3 and 2025 exam Q6c:

1. **Fix $\nu$ small (≈ 0.1)** by default.
2. **Choose $M$ by early stopping** on a held-out validation set, or by [[cross-validation]] (`cv.folds` in `gbm()`).
3. Sanity-check depth: deep trees overfit per-step and break the additive-correction logic.

> "We need to specify the number of trees, the depth, and the shrinkage. How do you determine good values?" — [[L27-summary|L27]]
> (Expected answer: cross-validation, with $\nu$ fixed small first.)

### 5.5 The "why weak learners" exam discussion question

> "This is like something that you should be prepared to think about and discuss in like an exam setting for example: like why would we want the trees not to be too deep?" — [[L20-boosting-2|L20]]

The canonical answer (the prof's own):

- Each tree should take a **small, controlled step** along the negative gradient. Deep trees overfit the gradient at that step → break the additive-correction logic.
- The [[bias-variance-tradeoff|bias-reduction]] angle: many weak (high-bias, low-variance) learners summed → bias gets aggregated away without inviting variance.
- The steepest-descent analogy: $\nu$ is the step size, weak learners enforce that no single step is too aggressive.

---

## 6. Stochastic gradient boosting

[[L20-boosting-2|L20]], [[stochastic-gradient-boosting]]; Exercise 9.4b

ISLP §8.2.3 doesn't cover this at all. Friedman 2002. Three subsampling variants — **all without replacement**, distinct from [[bagging]]'s [[bootstrap]]:

1. **Subsample rows** before each tree. Parameter `bag.fraction` (gbm), `subsample` (xgboost), `sample_rate` (h2o). Typical $0.5$.
2. **Subsample columns** before each tree. Parameter `colsample_bytree` (xgboost), `col_sample_rate_per_tree` (h2o).
3. **Subsample columns** before each split inside a tree (random-forest-style decorrelation). Parameter `colsample_bynode` (xgboost), `col_sample_rate` (h2o).

Friedman 2002 notation: $\eta$ = row-subsample fraction (distinct from the $\eta$ used for the learning rate in some lectures — context disambiguates).

**Mechanism:** more diversity across trees → less correlation between predictions → lower ensemble variance. Same logic as [[random-forest|RF]]'s correlated-trees formula, applied iteration-by-iteration inside boosting. Bonus: less data per tree = faster training.

The prof's three-item structure makes this a clean exam target: *"name three [[regularization]] strategies in gradient boosting"* → small $\nu$ + early stopping on $M$ + stochastic subsampling.

> "Subsampling here is without replacement, not bootstrapping." — [[L20-boosting-2|L20]] (paraphrased)

---

## 7. XGBoost

[[L20-boosting-2|L20]], [[L21-unsupervised-1|L21]], [[xgboost]]

ISLP §8.2.3 predates the [[xgboost|XGBoost]] framing and does not cover it. Everything below is delta on top of Algorithm 8.2.

### 7.1 Second-order gradients (Newton, not just gradient)

Vanilla gradient boosting (Algorithm 10.3, step 2(a)) is **first-order**: it uses only the gradient. XGBoost uses a **second-order Taylor expansion** of the loss around the current ensemble:
$$
L\!\big(y_i,\; f_{m-1}(x_i) + t\big) \;\approx\; L\!\big(y_i, f_{m-1}(x_i)\big) \;+\; g_{im}\, t \;+\; \tfrac{1}{2}\, h_{im}\, t^{2},
$$
with $g_{im} = \partial L / \partial f$ and $h_{im} = \partial^2 L / \partial f^2$ at $f_{m-1}$. Per-leaf, this gives a Newton-step closed form (modulo regularization, see §7.4 below).

> "Instead of just taking the first-order gradients… the distance we need to go in each direction of each gradient is actually given to us by the second-order information. That gives us our Newton method." — [[L20-boosting-2|L20]]

**Effect:** more accurate per-step descent → larger sensible step sizes → fewer trees needed for the same accuracy. The prof's analogy: vanilla GBM = first-order gradient descent, XGBoost = Newton-Raphson.

### 7.2 Parallelization and approximate split search

- **Parallelization.** Different CPU cores search different candidate splits at each node simultaneously. Same algorithm, parallel execution.
- **Approximate split search.** Instead of enumerating every possible split point for a continuous feature, XGBoost bins the feature into a fixed number of quantiles and only considers bin boundaries. Speed-up at minimal accuracy cost.

The combination is why XGBoost scales to millions of rows where `gbm()` doesn't.

### 7.3 Per-tree complexity penalty $\Omega(f)$

XGBoost augments the per-iteration loss with a tree-complexity regularizer. L2 form ([[ridge-regression|ridge]]-on-leaves):
$$
\Omega(T) \;=\; \gamma\, |T| \;+\; \tfrac{1}{2}\, \lambda\, \|w\|_{2}^{2},
$$
or L1 form ([[lasso]]-on-leaves):
$$
\Omega(T) \;=\; \gamma\, |T| \;+\; \alpha\, \|w\|_{1},
$$
where

- $|T|$ = number of leaves of the tree (so $\gamma |T|$ is a **pruning** penalty — large $\gamma$ → smaller trees),
- $w = (w_1, \ldots, w_{|T|})$ = leaf-value vector,
- $\lambda$, $\alpha$ = the L2 / L1 strengths,
- $\gamma$ = per-leaf penalty.

The full XGBoost objective per iteration is
$$
\sum_{i=1}^N L\!\big(y_i,\; f_{m-1}(x_i) + T(x_i; \Theta_m)\big) \;+\; \Omega(T).
$$

> "All we needed was a cost function, and then we could apply our algorithm." — [[L20-boosting-2|L20]]

This is [[ridge-regression|ridge]]/[[lasso]] from module 06 applied to the leaf-weight vector instead of regression coefficients.

### 7.4 The closed-form leaf weight

With the 2nd-order expansion of §7.1 plus the L2 regularizer of §7.3, the optimal leaf weight for the data falling into leaf $j$ admits a closed form (the ridge-shrunk Newton step):
$$
w_j^{\ast} \;=\; -\, \frac{\sum_{i \in R_j} g_i}{\sum_{i \in R_j} h_i \;+\; \lambda}.
$$

Reading: numerator is the *sum of gradients* in the leaf; denominator is the *sum of Hessians* plus the ridge penalty $\lambda$. Without the Hessian and $\lambda$ (and with squared error so $g_i = -(y_i - f)$), this collapses to the ordinary least-squares leaf mean.

(The full derivation of $w_j^*$ — substituting the Taylor expansion into the per-iteration objective, summing over leaves, minimizing leaf by leaf — is **out of scope** per the prof. Knowing the *form* of the regularizer and the *existence* of the closed-form Newton step is what the exam tests.)

### 7.5 Pruning parameter $\gamma$

The $\gamma$ in $\Omega(T) = \gamma|T| + \cdots$ acts as a **per-split threshold**: after a tentative split, XGBoost computes the gain from splitting; if the gain is below $\gamma$, the split is **rejected**. So $\gamma$ controls tree depth post-hoc, equivalent in spirit to [[cost-complexity-pruning|cost-complexity pruning]] of a single tree, applied per boosting iteration.

### 7.6 Dropout (in boosting, the DART booster)

Hinton's neural-network trick, ported to boosting. During training, randomly drop a subset of the already-fit trees when computing the residuals for the next tree. The next tree then fits residuals **as if the dropped trees weren't there** → "fills the void" → no single early tree dominates the final ensemble.

> "Dropout is a trick that came from Geoffrey Hinton, inspired by the brain. The idea is that you remove part of your model at different iteration steps so that the model doesn't become sensitive to one specific thing." — [[L20-boosting-2|L20]]

Brain-redundancy analogy: smack someone on the head, lose neurons, but you can still think because the brain has redundancy. Dropout forces the same redundancy. Resurfaces in module 11 in its native neural-network setting.

### 7.7 The full menu of XGBoost knobs

For a quick exam-flag reference of what XGBoost adds on top of vanilla GBM:

| Knob | Role | In ISLP §8.2.3? |
|---|---|---|
| 2nd-order gradients | Newton step instead of gradient | No |
| Parallelization | Engineering | No |
| Approximate split search | Engineering | No |
| $\lambda$ (L2 on leaves) | Shrink leaf weights | No |
| $\alpha$ (L1 on leaves) | Sparsify leaf weights | No |
| $\gamma$ (per-leaf penalty) | Prune small-gain splits | No |
| Dropout | Per-tree randomization | No |
| Row + column subsampling | Variance reduction | No |
| $\nu$ (learning rate) | Same as ISLP's $\lambda$ | Yes (as $\lambda$) |
| `max_depth` | Same as ISLP's $d$ | Yes (as $d$) |

> "You can see why people like it. But the key is the same as we talked about. It's all based on these weak learner trees… It's that low variance that really gets you down there." — [[L20-boosting-2|L20]]

---

## 8. Variable importance for boosting

[[L19-boosting-1|L19]], [[L27-summary|L27]], [[variable-importance]]

ISLP §8.2.1 develops [[variable-importance|variable importance]] for **[[bagging]]** (RSS / Gini decrease across splits, averaged over trees). ISLP §8.2.3 does **not** extend it to boosting. The prof teaches the same construction transposed to the boosted ensemble.

**Impurity-based importance** (the standard flavor for boosting):

- **Regression:** total decrease in RSS (8.1) due to splits on $X_j$, **summed across all $M$ boosted trees**, optionally averaged.
- **Classification:** total decrease in Gini index (8.6) due to splits on $X_j$, summed across the boosted ensemble.

The [[out-of-bag-error|OOB]]-permutation flavor (§8.2.1 alternative) requires [[bootstrap]] samples and an OOB notion, which **boosting doesn't have natively** — boosting uses the full data per tree (or a stochastic-GBM row-subsample fraction). In practice, packages report the impurity-based variant for boosting and reserve permutation importance for bagging / [[random-forest|RF]].

> "[Variable] importance plots are fair game, know what they mean and where they come from, but you won't compute them." — [[L27-summary|L27]]

So: the exam-relevant artifact is the **statement** that a boosted ensemble produces a per-variable importance score by aggregating impurity decrease across all $M$ trees, and that you can rank predictors by it.

---

## 9. Partial dependence plots (PDPs)

[[L21-unsupervised-1|L21]], [[L27-summary|L27]], [[partial-dependence-plots]]

ISLP §8.2.3 / §8.2.5 do not develop [[partial-dependence-plots|PDPs]] (the MOC's `isl-ref: 8.2.5` is aspirational — §8.2.5 in this edition is "Summary of Tree Ensemble Methods", not PDPs). PDPs are pure delta against the book file.

### 9.1 The mathematical ideal

$$
f_j(x_j) \;=\; \mathbb{E}_{X_{-j}}\!\left[\, f(x_j, X_{-j}) \,\right]
$$

— marginalize the model over the joint distribution of all other predictors $X_{-j}$, at $X_j = x_j$.

### 9.2 The empirical estimator (the formula to write on the exam)

$$
\bar f_j(x_j) \;=\; \frac{1}{N} \sum_{i=1}^{N} f\!\left( x_j,\; x_{i, -j} \right).
$$

For a chosen sweep of $X_j$ values, evaluate the trained model with $X_j = x_j$ but the other coordinates set to each observation's own values, then average.

### 9.3 Algorithm (pseudocode-friendly)

For a chosen variable $X_j$:

1. Pick a grid of values $x_j^{(1)}, \ldots, x_j^{(K)}$ over the range of $X_j$ (e.g. quantiles of training data).
2. For each grid value $x_j^{(k)}$:
   - For each training observation $i = 1, \ldots, N$, replace the $j$-th coordinate with $x_j^{(k)}$ keeping $x_{i,-j}$ as is.
   - Compute $\hat y_i^{(k)} = \hat f(\text{modified row})$.
   - Average: $\bar f_j(x_j^{(k)}) = \frac{1}{N} \sum_i \hat y_i^{(k)}$.
3. Plot $\bar f_j(x_j^{(k)})$ versus $x_j^{(k)}$.

### 9.4 Reading rules

- **Marginal effect, not causal effect.** PDPs show how the *prediction* changes with $X_j$ "after accounting for" the other variables, averaging over their joint distribution. Not "what would happen if $X_j$ changed and nothing else did."

> "The partial dependence function represents the effects of $X_j$ on $f(X)$ after accounting for the effects of all the other variables. And that's of course different than… what's the effect of $X$ if everything else was not there. That's a totally different question." — [[L21-unsupervised-1|L21]]

- **Fragility under correlation.** Averaging over the data with $X_j$ pinned can create implausible synthetic observations.

> "You might be looking at a tiny, tiny house with like 50 bedrooms, which doesn't make any sense." — [[L21-unsupervised-1|L21]]

- **Pairs naturally with [[variable-importance|variable importance]].** Importance tells you *which* variables matter; PDPs tell you *how* the top variables matter.

---

## 10. Worked-example test errors

These specific numerical results are quoted in lectures and are useful pattern-matches for the exam, but the underlying datasets are not used in ISLP's boosting section (ISLP §8.2.3 uses the 15-class gene-expression data; the prof uses Boston, Ames, and a synthetic 10-D Gaussian).

### 10.1 Boston classification

| Method | Test error |
|---|---|
| Single tree | ≈ 0.145 |
| [[bagging|Bagging]] (500 trees, $m=p$) | 0.15 |
| [[random-forest|Random forest]] ($m = \sqrt{p}$) | "lowest seen so far" |
| [[adaboost|AdaBoost]] | 0.10 |

> "Random forest on the Boston data: error rate 0.17. AdaBoost: 0.1. Quite a striking difference." — [[L19-boosting-1|L19]]

The prof's headline result for the [[bias-variance-tradeoff|bias-reduction]] angle (AdaBoost beats RF by a meaningful margin).

### 10.2 Ames housing (regression)

| Configuration | RMSE |
|---|---|
| `gbm()` default (`bag.fraction = 1`) | ≈ $22{,}600 |
| Stochastic GBM (`bag.fraction = 0.5`) | ≈ $22{,}400 |
| XGBoost (`eta = 0.1`, `lambda = 0.01`, `max_depth = 3`, `subsample = 0.8`, `colsample_bytree = 0.5`) | ≈ $20{,}000 |

### 10.3 Synthetic 10-D Gaussian (AdaBoost)

Data: 10-dim [[multivariate-normal|multivariate-normal]] features, label $= +1$ if $\sum X_j^2 > 9.34$, else $-1$. AdaBoost test error stays bad through ~5 iterations (the average is dominated by a handful of stumps), then drops steeply, plateauing around iteration 90.

---

## Notation and naming differences

These are notational drifts where the prof's lecture / slide notation differs from ISLP §8.2.3. Same object, different label.

| Object | Prof / wiki | ISLP §8.2.3 | Note |
|---|---|---|---|
| Number of boosting iterations | $M$ (concepts), $b$ (in [[L19-boosting-1|L19]] following Algorithm 8.2) | $B$ | Same object. |
| Shrinkage / learning rate | $\nu$ (concepts, slides) or $\eta$ (some lectures; also Friedman 2002 for row-subsample fraction — context disambiguates) | $\lambda$ (eq. 8.10) | ISLP's $\lambda$ collides with XGBoost's $\lambda$ (L2-on-leaves) and with ridge/lasso's $\lambda$ from module 6 — context required. |
| Tree complexity | $J$ (number of leaves) or $J_m$ | $d$ (number of splits) | $J = d + 1$ off-by-one. ISLP's `interaction.depth = d`. The prof switches between $J$ and $d$ within the same lecture. |
| AdaBoost classifier weight | $\alpha_m = \log\frac{1-\mathrm{err}_m}{\mathrm{err}_m}$ | (not in ISLP) | The FSAM derivation in §2.4 above gives $\beta_m = \tfrac12 \log\frac{1-\mathrm{err}_m}{\mathrm{err}_m}$; the factor of 2 is absorbed at the sign-rule output, so $\alpha_m = 2\beta_m$ in the algorithm statement. Same object. |
| Loss name `gaussian` | `distribution = "gaussian"` in `gbm()` slides | "squared-error loss" $L(y,f) = \tfrac12 (y - f)^2$ | Package-name shorthand for the same loss. |
| Loss name `bernoulli` | `distribution = "bernoulli"` in `gbm()` slides | (binomial deviance, not in ISLP §8.2.3) | Same NLL of the logistic / sigmoid model from logistic regression. |
| Row-subsample fraction | `bag.fraction` (gbm), `subsample` (xgboost), `sample_rate` (h2o), $\eta$ (Friedman 2002) | (not in ISLP §8.2.3) | Same object, four package names + one Greek letter. |
| Per-leaf L2 strength (XGBoost) | $\lambda$ | (not in ISLP §8.2.3) | Collides with ISLP's shrinkage $\lambda$ — context required. |
| Per-leaf L1 strength (XGBoost) | $\alpha$ | (not in ISLP §8.2.3) | Symbol collides with AdaBoost's classifier weight $\alpha_m$ — context required. |
| XGBoost pruning rate | $\gamma$ | (not in ISLP §8.2.3) | Symbol collides with the leaf-value $\gamma_{jm}$ in Algorithm 10.3 — same Greek letter, two objects. |
| Iteration index | $m$ (concepts, Algorithm 10.3) or $b$ (Algorithm 8.2 reuse) | $b$ throughout | Trivial. |
