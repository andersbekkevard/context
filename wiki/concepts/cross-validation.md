---
concept: cross-validation
modules: [04-classif, 05-resample, 06-modelsel, 07-beyondlinear, 08-trees, 09-boosting, 11-nnet]
lectures: [L09, L10, L11, L12, L13, L14, L15, L16, L18, L19, L20, L23, L24, L26, L27]
isl-ref: 5.1
exercises:
  - Exercise5.1 — describe k-fold CV with figure, aggregation, regression and classification examples
  - Exercise5.2 — k-fold vs validation-set vs LOOCV (bias / variance / compute trade-offs)
  - Exercise5.3 — wrong-vs-right way to do CV with feature pre-selection (p=5000 noise example)
  - Exercise6.3b — best-subset selection scored by 10-fold CV
  - Exercise8.2c — cv.tree() for choosing pruning size for a regression tree
  - Exercise8.3e — cv.tree with prune.misclass for classification spam tree
  - Exercise9.4a — gbm() with cv.folds for boosting
  - Exercise9.4c — xgb.cv on simulated data
  - CE1 problem 4a — write 10-fold CV pseudocode for KNN regression with MSE
  - CE1 problem 4b — true/false comparing CV variants (LOOCV vs k-fold bias/variance, val-set as 2-fold CV, LOOCV as bootstrapping)
related: [bias-variance-tradeoff, regularization, k-fold-cv, leave-one-out-cv, validation-set-approach, one-standard-error-rule, nested-cv-and-cv-pitfalls, bootstrap, aic-bic-conceptual]
tags:
  - concept
  - specials
aliases:
  - CV
  - k-fold cross-validation
  - cross validation
---

# Cross-validation

The prof's preferred way to choose any hyperparameter, full stop. He distrusts AIC/BIC/Cp because their assumptions "are not typically right"; CV makes fewer assumptions and works for any model. **k-fold with k=5 or 10** is the standard. The course returns to it in every module after module 5 — it's the universal hyperparameter knob, the universal generalization estimator, and the universal "are you doing this right?" diagnostic.

## Definition (prof's framing)

> "I almost always in almost everything I do, I use cross-validation of some sort… because assumptions are always wrong, right? They're just so wrong." — [[L12-modelsel-1]]

> "These things are used a lot and they're nice because they don't make so many assumptions." — [[L10-resample-1]]

The basic recipe ([[L10-resample-1]]):

1. Partition the training data into $k$ roughly-equal folds.
2. For $j = 1, \dots, k$: train on the $k-1$ other folds, evaluate on fold $j$, get a per-fold error $\text{MSE}_j$ (or $\text{Err}_j$ for classification).
3. Average to get $\text{CV}_k = \frac{1}{n} \sum_j n_j\, \text{MSE}_j$.

Used for **model selection** (which $\lambda$, which $K$, which polynomial degree, which tree depth) and **model assessment** (honest generalization estimate). When you want both at the same time, use [[nested-cv-and-cv-pitfalls|nested CV]].

## Returns in other modules

- [[L09-classif-3]] — the QDA-vs-LDA test-error split (LDA train 0.19, test 0.17; QDA train 0.17, test 0.32) is the bridge to module 5: *"In general, in many problems, you want to use the simplest possible thing that works."* CV is the honest tool for that comparison.
- [[L10-resample-1]] — full module-5 introduction: validation-set approach, [[leave-one-out-cv|LOOCV]] (with the OLS hat-matrix shortcut $\text{CV}_n = \tfrac{1}{n}\sum ((y_i - \hat y_i)/(1-h_{ii}))^2$), [[k-fold-cv|k-fold CV]], the [[one-standard-error-rule]]. The headline pitfall: **independence violations** — *"two points right next to each other, one in your training, one in your validation, it's the same damn thing."* Chunk by the dependency dimension first, then split.
- [[L11-resample-2]] — recap framed via [[bias-variance-tradeoff]]: validation set = high bias, low variance; LOOCV = low bias, high variance (folds nearly identical → correlated estimates → variance of average blows up); k=5/10 is the practical sweet spot. **[[nested-cv-and-cv-pitfalls|Nested CV]]** for combined selection + assessment, then the **right vs wrong way** trap (preselect predictors using y → CV gives ≈0% misclassification on pure noise).
- [[L12-modelsel-1]] — used to pick across $M_0, M_1, \dots, M_p$ in best-subset and stepwise selection. Prof flags "I really don't think I'm going to ask any questions about" AIC/BIC/Cp formulas — CV is the trusted alternative.
- [[L13-modelsel-2]] — picks $\lambda$ for ridge and lasso. *"We don't typically know what the bias and the variance are. These are just things we think about. If we knew what the bias was, we wouldn't be fitting a model."* CV implicitly captures the bias-variance balance via held-out error.
- [[L14-modelsel-3]] — three-step recipe: cross-validate over a grid of $\lambda$ → refit on all data with chosen $\lambda$ → optionally drop zeroed-out coefs from lasso and refit unpenalized.
- [[L15-modelsel-4]] — picks $M$ for PCR (and PLS); CV-MSE curve dips then climbs as you add components that explain $X$ but not $Y$.
- [[L16-beyondlinear-1]] — picks $\lambda$ for smoothing splines via **leave-one-out CV** specifically (the book's recommendation). Effective df = trace($S$) lets you specify smoothness as a non-integer df instead of $\lambda$.
- [[L18-trees-2]] — picks $\alpha$ for [[cost-complexity-pruning]]. The pruned-tree size at the CV minimum is the chosen complexity. `cv.tree(..., FUN=prune.misclass)` does it for classification trees.
- [[L19-boosting-1]] — for bagging/RF: **B is "use enough"** (not really tuned, no CV needed); OOB error is a free CV-equivalent. For boosting later: $M$ is a real hyperparameter and CV early stopping picks it.
- [[L20-boosting-2]] — early stopping on a validation curve picks $M$ for boosting; *"You want to monitor your prediction in held-out data, and then say, oh, well, now we're done."* `gbm()`'s `cv.folds` argument does this.
- [[L23-nnet-1]] — implicit in choosing every NN hyperparameter (number of layers, M hidden units, $\lambda$ for L1/L2, dropout rate, learning rate). The prof flags **overfitting your validation set** as a real risk if you tweak too many things ([[L24-nnet-2]]).
- [[L24-nnet-2]] — early stopping = "stop training when validation error starts climbing." Uses CV / a validation set to pick the stopping epoch. Same machinery applied to a different hyperparameter (epoch count).
- [[L26-nnet-3]] — picks $\lambda$ in the lasso comparison on Hitters; the CV-chosen lasso (12 vars, MAE 0.50) beats unregularized linear (0.56) and an unregularized 1049-param NN. Also: *"we're essentially always going back to the test set to determine our model complexity"* in the [[double-descent]] discussion.
- [[L27-summary]] — Q6b walkthrough: "given the CV curve, how would you pick $\lambda$?" → smallest CV error (or one-SE simpler). Q6c: "We need to specify the number of trees, depth, shrinkage. How do you determine good values?" → cross-validation.

## Notation & setup

- $n$ = total training-data size; $k$ = number of folds; $n_j$ = size of fold $j$ (≈ $n/k$).
- $C_j$ = the index set of fold $j$.
- $\hat f^{(-j)}$ = model fit on all folds except $j$.
- $\text{MSE}_j = \frac{1}{n_j} \sum_{i \in C_j} (y_i - \hat f^{(-j)}(x_i))^2$ for regression; misclassification rate for classification.
- $\text{CV}_k = \frac{1}{n} \sum_{j=1}^k n_j \, \text{MSE}_j$ (weighted by fold size).
- $\widehat{\text{SE}}(\text{CV}_k) = \sqrt{\sum_j (\text{MSE}_j - \overline{\text{MSE}})^2 / (k-1)}$ — the per-fold sample SD.

For LOOCV: $k = n$, each fold has 1 observation, $\text{CV}_n = \frac{1}{n} \sum_i (y_i - \hat f^{(-i)}(x_i))^2$.

## Formula(s) to know cold

### k-fold CV for regression

$$\boxed{\;\text{CV}_k = \frac{1}{n} \sum_{j=1}^k n_j \, \text{MSE}_j, \qquad \text{MSE}_j = \frac{1}{n_j} \sum_{i \in C_j} (y_i - \hat y_i^{(-j)})^2\;}$$

### k-fold CV for classification

$$\text{CV}_k = \frac{1}{n} \sum_{j=1}^k n_j \cdot \frac{1}{n_j}\sum_{i \in C_j} \mathbb{1}(y_i \neq \hat y_i^{(-j)})$$

### LOOCV shortcut for OLS

$$\boxed{\;\text{CV}_n = \frac{1}{n} \sum_{i=1}^n \left( \frac{y_i - \hat y_i}{1 - h_{ii}} \right)^2\;}$$

where $h_{ii}$ is the diagonal of the hat matrix $H = X(X^\top X)^{-1} X^\top$ and $\hat y_i$ is the **full-data** fitted value. **One fit, no refits.** Comes back in CE1 problem 4 territory.

### One-standard-error rule

Pick the simplest model whose CV error is within one SE of the minimum:

$$\text{CV}(\theta) \leq \text{CV}(\hat\theta) + \widehat{\text{SE}}(\text{CV}_k(\hat\theta))$$

[[L10-resample-1]] flags it as "not quite valid" because you reuse the held-out data to estimate both the mean and the SE, but it's still the prof's preferred selection criterion.

## Insights & mental models

### CV variants compared (the prof's headline table)

| Method | k | Bias of CV estimate | Variance | Compute |
|---|---|---|---|---|
| Validation set | 1 split | High (uses ½ of data) | High (depends on split) | Cheap (1 fit) |
| LOOCV | $n$ | **Lowest** (uses $n-1$ each fold) | **High** (folds highly correlated) | Expensive ($n$ fits, except OLS shortcut) |
| 5-fold | 5 | Slight upward bias | **Lower than LOOCV** | 5 fits |
| 10-fold | 10 | Tiny upward bias | Low | 10 fits |

> "Leave one out will have a better bias. But the K-fold will likely have a much better variance. And typically in this setting, you're winning by having less variance — because you want to know, you want to pick a model that will do well if you have another data set, and it's that variance across data sets that you really want to reduce." — [[L11-resample-2]]

The **variance-of-LOOCV** intuition: training sets differ by only one observation → per-fold estimates highly correlated → variance of their average has a $\rho\sigma^2$ floor that doesn't shrink (same algebra as bagged trees in [[L19-boosting-1]]).

### Why we need CV at all

> "Your assumptions have to be right. And they're not. They're not typically right." — [[L10-resample-1]]

[[aic-bic-conceptual|AIC/BIC/Cp]] derive penalties for training error under specific (often violated) distributional assumptions. CV asks the same question with **fewer assumptions** — just "how well does this hyperparameter generalize on held-out data from this dataset?"

### Three jobs, three partitions

[[L10-resample-1]] / [[training-validation-test-split]]:
- **Training** — fit the model.
- **Validation** — select among candidate models / pick hyperparameters.
- **Test** — report final performance.

> "We will be too optimistic if we report the error on the test set when we have already used it to choose the best model… Don't do that. I have lots of examples of where people have done that. It's very sad. It's very common. It kind of sucks." — [[L10-resample-1]]

CV does the validation job efficiently inside the training/validation split. The test set must remain untouched until the end (or use [[nested-cv-and-cv-pitfalls|nested CV]]).

### Pseudocode for k-fold (CE1 problem 4a target)

```
Input: data {(x_i, y_i)}_{i=1}^n, model class M, hyperparameter θ, k folds.
1. Randomly partition {1,...,n} into k disjoint folds C_1, ..., C_k of size ≈ n/k.
2. For j = 1, ..., k:
     Fit model M(θ) on data {(x_i, y_i) : i ∉ C_j}.
     Compute MSE_j = (1/|C_j|) Σ_{i ∈ C_j} (y_i - ŷ_i)².
3. Return CV_k(θ) = (1/n) Σ_j |C_j| · MSE_j.
```

To pick $\theta$: repeat for each candidate $\theta$ on a grid; take $\arg\min_\theta \text{CV}_k(\theta)$ (or apply the one-SE rule). Then **refit on all training data** (no fold held out) using the chosen $\theta$.

### The independence trap (recurring exam-bait)

> "Be careful with independence — independence of points. For example, let's say your data is spatial in nature. There's a natural correlation between things spatially, and this also happens temporally in time… If you just randomly sort them into the test and fit and validation without any concern of these things, it's going to suck. Because you're going to take two points that are right next to each other, but one in your training data, one in your validation data — it's the same damn thing." — [[L10-resample-1]]

Mitigation: chunk by the dependency dimension (whole spatial blocks, whole time windows) before splitting. The prof's neuro-data trick: throw away 80–90% of nearby time bins to enforce independence between fold boundaries.

LOOCV is **terrible** under temporal/spatial correlation because the held-out point is essentially predicted by its near-neighbor → fold error is artificially tiny → CV picks the most complex model possible. ([[L10-resample-1]])

### The wrong-way / right-way trap (Exercise 5.3)

> "If you preselect predictors using $y$ (e.g., correlation filter), step 1 is part of training and **must** be inside the CV loop. Doing it outside can give misclassification ≈ 0 on pure noise." — [[L11-resample-2]]

ISL §5.1.4 example: $p=5000$ noise predictors, $n=50$ random labels. Filter to top-25 correlated with $y$ (uses labels!), then do CV on logistic-regression-of-25. CV says ≈0% misclassification because the filter has already "peeked." The right way: redo the filter inside each training fold, on training data only.

### Why the [[one-standard-error-rule]] is preferred

The CV-minimum is itself a noisy estimator. Many models within ±1 SE of the minimum are statistically indistinguishable. Pick the **simplest** of them — better for interpretability, more conservative for variance.

> [!note] Verbatim qualifier
> "Strictly speaking, this estimate is not quite valid. Why?" — [[L10-resample-1]] (slide footnote, prof leaves as thought question)
> Answer: the SE is computed on the same fold errors used to pick the minimum, so it's not an independent SE estimate.

## Exam signals

> "I almost always in almost everything I do, I use cross-validation of some sort… because assumptions are always wrong, right?" — [[L12-modelsel-1]]

> "These things are used a lot and they're nice because they don't make so many assumptions." — [[L10-resample-1]]

> "Using the test set for both model selection and estimation tends to overfit the test data, and the bias will be underestimated." — [[L11-resample-2]]

> "We would maybe try this multiple times — break it into threes and try it three times — and then compare them more rigorously." — [[L09-classif-3]] (foreshadowing of the LDA-vs-QDA decision via CV)

> "We need to specify the number of trees, the depth, and the shrinkage. How do you determine good values?" — answer: **cross-validation** ([[L27-summary]] Q6c).

> "How would you pick $\lambda$ here?" — answer: smallest CV error (or one-SE simpler) ([[L27-summary]] Q6b).

## Pitfalls

- **Wrong-way CV**: any preprocessing that uses the response (correlation filter, supervised PCA, label-aware feature engineering) **must** live inside the CV loop. Doing it once outside gives 0% error on pure noise. [[L11-resample-2]] flags this as "lying with statistics."
- **Independence violations**: random splits leak information across folds when data is spatially/temporally/relationally correlated. Pre-chunk by the dependency dimension.
- **Reusing test for selection**: the validation/test distinction is non-negotiable. Use [[nested-cv-and-cv-pitfalls|nested CV]] when you need both.
- **LOOCV's high variance**: training sets differ by only one observation → per-fold estimates correlate strongly → average has high variance. k-fold (k=5/10) is the recommended compromise.
- **5-fold CV is NOT lower-bias than LOOCV** (CE1 4b (i) is FALSE — it's the *opposite*: 5-fold has more bias because each fold trains on less data).
- **The validation-set approach is NOT 2-fold CV** (CE1 4b (iii) is FALSE — in 2-fold CV you swap roles and average; the val-set approach trains once on one half).
- **LOOCV is NOT bootstrapping** (CE1 4b (iv) is FALSE — LOOCV partitions without replacement; bootstrap samples with replacement).
- **CV-chosen complexity is the test-MSE *minimum*, not the true bias-variance optimum.** The true optimum is unobservable; CV is an estimate. The one-SE rule is the corrective.
- **Reusing the validation set too many times** ([[L24-nnet-2]] flag): if you tweak hyperparameters by repeatedly checking validation error, you've effectively trained on it. Defense: nested CV, or hold out a fresh test set.
- **Don't compare CV errors across models with completely different scales of loss** — the loss has to be the same metric.

## Scope vs ISLR

- **In scope:** validation set, LOOCV, k-fold (definitions, pros/cons, bias-variance trade), choosing hyperparameters via CV, the OLS LOOCV shortcut, the one-SE rule, [[nested-cv-and-cv-pitfalls|nested CV]] for selection + assessment, the wrong-vs-right way trap, the independence-violation pitfall.
- **Look up in ISLR:** §5.1 (full CV chapter, §5.1.1–5.1.5); §5.1.4 specifically for the bias-variance trade in CV variants.
- **Skip in ISLR (book-only, prof excluded):** AIC/BIC/Cp algebra and derivations — *"I really don't think I'm going to ask any questions about this."* — [[L12-modelsel-1]]. Concept stays in scope (penalize complexity); formulas don't.

## Exercise instances

- **Exercise5.1** — describe the k-fold CV algorithm with a figure, aggregation, examples for both regression (polynomial / KNN) and classification (polynomial / KNN).
- **Exercise5.2** — compare k-fold to validation-set and LOOCV in terms of bias, variance, and compute.
- **Exercise5.3** — simulate $p=5000$ random predictors with random labels; show wrong-way CV gives ~0% misclass while right-way gives ~50% (the truth).
- **Exercise6.3b** — wrap CV around best-subset selection on Credit; pick the best model size.
- **Exercise8.2c** — `cv.tree()` to find the optimal pruning size for a regression tree on Carseats.
- **Exercise8.3e** — `cv.tree()` with `prune.misclass` for a classification tree on the spam data.
- **Exercise9.4a** — `gbm()` with `cv.folds=10` to fit a basic gradient boosting model on simulated genomic data.
- **Exercise9.4c** — `xgb.cv` on the same data.
- **CE1 problem 4a** — write 10-fold CV pseudocode for KNN regression with MSE.
- **CE1 problem 4b** — true/false comparing CV variants (see Pitfalls).

## How it might appear on the exam

- **Pseudocode question**: write k-fold CV from scratch for a stated model. Math, plain English, or pseudocode all OK ([[L27-summary]] format note). Include the inner loop, the aggregation, and what you'd return.
- **Read a CV plot**: given $\text{CV}_k(\lambda)$ (or $\text{CV}_k(K)$ for KNN, or $\text{CV}_k(\text{tree size})$), pick the best hyperparameter and explain. Bonus: apply the one-SE rule.
- **Direction-of-effect T/F**: LOOCV bias < k-fold bias (TRUE), LOOCV variance < k-fold variance (FALSE), val-set = 2-fold CV (FALSE), LOOCV = bootstrap (FALSE), $\hat\sigma^2$ from training underestimates test error (TRUE).
- **Why CV over AIC/BIC?** — fewer assumptions, works for any model class, trustworthy when distributional assumptions don't hold (the prof's running answer).
- **The wrong-way trap** — given a description of a feature-selection-then-CV pipeline, identify the leak. Explain that selection must be redone inside each fold.
- **The independence pitfall** — given temporal/spatial data, explain why naive CV underestimates test error and how you'd fix it (block by dependency, leave gaps).
- **OLS LOOCV shortcut** — given the hat matrix's diagonal, compute the LOOCV error in one shot. Cite the formula $\text{CV}_n = \frac{1}{n}\sum ((y_i - \hat y_i)/(1-h_{ii}))^2$.
- **CV used inside a larger pipeline** — picking $\alpha$ in tree pruning (Q5), $\lambda$ in ridge/lasso (Q6b), $M$ in PCR, hyperparameters in boosting (Q6c) — all canonical exam set-ups.
- **Method comparison via CV**: given the CV-MSE numbers for two models, decide which to use. The 2025 Q6b walkthrough has the lasso CV beating unregularized linear by 0.02 — the prof's answer is *"the reduction in variance is not offset by the increase in bias, so don't add a regularizer."*

## Related

- [[bias-variance-tradeoff]] — CV is how we estimate the test MSE that the bias-variance decomposition predicts; the variance of CV estimators is itself a bias-variance argument
- [[regularization]] — CV is *the* way to choose the strength of any regularizer ($\lambda$ in ridge/lasso, $\alpha$ in tree pruning, dropout rate, weight decay)
- [[k-fold-cv]] — the workhorse variant
- [[leave-one-out-cv]] — the special case k=n with the OLS shortcut
- [[validation-set-approach]] — the simplest, used when you have lots of data
- [[one-standard-error-rule]] — the prof's preferred selection criterion
- [[nested-cv-and-cv-pitfalls]] — combined model selection + assessment, plus the wrong-way trap
- [[bootstrap]] — sister resampling method; bootstrap aims at sampling distributions, CV at generalization error
- [[aic-bic-conceptual]] — the alternative the prof distrusts (assumptions don't hold)
- [[out-of-bag-error]] — a free CV-equivalent for bagging/RF (each bootstrap sample's ~1/3 left-out points serve as that tree's validation set)
- [[double-descent]] — even past the interpolation point, CV is still the right tool for picking the operating regime
