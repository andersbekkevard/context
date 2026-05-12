# Module 5 Book-Delta: Resampling

Exam-time lookup reference for concrete artifacts that the prof taught in module 5 but that **are not cleanly findable in ISLP Chapter 5**. Each item is fully reproduced; cite back as `[L10]`, `[L11]`, `[concept: ...]`, `[slides]`.

The mapped ISLP chapter (`wiki/book/05-resample.md`) already contains: LOOCV formula (eq. 5.1), LOOCV OLS shortcut (eq. 5.2), $k$-fold CV (eq. 5.3), classification CV (eq. 5.4), the $\alpha$ minimum-variance bootstrap example (eq. 5.6–5.7), the bootstrap SE formula (eq. 5.8), and the $(1-1/n)^n$ derivation in exercise 5.2. Those are **not duplicated** below.

---

## 1. Standard error of the CV estimate

**Not in ISLP Ch. 5.** ISLP states "[[k-fold-cv|k-fold CV]] has lower variance than [[leave-one-out-cv|LOOCV]]" but never writes the formula for $\widehat{\mathrm{SE}}(\mathrm{CV}_k)$. The slide deck does, and the prof reproduces it; it is the only way to apply the [[one-standard-error-rule|one-standard-error rule]].

For a given hyperparameter $\theta$ and a $k$-fold split with per-fold errors $\mathrm{MSE}_1(\theta),\dots,\mathrm{MSE}_k(\theta)$ and average $\overline{\mathrm{MSE}}(\theta) = \frac{1}{k}\sum_j \mathrm{MSE}_j(\theta)$:

$$
\boxed{\;\widehat{\mathrm{SE}}\!\bigl(\mathrm{CV}_{(k)}(\theta)\bigr) \;=\; \sqrt{\frac{1}{k-1}\sum_{j=1}^{k}\bigl(\mathrm{MSE}_j(\theta) - \overline{\mathrm{MSE}}(\theta)\bigr)^{2}}\;}
$$

This is the sample standard deviation of the per-fold MSEs (the slide deck's formula prints without the inner square; the prof's lecture and the canonical form both square the residual inside the sum, as written above). [slides; [[L10-resample-1|L10]]; concept: [[k-fold-cv]]]

> [!note] Why "not quite valid"
> Slide-deck footnote, prof flagged as a thought question: *"Strictly speaking, this estimate is not quite valid. Why?"* — the same held-out folds are used to pick the minimum and to estimate the SE, so the SE is not an independent estimate of CV-error variability. [[[L10-resample-1|L10]]; concept: [[one-standard-error-rule]]]

---

## 2. The one-standard-error rule

**Not in ISLP Ch. 5.** ISLP introduces it briefly in §6.1.3 in the model-selection chapter, not in chapter 5. For module 5 it is a delta: the prof's preferred selection criterion, defined in the resampling lecture.

**Definition.** Given a $k$-fold CV curve $\mathrm{CV}(\theta)$ with associated SE $\widehat{\mathrm{SE}}\bigl(\mathrm{CV}(\theta)\bigr)$:

1. Find $\hat\theta = \arg\min_\theta \mathrm{CV}(\theta)$.
2. Walk $\theta$ in the direction of "simpler" (lower polynomial degree, larger $K$ in KNN, larger $\lambda$ in ridge/lasso, fewer terminal nodes in trees, fewer predictors in subset selection) while the bound

$$\mathrm{CV}(\theta) \;\le\; \mathrm{CV}(\hat\theta) + \widehat{\mathrm{SE}}\bigl(\mathrm{CV}(\hat\theta)\bigr)$$

still holds.
3. The simplest $\theta$ still satisfying the bound is $\theta^{*}$:

$$
\boxed{\;\theta^{*} \;=\; \min\Big\{\theta \text{ simpler than } \hat\theta \;:\; \mathrm{CV}(\theta) \le \mathrm{CV}(\hat\theta) + \widehat{\mathrm{SE}}\bigl(\mathrm{CV}(\hat\theta)\bigr)\Big\}\;}
$$

[slides; [[L10-resample-1|L10]]; concept: [[one-standard-error-rule]]]

---

## 3. Variance-of-a-sum identity (the LOOCV-variance argument)

**Not in ISLP Ch. 5.** ISLP §5.1.4 gives the *intuition* "highly correlated outputs → high variance of the average" but does not write the underlying identity. The slide deck prints it explicitly to motivate [[leave-one-out-cv|LOOCV]]'s high variance, and the prof uses it in L10.

For real-valued random variables $X_1,\dots,X_n$ and constants $a_1,\dots,a_n$:

$$
\mathrm{Var}\!\Big(\sum_{i=1}^{n}a_i X_i\Big) \;=\; \sum_{i=1}^{n}\sum_{j=1}^{n} a_i a_j\,\mathrm{Cov}(X_i,X_j)
$$

$$
\boxed{\;=\; \sum_{i=1}^{n} a_i^{2}\,\mathrm{Var}(X_i) \;+\; 2\sum_{i=2}^{n}\sum_{j=1}^{i-1} a_i a_j\,\mathrm{Cov}(X_i,X_j)\;}
$$

**Application to LOOCV:** $X_i = \mathrm{MSE}_i$, $a_i = 1/n$. The $n$ training sets differ by only a single observation, so pairwise $\mathrm{Cov}(\mathrm{MSE}_i,\mathrm{MSE}_j)$ is large and positive. The cross-covariance term dominates, and the variance of $\mathrm{CV}_{(n)} = \frac{1}{n}\sum_i \mathrm{MSE}_i$ does not collapse to $\sigma^2/n$ — it is large. In $k$-fold CV with $k\!<\!n$, two training sets share only $(k-2)/k$ of the data (60% for $k=5$), so the cross-covariance term is much smaller. [slides; [[L10-resample-1|L10]]; concept: [[leave-one-out-cv]]]

---

## 4. Weighted $k$-fold CV formula (slide form)

**Differs from ISLP.** ISLP §5.1.3 prints $\mathrm{CV}_{(k)} = \frac{1}{k}\sum_{j=1}^{k}\mathrm{MSE}_j$ (eq. 5.3), implicitly assuming equal-sized folds. The slide deck gives the **size-weighted** form which is the same when folds are equal but technically more general:

$$
\mathrm{MSE}_j \;=\; \frac{1}{n_j}\sum_{i\in C_j}(y_i-\hat y_i)^{2}, \qquad
\boxed{\;\mathrm{CV}_{(k)} \;=\; \frac{1}{n}\sum_{j=1}^{k} n_j\,\mathrm{MSE}_j\;}
$$

where $C_j$ are the index sets of the $k$ folds, $n_j = |C_j|$, and $\hat y_i$ is the prediction from the model trained on all folds except $j$ (for $i\in C_j$). [slides; [[L10-resample-1|L10]]; concept: [[k-fold-cv]]]

For classification, swap to misclassification:

$$
\mathrm{Err}_j \;=\; \frac{1}{n_j}\sum_{i\in C_j}\mathbb{1}(y_i\ne\hat y_i), \qquad
\mathrm{CV}_{(k)} \;=\; \frac{1}{n}\sum_{j=1}^{k} n_j\,\mathrm{Err}_j
$$

---

## 5. Nested cross-validation (selection + assessment)

**Not in ISLP Ch. 5.** ISLP §5.1.4's [[bias-variance-tradeoff|bias-variance]] discussion implicitly assumes you only do *one* of selection or assessment with CV; [[nested-cv-and-cv-pitfalls|nested CV]] is the prof's solution when you must do both, drawn on the board in L11.

**Why needed.** Once you have used CV to *select* a hyperparameter, that same CV error cannot honestly *assess* the chosen model — you optimized against it. Per the slide deck: *"using the test set for both model selection and estimation tends to overfit the test data, and the bias will be underestimated."* [slides; [[L11-resample-2|L11]]]

**Procedure.** Two layers of CV:

```
Input: data of size n, candidate hyperparameters θ ∈ Θ, outer folds k_out, inner folds k_in.
Randomly partition {1,...,n} into k_out outer folds O_1, ..., O_{k_out}.
For ℓ = 1, ..., k_out:                          # OUTER: assessment
    outer_train_ℓ = data with O_ℓ removed
    outer_test_ℓ  = data in O_ℓ
    Randomly partition outer_train_ℓ into k_in inner folds I_1, ..., I_{k_in}.
    For each θ ∈ Θ:                              # INNER: selection
        For m = 1, ..., k_in:
            Fit model M(θ) on outer_train_ℓ minus I_m.
            Compute err_m(θ) on I_m.
        Inner_CV(θ) = (1/k_in) Σ_m err_m(θ)
    Pick θ̂_ℓ = argmin_θ Inner_CV(θ).
    Fit M(θ̂_ℓ) on all of outer_train_ℓ.
    Score on outer_test_ℓ → err_ℓ.
Outer_CV = (1/k_out) Σ_ℓ err_ℓ        # ← honest assessment estimate
```

Mapped onto the three partitions: outer test fold = "test", inner training folds = "training", inner validation folds = "validation". The outer CV's score is the honest test error of the *selection procedure* (not of any single $\theta$). If the inner CV picks the same $\hat\theta$ on most outer folds, fit the final model on all data with that $\hat\theta$ and report Outer_CV as the assessment. [[[L11-resample-2|L11]]; concept: [[nested-cv-and-cv-pitfalls]]]

---

## 6. The right vs wrong way to do CV (selection-bias trap)

**Not in ISLP Ch. 5** (it is in *Elements of Statistical Learning* §7.10, which the slide deck references). The prof devotes ~¼ of L11 to it and flags it as exam-bait.

### Setup that exposes the trap

- $n=50$, $p=5000$. All $p$ predictors generated as iid $\mathcal{N}(0,1)$. **No relationship to $y$.**
- $y$ assigned as 50/50 random labels. True Bayes error = 50%.
- Two-step pipeline:
  1. Compute the correlation between each predictor and $y$. Keep the top $d=25$.
  2. Fit logistic regression on those $d=25$.

### Wrong way

```
Run step 1 on the FULL data → selected predictors S.
For j = 1, ..., k:
    Train logistic regression on data[-fold j, S], evaluate on data[fold j, S].
CV error ≈ 0% (or ~20% at best) on pure noise.
```

This is a lie: step 1 already used the labels on the full dataset, so the held-out fold has leaked into selection. *"The correlation is already doing a bit of the work for you. It's already a statistical model. You already are selecting parameters based on this for this specific data."* [[[L11-resample-2|L11]]]

### Right way

```
For j = 1, ..., k:
    training_j  = data with fold j removed
    validation_j = fold j
    On training_j ONLY:
        Compute correlation between each predictor and y on training_j.
        Pick top d predictors → selected_j   (different per fold!)
        Fit logistic regression on training_j[, selected_j].
    Predict on validation_j[, selected_j]; compute misclassification on fold j.
CV error ≈ 50%, the truth.
```

**General rule:** anything that uses $y$ — correlation filter, variance filter, supervised PCA, even informal peeks at residual plots — is part of training and must live **inside** the CV loop. [[[L11-resample-2|L11]]; slides; concept: [[nested-cv-and-cv-pitfalls]]]

> [!warning] How bad it gets, verbatim
> *"In the recommended exercises, one of the exercises is basically create fake data using data where there should be no relationship at all, but by pre-selecting which variables you use, you actually get a misclassification error of zero — suggesting like, this is an excellent model, when in reality we know that it's crap."* [[[L11-resample-2|L11]]]

---

## 7. Three-partition rule (training / validation / test)

**Not stated as a formal artifact in ISLP Ch. 5** — ISLP folds it into the chapter introduction. The prof treats it as the foundational discipline for everything in modules 5–11.

| Partition | Job |
|---|---|
| Training | Fit the model. |
| Validation | Select among candidate models / pick hyperparameters (model selection). |
| Test | Report final performance (model assessment). |

**Data-reuse principle (verbatim):** *"We will be too optimistic if we report the error on the test set when we have already used it to choose the best model… Don't do that."* [[[L10-resample-1|L10]]; slides]

CV replaces the **validation** set, not the test set. Pipeline: (i) split off a test set once at the start; (ii) on the rest, run $k$-fold CV to pick the model; (iii) refit on all of "the rest" with the chosen hyperparameter; (iv) report performance on the held-out test set. [[[L10-resample-1|L10]]; concept: [[training-validation-test-split]]]

---

## 8. The independence trap (spatial / temporal correlation)

**Not in ISLP Ch. 5.** The prof's longest tangent in L10, flagged as the headline pitfall of the whole module.

**The problem.** If observations are correlated (in space, in time, by family / clustering), random partitioning into train/validation gives two sets that leak into each other:

> *"You're going to take two points that are right next to each other, but one in your training data, one in your validation data — it's the same damn thing. You haven't created two sets of data. You've just partitioned the same data twice, basically."* [[[L10-resample-1|L10]]]

**Effect on LOOCV (extra bad).** The held-out point is essentially predicted by its near-neighbor → fold error is artificially tiny → CV recommends the most complex model possible. *"It will be basically identical to just using the likelihood without any penalization."* [[[L10-resample-1|L10]]]

**Mitigation.**
- Chunk by the dependency dimension (whole spatial blocks, whole time windows) **before** splitting.
- Prof's neuro-data trick: throw away nearby time bins to enforce independence between fold boundaries — *"I'll throw away 80% or 90% of my data by taking a time bin and then jumping ahead another 10 time bins away and throwing all the stuff away in the middle."* [[[L10-resample-1|L10]]]
- $k$-fold tolerates the fix better than LOOCV because you can deliberately construct folds that respect the dependency structure (put a whole block in one fold).

[[[L10-resample-1|L10]]; concept: [[cross-validation]]]

---

## 9. Bagging (preview from L11; full treatment is in module 8)

**Not in ISLP Ch. 5.** ISLP places [[bagging]] in §8.2.1. The prof previews it inside module 5 because the resampling machinery is identical to the [[bootstrap]], and the variance formula uses the same correlated-mean argument used to explain LOOCV variance.

### Procedure

Training data $(x_1,y_1),\dots,(x_n,y_n)$. For $b=1,\dots,B$:

1. Draw a bootstrap sample of size $n$ with replacement → $(x^{*}_{1,b},y^{*}_{1,b}),\dots,(x^{*}_{n,b},y^{*}_{n,b})$.
2. Fit the base model on it → $\hat f^{*b}$.

Aggregate:

$$
\boxed{\;\hat f_{\text{bag}}(x) \;=\; \frac{1}{B}\sum_{b=1}^{B}\hat f^{*b}(x)\;} \quad\text{(regression)}
$$

For classification: majority vote across $\{\hat f^{*b}(x)\}_{b=1}^{B}$, or average class probabilities and take $\arg\max$. [[[L11-resample-2|L11]]; slides; concept: [[bagging]]]

### Variance formula (the "$\rho\sigma^2$ floor")

Let $\sigma^2 = \mathrm{Var}(\hat f^{*b}(x))$ at a fixed query point $x$ (variance across bootstrap realizations of the fit), and let $\rho$ be the pairwise correlation $\mathrm{Corr}(\hat f^{*a}(x),\hat f^{*b}(x))$ for $a\ne b$. Then:

$$
\boxed{\;\mathrm{Var}\!\Big(\tfrac{1}{B}\sum_{b=1}^{B}\hat f^{*b}(x)\Big) \;=\; \rho\sigma^2 \;+\; \frac{1-\rho}{B}\sigma^2\;}
$$

**Derivation (using the variance-of-sum identity in §3):**

$$
\mathrm{Var}\!\Big(\tfrac{1}{B}\sum_b \hat f^{*b}\Big) = \tfrac{1}{B^2}\Big[\sum_b \mathrm{Var}(\hat f^{*b}) + \sum_{a\ne b}\mathrm{Cov}(\hat f^{*a},\hat f^{*b})\Big] = \tfrac{1}{B^2}\bigl[B\sigma^2 + B(B-1)\rho\sigma^2\bigr] = \tfrac{\sigma^2}{B} + \tfrac{B-1}{B}\rho\sigma^2.
$$

Algebraic rearrangement gives $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$. [[[L11-resample-2|L11]]; concept: [[bagging]]]

**Consequences.**
- If $\rho=0$ (IID samples), variance collapses to $\sigma^2/B$ — the textbook variance-of-the-mean.
- If $\rho>0$, the first term $\rho\sigma^2$ is a **floor that does not shrink with $B$**. This is the motivation for [[random-forest|random forests]] (module 8), which decorrelate trees by randomizing the predictor subset at each split — pushing $\rho$ down.

### When bagging works / fails

- **Works** for high-variance, low-bias base learners (trees especially, KNN with small $K$, deep neural nets in some regimes).
- **Fails to help** for already-low-variance learners (linear regression).
- **The "use enough $B$"** discipline: $B$ is not a tuned hyperparameter for bagging / RF; pick a big number, no CV needed. (Contrast with boosting.)

> *"It's actually using this bagging trick implicitly"* — the prof's aside on why very large over-parameterized single models can win: they implicitly bag across their many parameter subsets, the seed of the [[double-descent]] discussion. [[[L11-resample-2|L11]]]

---

## 10. Out-of-bag (OOB) error

**Not in ISLP Ch. 5.** ISLP places OOB in §8.2.1. The prof introduces it as the natural by-product of bagging because it reuses the $1-1/n$ probability calculation from §5.

### The 1/e result

For a bootstrap sample of size $n$ drawn with replacement from $n$ original observations:

$$
P(\text{obs }i\text{ NOT drawn on a single draw}) = 1 - 1/n
$$

$$
P(\text{obs }i\text{ NOT drawn in any of }n\text{ draws}) = (1-1/n)^n
$$

$$
\boxed{\;(1-1/n)^n \xrightarrow{n\to\infty} \frac{1}{e} \approx 0.368\;}
$$

So $\approx 36.8\%$ of observations are out-of-bag for any given bootstrap tree, and $\approx 63.2\%$ are in-bag. Convergence is fast: $n=10$ gives $\approx 0.349$, $n=100$ gives $\approx 0.366$.

(The derivation itself appears in ISLP §5.2 exercise 2; the *use of it as a free test-set estimator* is module-5 delta vis-à-vis Ch. 5.) [concept: [[out-of-bag-error]]]

### OOB prediction and OOB error

For each $i \in \{1,\dots,n\}$, define $\mathrm{OOB}_i = \{b : i\notin\mathrm{InBag}_b\}$ — the trees that did **not** see observation $i$. The OOB prediction is

$$
\hat y_i^{\,\text{OOB}} \;=\; \text{average (regression) or majority vote (classification) over }\{\hat f^{*b}(x_i) : b\in\mathrm{OOB}_i\}
$$

and the OOB error is

$$
\text{OOB-MSE} \;=\; \frac{1}{n}\sum_{i=1}^{n}\bigl(y_i - \hat y_i^{\,\text{OOB}}\bigr)^{2} \quad\text{(regression)}
$$

$$
\text{OOB-Err} \;=\; \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\bigl(y_i \ne \hat y_i^{\,\text{OOB}}\bigr) \quad\text{(classification)}
$$

**Key claim.** OOB error replaces a separate test set / a separate $k$-fold pass for bagging-family methods. It is approximately equivalent to LOOCV in the limit of large $B$ — each observation is held out from a random subset of trees, and the OOB prediction averages over the trees that did not see it. [[[L11-resample-2|L11]]; concept: [[out-of-bag-error]]]

OOB only works for bagging-family methods. [[boosting|Boosting]] fits trees sequentially on a single re-weighted training set; no per-tree OOB concept exists.

---

## 11. Bootstrap pseudocode for regression coefficients

ISLP §5.2 gives the bootstrap SE formula (eq. 5.8) abstractly and demos it in the lab for the $\alpha$ statistic. The **regression-coefficient version** (exercise 5.5–5.6 of the course and CE1 problem 4d in modified form) is the explicit pseudocode form the prof expects you to be able to write.

```
Input: data {(x_i, y_i)}_{i=1}^n, regression model formula, number of resamples B.
Fit OLS on the original data → β̂ = (XᵀX)⁻¹Xᵀy.
For b = 1, ..., B:
    Draw n row-indices from {1,...,n} with replacement → idx_b.
    Form (X*, y*) = (X[idx_b], y[idx_b]).         # SAME (x_i, y_i) pairs, just resampled
    Fit OLS: β̂*_b = ((X*)ᵀ X*)⁻¹ (X*)ᵀ y*.
    Store β̂*_b.
Compute, for each coefficient j:
    β̄_j*  = (1/B) Σ_b β̂*_{b,j}
    SE_boot(β̂_j) = sqrt( (1/(B−1)) Σ_b (β̂*_{b,j} − β̄_j*)² )
Confidence interval for β_j:
    Percentile method: 2.5% and 97.5% quantiles of {β̂*_{b,j}}_{b=1}^B
    Normal-approx:     β̄_j* ± 1.96 · SE_boot(β̂_j)
```

Compare against the closed-form $\widehat{\mathrm{Cov}}(\hat\beta) = \hat\sigma^2(\mathbf{X}^{\!\top}\mathbf{X})^{-1}$ from module 3. They agree when residuals satisfy the standard IID-Gaussian assumptions; they diverge when those fail, in which case **the bootstrap is closer to the truth** because the closed-form depends on the (potentially wrong) estimate $\hat\sigma^2 = \mathrm{RSS}/(n-p-1)$. [[[L11-resample-2|L11]]; concept: [[bootstrap]]]

### Bootstrap for a derived quantity (CE1 4d pattern)

Same recipe, with the statistic being a predicted probability rather than a coefficient:

```
Fit logistic regression on original data → p̂(x_0) for a specified x_0.
For b = 1, ..., B:
    Draw n rows with replacement.
    Refit logistic regression on the bootstrap rows.
    Compute p̂*_b(x_0).
SE_boot(p̂(x_0)) = sample SD of {p̂*_b(x_0)}_{b=1}^B
95% CI: 2.5% and 97.5% percentiles of {p̂*_b(x_0)}
```

This is the prof-flagged "bootstrap a derived quantity" pattern where no closed-form SE exists. [concept: [[bootstrap]]]

---

## 12. Why-the-classification-CV-curve-rises explanation

**Not in ISLP Ch. 5.** ISLP §5.1.5 shows the classification CV plot (Figure 5.7) but explains the upturn only as a generic [[bias-variance-tradeoff|bias-variance]] shift. The prof gives the sharper explanation in L11.

**Claim.** The training error of [[logistic-regression|logistic regression]] *can rise* with polynomial degree on the misclassification metric, even though training likelihood always improves with more parameters.

**Reason.** *"The error rate that they use on the y is not actually computed the same way as the log likelihood or the likelihood of the model. So you're not actually fitting the error rate directly when you're fitting logistic regression. So even though you're adding more parameters to your model, so you're getting what should be a better fit in your logistic regression, it's not actually minimizing the same thing as the misclassification rate."* [[[L11-resample-2|L11]]]

**General principle.** When the loss you optimize (likelihood / cross-entropy) differs from the metric you report (0/1 misclassification), "more flexibility" no longer guarantees "lower reported error." Comes up later in every classification context where the model is fit by likelihood but evaluated by misclassification or AUC. [[[L11-resample-2|L11]]]

---

## 13. Bootstrap central idea, verbatim

**Not in ISLP Ch. 5 in this form.** ISLP describes the procedure operationally; the prof's framing is the one to write on the A5 sheet:

> *"Your best model for the real world — like for the real data, not just the sample that you have but the bigger sample everywhere — your best model for that is the data itself. And so if you want to look at different realizations of the data, you resample from that same data with replacement, because it's always going to be the best model for the world."* [[[L11-resample-2|L11]]]

The empirical distribution $\hat f$ puts mass $1/n$ on each observed point. **Sampling from $\hat f$ with replacement at full size $n$ is the bootstrap.** Without replacement at full $n$ is a permutation — useless. The same-length-with-replacement requirement is what makes it different. [[[L11-resample-2|L11]]; concept: [[bootstrap]]]

---

## 14. Notation and naming differences

| Slides / prof | ISLP Ch. 5 | Notes |
|---|---|---|
| $\mathrm{CV}_{(k)} = \frac{1}{n}\sum_{j=1}^{k} n_j\,\mathrm{MSE}_j$ (weighted) | $\mathrm{CV}_{(k)} = \frac{1}{k}\sum_{j=1}^{k}\mathrm{MSE}_j$ (unweighted, eq. 5.3) | Same when folds equal-sized. Prof's form is general. |
| "Validation set approach" | "Validation set approach" (§5.1.1) | Same. Prof emphasizes it is not strictly *cross*-validation. |
| LOOCV: $h_i$ for hat-matrix leverage | ISLP uses $h_i$ (eq. 5.2) | Same. Some treatments use $h_{ii}$. |
| $\mathrm{Err}_i = \mathbb{1}(y_i\ne\hat y_i)$ | $\mathrm{Err}_i = I(y_i\ne\hat y_i)$ | Identical, indicator notation. |
| Bootstrap SE: $\widehat{\mathrm{SE}}_B(\hat\theta)$ from $B$ resamples | $\mathrm{SE}_B(\hat\alpha)$ (eq. 5.8) | Identical formula. |
| $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$ (bagging variance) | ISLP §8.2.1 mentions decorrelation but does **not** print this formula | Formula is module 5 slide + L11 derivation. |
| "Wrong-way CV" / "selection bias in CV" | Hastie–Tibshirani §7.10 (ESL); not in ISLP Ch. 5 | Different book. The prof's verbatim framing is "lying with statistics." |
| Three-partition: training / validation / test | Same names | Identical. |
| "Inner / outer CV" for nested CV | Not formally named in ISLP Ch. 5 | Module 5 delta. |
| OOB fraction: $1-1/e\approx 0.368$ | ISLP exercise 5.2 derives $(1-1/n)^n$ but does not introduce OOB until §8.2.1 | Module 5 delta. |
| `cv.glm` / `cv.tree` / `boot::boot` (R) — function names | ISLP uses Python: `cross_validate`, `KFold`, `ShuffleSplit`, custom `boot_SE` | **Out of scope per prof's exam policy** — no language, no package names. Listed only because slide deck uses R names. |
