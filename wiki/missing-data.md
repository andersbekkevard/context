---
title: Missing Data — what the prof said, what it means for every model on the exam
description: A query-time synthesis of the (sparse) lecture treatment of missing predictor values across TMA4268, anchored on L18 (Trees), with model-by-model implications for the 2026 final.
---

# Missing data — a cross-cutting issue the prof flagged but barely taught

Missing predictor values is one of those topics the professor keeps gesturing at — "we've talked about this multiple times" — but which, when you actually scan the transcripts, gets exactly **one** substantive treatment all semester, buried in the Trees lecture. Every other reference is a half-sentence aside. This page exists because (a) it really is the hardest thing about real data, (b) the prof's offhand framing leaves you under-prepared to *discuss* it on the exam, and (c) it interacts differently with every model class on the curriculum. We anchor on what he actually said, then extend it to a clean mental model you can talk fluently about.

> [!note] What this page is
> A synthesis. ISLP and the prof together give a thin treatment; the *exam-style* discussion ("compare how method X and Y handle incomplete data") has to come from the tutor side. We mark prof material with verbatim quotes; the rest is calibrated extension.

---

## 1. What the prof actually said

There is exactly one substantive passage in all 27 lectures. It comes in **[[L18-trees-2]]** (2026-03-16), near the end, while the prof is walking through the advantages and disadvantages of trees:

> "And the other one is you often have missing variables, and this has been a problem that we've talked about multiple times, that in regression often your biggest source of screwing up is simply you don't have the right variables. And there's nothing you can really… I mean we look at different ways of how to accommodate that, but it really is a major problem, right? Because often the theories break because of it. So here, yeah, if we have that, we can impute missing variables, like missing values — sorry, these are missing values. Ah, okay, I was confused. I remembered variables, but this is value. So if you have a frame where you have an instance in your experiment and you're missing variables, you can either discard them completely or you can try to impute them, meaning make some estimate of what they should be, either by taking a mean or a more complicated model. You can also do fancier methods but we won't talk about those."

— `transcripts/tma4268-2026-03-16.txt` (around byte 35 500; [[L18-trees-2]])

Three things to extract from this:

1. **He distinguishes "missing variables" from "missing values"** — and then catches himself doing it. *Missing variables* = you never collected the right predictor at all (an **omitted-variable** problem, which is structural and degrades every model). *Missing values* = you collected the variable, but some rows are blank (an **incomplete-data** problem, which is the focus of this page). The two are easy to conflate and you should keep them separate when discussing.
2. **The whole menu, in his telling, is just two options**: *discard* the affected rows (listwise deletion / `na.omit`) or *impute* (replace with the mean, or with a model-based estimate). He flags that "fancier methods" exist but explicitly puts them out of scope.
3. **He calls it "a major problem" where "the theories break"** — meaning the inference machinery (standard errors, p-values, confidence intervals) is built on the assumption of a complete IID sample. Once you start dropping or imputing, those theoretical guarantees no longer hold without further care.

Everywhere else in the semester this topic is essentially absent. The closest companion remarks are:

- **[[L04-statlearn-3]]** (Jan 19, EDA discussion): *"Throwing — just deleting data is generally a really bad idea."* He's talking about outliers there, not missingness, but the prior is the same: don't reach for the delete key as a first move.
- **[[L05-linreg-1]]** (Jan 20): a passing aside about removing outliers being "a discussion for another day" when contrasting Gaussian vs Laplace error models.
- **[[L20-boosting-2]]** (Apr 7): he mentions XGBoost as "the one everyone uses" but does **not** flag that XGBoost handles missing values natively — a famous feature he passes over.
- **[[L21-unsupervised-1]]**: he teaches PCA but skips ISLP §12.3 entirely, which is the textbook's whole treatment of **PCA-as-imputation** (the Netflix problem, soft-impute, hard-impute).

So: the prof gives you the headline ("discard or impute") but leaves the cross-method comparison to you. That comparison is the rest of this page.

---

## 2. The frame the prof skipped: why missingness has a *mechanism*

You can't reason about missing data without knowing **why** it's missing. The standard taxonomy (Rubin 1976) is three levels:

- **MCAR — Missing Completely At Random.** The probability that a value is missing is independent of every variable, observed or not. E.g. a sensor randomly drops 1% of readings. Listwise deletion is *unbiased* here (you just lose efficiency).
- **MAR — Missing At Random.** The probability of missingness depends only on *observed* values. E.g. older subjects skip the income question more often, but you observe age. Imputation conditioned on observed variables can recover unbiased estimates.
- **MNAR — Missing Not At Random.** The probability of missingness depends on the *unobserved* value itself. E.g. people with very high income refuse to disclose income. No purely statistical procedure can fix this without external assumptions; you need domain knowledge or sensitivity analysis.

The prof never names this taxonomy, but it's the missing pillar (pun intended) for *discussing* trade-offs. When you say "imputation is fine here," you are tacitly asserting MAR. When the question is open-ended ("discuss the pros and cons of dropping vs imputing"), MCAR/MAR/MNAR is the axis the markers expect you to recognise even if you don't name it.

---

## 3. The two strategies, sharpened

### Strategy A — Discard (listwise deletion, `na.omit`, complete-case analysis)

**How:** Drop every row that has *any* missing value before fitting.

**Pros.** Trivial to implement. Statistically clean *if MCAR*: the surviving rows are an unbiased (just smaller) sample from the same population. No new modelling assumptions are introduced; the inference machinery the prof loves (SEs, $t$-tests, $F$-tests, confidence intervals, $\chi^2$ deviance) still holds on the reduced sample.

**Cons.**
- **Sample-size hemorrhage in high dimensions.** With $p$ predictors and missingness probability $q$ per cell, the fraction of rows with *zero* missing entries is roughly $(1-q)^p$. For $p = 20, q = 0.05$ that's $\approx 36\%$ — you've thrown away two-thirds of your data. The prof's L04 instinct ("deleting data is generally a really bad idea") kicks in here for a quantitative reason.
- **Bias under MAR/MNAR.** If older respondents skip a question, the complete-case dataset is younger than the population — every estimate that depends on age (directly or through correlated predictors) is biased.
- **Inference is over-confident in a hidden way.** Your CIs use the complete-case $n$, but you have a non-random subsample. The prof's "the theories break" lands here.

### Strategy B — Impute (mean, model-based, multiple imputation)

**How:** Fill in the blanks before fitting.

- **Mean / median / mode imputation** (the prof's first example). Replace each missing value with the marginal mean of that column.
  - *Pros.* One line of code; you keep every row.
  - *Cons.* Shrinks the variance of the imputed column toward zero (you've added a spike at the mean), which **attenuates regression coefficients** and **inflates** $R^2$ artefactually for predictors with much missingness. It ignores correlations among predictors entirely. The bias from mean-imputation is in many cases *worse* than from listwise deletion.

- **Model-based / conditional imputation** ("a more complicated model" in the prof's words). Regress the missing column on the observed columns and use the predicted value. The textbook version is *k*-NN imputation or chained-equations regression imputation.
  - *Pros.* Respects the correlation structure; works correctly under MAR.
  - *Cons.* You're now stacking a model on top of a model; if the imputation model is mis-specified, you bake that error into the downstream fit. Standard errors computed naively are still too small (single imputation undercounts the imputation uncertainty).

- **Multiple imputation** (the "fancier methods" the prof exiled). Draw $M$ plausible imputed datasets, fit your model $M$ times, combine the estimates with Rubin's rules. This properly propagates imputation uncertainty into the final standard errors. Out of scope for our exam, but worth knowing the name exists — that's what you'd appeal to if pushed on "how would inference actually be valid here?".

- **PCA / matrix-completion imputation** (ISLP §12.3, *skipped in lectures*). Treat the data matrix as low-rank-plus-noise and fill in missing entries by iteratively (i) running PCA on the current fill, (ii) replacing missing cells with their rank-$M$ reconstruction. This is the **Netflix problem** algorithm. The prof didn't teach it, so it is **out of scope** for the exam — but be aware it's the natural Module 10 extension, and ISLP labs it.

---

## 4. Method-by-method: how each model on your exam interacts with missing predictors

This is the section that lets you *discuss* missing data at the exam, since the prof's quote is a generic "discard or impute" and the marker may push you into "but what about for X?". Here's the matrix you need in your head, walking the curriculum in module order.

### Linear regression (Module 3) and logistic regression (Module 4)

These are fit by closed-form normal equations or by IRLS, both of which need a complete design matrix $\mathbf X \in \mathbb R^{n\times p}$. **Any row with a missing cell must be removed or imputed before fitting** — there is no native handling. The prof's L18 framing applies most directly here: drop or impute. The classical inference machinery (the $t$-tests in the `summary()` output, the F-test for nested models, the deviance test for logistic) is built on a fully observed IID sample, so the bias risk from MAR is highest in exactly the methods where you most rely on inference. **This is the model class where missingness most clearly "breaks the theories."**

A subtle point: if you `na.omit()` and then run model selection (subset, ridge, lasso), you've conditioned the *training set* on the variables you happened to include, so changing the variable list also changes which rows survive. The fitted models are then literally trained on different data and can't be compared via CV the way you'd hope.

### LDA, QDA (Module 4)

Same situation as linear regression: closed-form via class means and a (pooled or per-class) covariance matrix. **Complete rows only.** The covariance estimates are particularly sensitive to imputation choices because mean-imputation systematically reduces estimated variances and biases the covariance toward zero on the imputed columns, which then bends the decision boundary. If you have to impute before LDA, do it *within each class* — otherwise you bleed class information across the boundary.

### KNN (Module 4)

KNN computes Euclidean distances between rows. A missing cell makes the distance undefined. Workarounds: drop the row, impute, or compute distance on the observed-pairs subset and rescale (the latter is the "available-case" trick, rarely used here). **The curse of dimensionality interacts badly with missingness**: as $p$ grows, distances become uniform, and adding noisy imputed values for missing cells degrades that already-fragile signal further. KNN is the model class where mean-imputation is most dangerous, because *every* coordinate matters for the metric.

### Cross-validation and bootstrap (Module 5)

The prof's L10 emphasis on independence of folds matters here too. **If you impute on the full dataset and then CV, you've leaked test-set information into the imputation step** — your CV error is optimistically biased. The correct pipeline is to impute *inside each fold*, fitting the imputation model on the training fold and applying it to the validation fold. Same lesson as standardization (which he does cover in [[L13-modelsel-2]] for ridge/lasso): all preprocessing that uses the response or other rows must live inside the CV loop. Bootstrap inherits the same issue: missing cells must be re-imputed on each bootstrap draw if you want honest standard errors.

### Best subset, ridge, lasso (Module 6)

All three optimise a quadratic loss on a complete design matrix — same constraint as plain regression. **Ridge and lasso are particularly vulnerable to imputation artefacts** because they shrink coefficients in proportion to predictor variance, and mean-imputation distorts that variance. The prof teaches that ridge/lasso require **column standardisation** ([[L13-modelsel-2]]); if you mean-impute first and then standardise, you've shrunk the SD of imputed columns, which makes the penalty effectively weaker on them — they survive variable selection too easily. Order matters: impute → standardise inside CV.

### PCA, PCR, PLS (Module 6) and unsupervised PCA (Module 10)

This is where the missing-data story has its most beautiful arc *in the textbook* — and the prof gave you none of it. ISLP §12.3 develops **matrix completion via PCA**: the same iterative algorithm that gives you principal components also solves the missing-cell problem, by alternately (i) replacing missing cells with their low-rank reconstruction and (ii) re-running PCA on the filled matrix. The fixed point is the rank-$M$ best-completion. It's the Netflix Prize algorithm. The prof skipped it in [[L21-unsupervised-1]], so it is **out of scope for the exam in the technical sense**, but if a discussion question asks *"how could PCA help with missing data?"*, the answer worth knowing is exactly this — and the conceptual link (low-rank-plus-noise data has redundant information that can fill gaps) is in scope by virtue of PCA itself being in scope.

### Splines, smoothing splines, local regression, GAMs (Module 7)

GAMs sum univariate basis fits $f_1(X_1) + \dots + f_p(X_p)$, which in principle could be **fit column-by-column on whatever rows have that column observed**. In practice, the back-fitting algorithm needs all $p$ partial residuals at every iteration, so you still effectively need complete cases — but the *backfitting structure* makes GAMs the easiest place to retrofit a column-specific imputation. Univariate smoothers (a polynomial in one variable, a smoothing spline in one variable) don't care about other columns and can ignore those rows' missingness in the other coordinates.

### Trees, bagging, random forests (Module 8)

This is the one method class where the model itself has *native* missing-data handling — and it is precisely the section the prof skipped despite teaching for two full lectures. The standard CART techniques are:

- **Surrogate splits.** At every node, in addition to the primary split (e.g. "Age $\le 45$"), the algorithm stores backup splits on correlated variables ("Income $\le 50k$"). When a test observation has Age missing, it uses the highest-ranked surrogate that *is* observed. Built into `rpart` in R.
- **Distribute down with weights.** Send the observation down *all* children with weights proportional to the training-data class proportions in each child; aggregate the leaf predictions weighted by those flow weights.
- **Send to most common child.** Simplest variant: the observation goes to whichever child contains more training observations.

Bagging and random forests inherit whichever scheme the base CART implementation uses. Random forests have a clever bonus: after fitting, you can use **proximity-based imputation** (the matrix of pairwise "how often did rows $i$ and $j$ end up in the same leaf?" is itself a similarity measure for filling missing cells).

**Trees are therefore the most missing-data-robust method on your curriculum** — and that is exactly the *advantage* the prof was *about to* mention in the L18 quote before he got distracted clarifying "variable" vs "value." Worth highlighting on the exam if asked which method is most practical for messy data.

### Boosting, gradient boosting, XGBoost (Module 9)

XGBoost's signature feature, beyond what plain gradient boosting offers, is **default-direction learning at every split**: at training time, for each split it learns which side ("left" or "right") missing values should go to *as part of the loss-minimisation* — this is the feature that more than anything made XGBoost the dominant tabular ML model of the 2010s. LightGBM has the same property. The prof mentioned XGBoost in [[L20-boosting-2]] as the one "everyone uses" but did **not** explain this feature. Worth knowing because if a question asks "why has XGBoost become standard on Kaggle?", native missing-data handling is one of the two real answers (alongside its scalability tricks).

### Neural networks (Module 11)

Neural networks need a complete tensor on every forward pass. **There is no native missingness mechanism** the way there is in trees. Real-world practice uses:

- **Imputation before training** (mean, k-NN, or a separate small network).
- **Indicator augmentation**: add a 0/1 column $\mathbf 1[X_j \text{ missing}]$ alongside each imputed column, so the network can learn that "imputed mean" is a different signal from "observed mean." (Out of scope but conceptually clean.)
- **Embedding "missing" as a category** for one-hot inputs (common in tabular deep learning).
- **Masked attention / masked losses** in sequence models — out of scope.

The relevant exam-discussion point: neural networks share the same "fully-observed design matrix" requirement as linear regression, but because they have many more parameters they are *more* vulnerable to spurious patterns introduced by naive imputation. The bias-variance angle the prof loves ([[L03-statlearn-2]]): a NN with mean-imputed inputs can over-fit to the artefact spike at the mean.

---

## 5. The exam-ready summary table

| Method | Native handling? | What works in practice | Biggest pitfall |
|---|---|---|---|
| Linear / logistic regression | None | Drop (MCAR) or model-based impute (MAR) | Mean-imputation attenuates coefficients; inference invalid |
| LDA / QDA | None | Impute within each class | Imputation bleeds class info across boundary |
| KNN | None | Drop; mean-imputation hurts the metric | Distance dominated by imputed cells |
| CV / bootstrap | n/a (it's the wrapper) | Impute *inside* each fold | Test-set leakage if imputed on full data |
| Ridge / lasso | None | Impute → standardise → CV, in order | Mean-imputation makes shrinkage too weak on imputed cols |
| PCR / PCA | Natural — matrix completion | Iterative PCA fill (out of TMA4268 scope) | Skipped in lectures; ISLP §12.3 if you want depth |
| Splines / GAMs | Column-wise | Fit column-by-column on observed rows | Backfitting still wants complete cases |
| Trees / RF | **Yes — surrogate splits** | Just hand the data to `rpart`/`randomForest` | Prof didn't teach this; know the term "surrogate split" |
| XGBoost / LightGBM | **Yes — default-direction learning** | Pass NA through; XGBoost learns split-side per node | Prof mentioned XGBoost without explaining why it dominates |
| Neural networks | None | Impute + add missing-indicator columns | Over-fits to imputation artefacts |

---

## 6. How to talk about this on the exam

If a 2026 question lands on missing data — and given the prof's "major problem" framing it's plausible as a 2–4 mark discussion sub-question — you have three reliable beats:

1. **Distinguish missing-variables from missing-values**, citing the L18 distinction. Whoever marks it has heard the prof catch himself on this and will reward the precision.
2. **Name the strategy menu**: listwise deletion vs single imputation (mean or model-based) vs multiple imputation, and note that the prof flagged the last as "fancier" / out of scope. Pair this with the MCAR / MAR / MNAR axis even though he didn't name it — that's the way to discuss "pros and cons" rather than just "options."
3. **Anchor on method-class differences**: trees and boosted trees handle it natively (surrogate splits; default-direction learning); everything else needs pre-processing; PCA can in principle do matrix completion but he skipped it. If the question is open-ended, this is the part that distinguishes a fluent answer from a list.

And the one line the prof would probably appreciate hearing back: *yes, the theories break — the inference you're taught in regression assumes a complete IID sample, and the moment you delete or impute, you've stopped doing the inference the standard errors claim you're doing.*

---

## See also

- [[L18-trees-2]] — the one substantive passage
- [[L04-statlearn-3]] — "deleting data is generally a really bad idea" (about outliers, not missingness, but related instinct)
- [[L05-linreg-1]] — outlier-vs-robust-regression aside
- [[L21-unsupervised-1]] — PCA, where matrix completion would have lived if taught
- ISLP §12.3 — matrix completion (the only place in the book this is developed)
