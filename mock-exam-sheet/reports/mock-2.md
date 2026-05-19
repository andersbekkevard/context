# Mock Exam 2 — Cheat-sheet candidates

## Question summary

- **P1** (10 P) Fill-in-the-blank concept passes (10 blanks): validation/CV/bootstrap/forward-stepwise/lasso/sparse/PCA/local-min/dendrogram/boosting.
- **P2a** Bootstrap T/F: sampling, B's role, OOB ≈ 1/e direction, percentile CI.
- **P2b** CV pitfalls T/F: LOOCV-vs-kfold bias/variance, wrong-way CV (filter outside loop), CV under autocorrelation, nested CV inner/outer roles.
- **P2c** Subset counts: best-subset $2^p$, forward-stepwise $1+p(p+1)/2$, backward-stepwise impossible in $p>n$.
- **P2d** Spline df: cubic spline $K+3$, natural cubic $K+1$ (excl. intercept), continuity through $C^2$.
- **P2e** Logistic: odds factor $e^{\beta\Delta}$, sigmoid eval, "β = change in probability" trap (false).
- **P2f** Confusion-matrix arithmetic (accuracy/sens/spec/precision).
- **P2g** K-means T/F: local-min, monotone WCSS, scale sensitivity, non-nested across K.
- **P2h** Tree-ensemble T/F: mtry-decorrelation, bagging doesn't overfit in B, GB = residual fitting under squared loss, mtry defaults.
- **P2i** Direction-of-effect T/F: smoothing-spline λ↑ smoother (true), boosting ν↓ needs fewer trees (FALSE).
- **P3a** Derive bagging variance $\rho\sigma^2 + (1-\rho)\sigma^2/B$ from scratch (8 P — the prof-flagged mathy one).
- **P3b** Single-linkage hierarchical clustering by hand on 5×5 matrix; first-merge linkage-invariance.
- **P3c** Bootstrap: $P(i\notin)=(1-1/n)^n\to 1/e$ derivation + algorithm for SE of the median.
- **P4a** Read OLS output with interaction; "main effect at zero of the moderator" trap; CI for ptratio coef.
- **P4b** Forward-stepwise vs best-subset model-counts; greedy-disagreement reason.
- **P4c** Ridge: objective with un-penalized intercept; lasso vs ridge sparsity; standardization rationale; ridge-beats-OLS ⇒ OLS variance-dominated.
- **P4d** PCR eigenvalue 80% cutoff; PCR-as-discretized-ridge; GAM total df count.
- **P4e** Boosting: nonlinearities/interactions inference, B is a real tuning parameter (overfits), VIP + PDP roles.
- **P5a** LDA discriminant 1D, decision boundary $=(\mu_0+\mu_1)/2$ for equal priors/var; prior-shift direction; QDA derivation showing $x^2$ coef ≠ cancel.
- **P5b** Logistic with age:sex interaction — odds factor differs by sex, age:sex sign flip, η/p̂ calc for new patient, hierarchy/marginality principle.
- **P5c** Sens/spec/error from confusion matrix; AUC definition + AUC=0.5; threshold-lowering direction on ROC.
- **P5d** Random-forest hyperparam justification; sens-vs-spec model recommendation; error-rate misleading at class imbalance.

---

## High-priority surfacing

### A. Prof-specific / not-in-ISLP-as-formula

- **Bagging variance under correlation: $\mathrm{Var}(\bar f_{\text{bag}}) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$** (filter A, D)
  - The whole P3a derivation. ISLP states this only verbally.
  - Hits P3a (8 P derivation), P2h(i), and the "why RF beats bagging" question implicitly behind P5d.
  - Confirmed prof-specific: `wiki/book-deltas/m08-trees.md:20` ("ISLP §8.2.2 explains the idea verbally … but does **not** write down the formula. The prof did it on the board"), and `wiki/concepts/random-forest.md:42-55`.
  - Cheat-sheet form:
    $$\mathrm{Var}\!\Big(\tfrac{1}{B}\textstyle\sum X_b\Big) = \tfrac{1}{B^2}[B\sigma^2 + B(B-1)\rho\sigma^2] = \rho\sigma^2 + \tfrac{1-\rho}{B}\sigma^2.\ \text{Floor }\rho\sigma^2\text{ as }B\to\infty.$$

- **PCR = discretized ridge** (filter A, C)
  - Prof's headline framing: ridge applies smooth shrinkage $d_j^2/(d_j^2+\lambda)$ to SVD directions, PCR applies a 0/1 truncation. Same "pressure on low-variance directions."
  - Hits P4d(ii).
  - Source: `wiki/concepts/principal-component-regression.md:19` (verbatim L15 quote), 79-89.
  - Compact: Ridge shrinks the $j$-th SVD direction by $\tfrac{d_j^2}{d_j^2+\lambda}$ (smooth); PCR keeps top $M$ at weight 1, drops the rest at weight 0 (discrete).

- **Spline df cheat-sheet (cubic vs natural, with/without intercept)** (filter A, C)
  - The 2024 + 2025 exams both used this, and P2d(i)–(ii) + P4d(iii) lean on it. ISLP's df count is muddled.
  - Source: `wiki/concepts/regression-splines.md:55,58,104`.
  - Compact:
    - Cubic spline, $K$ interior knots: **$K+3$** params (excl. intercept), $K+4$ incl.
    - Natural cubic spline: **$K+1$** (excl. intercept), $K+2$ incl. (boundary linearity = 2 constraints, that's why natural is $-2$).
    - Cubic spline at every knot: $C^2$ continuous (value, slope, curvature); only the 3rd derivative jumps.

- **GAM-with-spline df assembly** (filter C)
  - Sum across terms: intercept (1) + spline-term df (K+3 for cubic-$K$-knots, excl. intercept) + each linear cont. predictor (1) + each dummy column.
  - Hits P4d(iii) directly. $1 + 6 + 5 + 1 = 13$.
  - Compact: $\text{df}_{\text{GAM}} = 1 + (K+3) + (p_{\text{lin}}) + \sum(\text{levels}_j - 1)$.

- **Permutation variable importance vs impurity (Gini) importance** (filter A, B)
  - Prof prefers permutation. Hits P4e(iii) (VIP diagnostic).
  - Source: `wiki/book-deltas/m08-trees.md:107,142`.
  - Compact: permutation VI = drop in OOB accuracy after shuffling $X_j$; works on any model; impurity VI is biased toward high-cardinality predictors.

- **Hierarchical-clustering hand-procedure with $-1$ per mistake** (filter A, C)
  - Hits P3b. Prof flagged as canonical exam question.
  - Source: `wiki/concepts/hierarchical-clustering.md:20,120`.
  - Compact recipe: find min off-diagonal → merge at that height → recompute new row/col by linkage rule (single=min, complete=max, average=mean) → repeat. **First merge is linkage-invariant** (only singletons involved).

- **OOB derivation $(1-1/n)^n \to 1/e \approx 0.368$** (filter A, D)
  - Hits P2a(iii) and P3c(i,ii).
  - Source: `wiki/book-deltas/m08-trees.md` §2 (derivation listed as delta-worthy).
  - Compact: $P(i\notin \text{boot})=(1-1/n)^n \to e^{-1}\approx 0.368$, so $P(i\in)\to 1-e^{-1}\approx 0.632$.

### B. Direction-of-effect / interpretation traps

- **Boosting shrinkage ν and tree count B move in OPPOSITE directions** (filter B)
  - P2i(ii) is the trap. Halving ν roughly doubles required B.
  - Source: `exam_analysis.md:152` (Mar 17, Apr 13); `exam_analysis.md:439` study plan.
  - Compact: $\nu\downarrow \Rightarrow B\uparrow$ (each tree contributes less, more trees needed).

- **Bagging/RF do NOT overfit in B; boosting DOES** (filter B, C)
  - P2h(ii) trap. The B floor is $\rho\sigma^2$, not 0; can't drive it negative.
  - Source: `wiki/concepts/bagging.md:87,110`; `exam_analysis.md` (boosting B is a real tuning parameter).
  - Compact: RF: pick B large enough that OOB error plateaus; never CV $B$. Boosting: $B$ is a real tuning parameter — too large → overfit.

- **Main-effect coefficient under an interaction = effect at moderator = 0** (filter B)
  - P4a(ii): "chas main effect is 1.20 at standardized rm=0, full effect is $\hat\beta_{\text{chas}}+\hat\beta_{\text{rm:chas}}\cdot\text{rm}$, depends on rm."
  - P5b(iv): same trap with age:sex — sex main effect is effect at age = 0, far outside data; do not drop main effect while keeping interaction (**hierarchy/marginality principle**).
  - Source: `wiki/concepts/categorical-encoding-and-interactions.md:81`; `exam_analysis.md` §4f trap #1.
  - Compact: With interaction $x_1:x_2$ in the model, $\hat\beta_{x_1}$ is the slope of $x_1$ at $x_2=0$; never drop a main effect whose interaction stays. Total effect of $x_2$ on response = $\hat\beta_{x_2}+\hat\beta_{x_1:x_2}\cdot x_1$.

- **LDA prior shift moves boundary AWAY from the higher-prior class** (filter B)
  - P5a(iii). With $\pi_0=0.8$, boundary moves rightward (toward $\mu_1$), needs more evidence to predict class 1.
  - Source: `wiki/concepts/linear-discriminant-analysis.md:94,126`.
  - Compact: Boundary shifts **toward** the low-prior class mean. Easy to flip under time pressure.

- **Logistic β is on log-odds scale, NOT probability** (filter B)
  - P2e(iii) explicit trap.
  - Source: `exam_analysis.md:159,174`.
  - Compact: $\Delta x_j = 1 \Rightarrow \text{odds} \times e^{\beta_j}$. Probability change depends nonlinearly on current $p$ (max near $p=0.5$).

- **Wrong-way CV: pre-screening predictors outside the loop biases CV downward** (filter B)
  - P2b(ii) trap.
  - Source: `wiki/concepts/nested-cv-and-cv-pitfalls.md:34,55`.
  - Compact: **Anything that uses $y$ must live inside the CV loop.** Pre-filter then CV ⇒ optimistic test-error estimate (Exercise 5.3 gives ~0% on pure noise).

- **Test-error rate misleading under class imbalance** (filter B)
  - P5d(iii) explicit setup: $\Pr(\text{chd})=0.27$, trivial classifier beats logistic.
  - Source: `wiki/concepts/sensitivity-specificity.md:114`; `wiki/concepts/confusion-matrix.md:83,108`.
  - Compact: Baseline = $\max(\hat\pi_k)$. Beat that, or report (sens, spec) pair / AUC instead.

- **Threshold $\downarrow$: sens $\uparrow$, spec $\downarrow$, ROC up-and-right** (filter B)
  - P5c(iii).
  - Source: `wiki/concepts/roc-auc.md` (general).
  - Compact: Lower threshold $\Rightarrow$ more positives flagged $\Rightarrow$ TPR↑ and FPR↑; operating point slides up-and-right along ROC.

- **K-means is NOT nested across K** (filter B)
  - P2g(iv) trap. Going $K=4\to 5$ can reorganise globally; only hierarchical is nested.
  - Source: `wiki/concepts/k-means-clustering.md`.

### C. Decision rules / cross-method comparisons

- **Subset-selection counting cheats** (filter C)
  - Best-subset: $2^p$ (with null) or $\sum_{k=0}^{K}\binom{p}{k}$ up to size $K$.
  - Forward stepwise total: $1 + p(p+1)/2$ for the full path; up through size $K$: $1 + p + (p-1) + \cdots + (p-K+1)$.
  - Hits P2c(i,ii) and P4b(i,ii).
  - Source: `exam_analysis.md:184`.
  - Compact: $p=8$: best $2^8=256$; forward full $37$. $p=7$ to size 4: forward $1+7+6+5+4=23$; best $\sum_{k=0}^4 \binom{7}{k}=99$.

- **Backward stepwise dies in $p>n$** (filter C)
  - P2c(iii). Full OLS isn't defined ($X^TX$ singular). Forward survives until size $\min(n-1,p)$.

- **Ridge objective with un-penalized intercept** (filter C, A)
  - P4c(i). Always exclude $\beta_0$ from the penalty; with standardized $X$ the intercept is $\bar y$.
  - Compact: $\hat\beta^R_\lambda=\arg\min\{\sum_i(y_i-\beta_0-x_i^\top\beta)^2 + \lambda\sum_{j=1}^p\beta_j^2\}$, sum over $j=1,\dots,p$ only. Closed form on centred data: $(X^\top X+\lambda I_p)^{-1}X^\top y$.

- **Ridge beats OLS in test MSE ⇒ OLS was variance-dominated** (filter C)
  - P4c(iv). Ridge trades bias for variance reduction; only wins when OLS variance is the binding constraint.
  - Compact: $\text{MSE}_{\text{ridge}}<\text{MSE}_{\text{OLS}}\Rightarrow$ variance dominates; often signals predictor correlation / borderline $n/p$ ratio.

- **Random-forest mtry defaults** (filter A, C)
  - P2h(iv) + P5d(i). $m=\sqrt{p}$ classification, $m=p/3$ regression. ISLP §8.2.2 states classification default but not regression default explicitly.
  - Source: `wiki/book-deltas/m08-trees.md:159`.
  - Compact: $m=\lfloor\sqrt p\rfloor$ (class), $m=\lfloor p/3\rfloor$ (reg). Smaller $m$ → less correlated trees (lower $\rho$).

- **LDA/QDA discriminant formulas (1D)** (filter A, C)
  - P5a(i,iv): the prof gives the formula in the question, but exam-day cheat-sheet should still carry it.
  - Compact:
    - LDA (1D, shared $\sigma^2$): $\delta_k(x)=x\mu_k/\sigma^2 - \mu_k^2/(2\sigma^2) + \log\pi_k$.
    - QDA (1D): $\delta_k(x)=-\tfrac{x^2}{2\sigma_k^2}+\tfrac{x\mu_k}{\sigma_k^2}-\tfrac{\mu_k^2}{2\sigma_k^2}-\tfrac{1}{2}\log\sigma_k^2+\log\pi_k$.
  - **Equal priors + equal variance, 2 classes, 1D: boundary at $(\mu_0+\mu_1)/2$.**
  - Where the quadratic in QDA comes from: $-x^2/(2\sigma_k^2)$ has different coefficients across $k$ when $\sigma_k^2$ differ, so the $x^2$ term doesn't cancel when you subtract $\delta_0-\delta_1$.
  - Source: `wiki/concepts/linear-discriminant-analysis.md:95,116`; `exam_analysis.md:203-204`.

- **AUC definition** (filter C)
  - P5c(ii).
  - Compact: AUC = $P(\hat f(X^+)>\hat f(X^-))$ for random positive/negative pair = area under ROC over all thresholds. AUC = 0.5 = random guess.

### D. Math derivations the prof flagged

- **Bagging variance derivation** — already surfaced under Filter A above; this is the load-bearing P3a mathy one. Prof flagged at least one derivation question for 2026 (exam_analysis §4g) — bagging-variance, LDA boundary, and bias-variance are the three top candidates. Bagging-variance shows up here as 8 P.

- **LDA decision boundary derivation** — surfaced under cross-method comparisons; flagged twice in `wiki/concepts/linear-discriminant-analysis.md:114-116` as exam-canonical.

- **$(1-1/n)^n\to 1/e$ OOB derivation** — surfaced under Filter A above; the prof did it on the board.

---

## Lower-priority but useful

- **Sigmoid evaluation:** $\hat p = e^\eta/(1+e^\eta)$. P2e(ii), P5b(iii). ISLP-standard.
- **Sensitivity / specificity definitions in clinical language:** P5b(v). Prof emphasizes the verbal framing ("of patients who really get sick, how many do we catch"), not just the formula. Source: `wiki/concepts/sensitivity-specificity.md`.
- **Confusion matrix four metrics:** accuracy, sensitivity (recall, TPR), specificity (TNR), precision (PPV). P2f. ISLP-standard.
- **K-means objective is non-increasing each step but only finds local min:** P2g(i,ii). ISLP-standard but worth a one-liner.
- **Standardization is mandatory before:** PCA / Ridge / Lasso / K-means / hierarchical-clustering / KNN. P4c(iii), P2g(iii). Source: `exam_analysis.md:217`.
- **Percentile bootstrap CI:** the 2.5%/97.5% quantiles of the bootstrap distribution. P2a(iv) + P3c(iii). Source: `wiki/concepts/bootstrap.md:54`.
- **Nested CV roles:** **inner** = selection/tuning, **outer** = assessment. P2b(iv) tests the swap. Source: `wiki/concepts/nested-cv-and-cv-pitfalls.md:27-30`.
- **LOOCV vs k-fold bias-variance:** LOOCV = low bias, **high variance** (correlated folds); k=5/10 = lower variance, slight upward bias. P2b(i). Source: `exam_analysis.md:162`.
- **K-fold CV under autocorrelation breaks:** P2b(iii). Need rolling/expanding window CV. Trap flagged by prof (Feb 9 anecdote, `exam_analysis.md` §4f #2).
- **Standardized-coef CI:** $\hat\beta \pm 1.96\cdot\widehat{\text{SE}}$ on standardized predictors directly reads "effect of 1 SD increase." P4a(iii).
- **PCA variance share = $\lambda_j/\sum_j\lambda_j$**, cumulative-sum to 80/90/95%. P4d(i). Standardized predictors → trace = $p$.

---

## Things explicitly in ISLP (skipped)

- Bootstrap = sampling with replacement, same size $n$ (P2a(i)) — ISLP §5.2.
- "Single resample is enough" trap (P2a(ii)) — ISLP §5.2 explicit.
- Forward stepwise computational tractability for $p>n$ (P2c hint) — ISLP §6.1.2.
- $K$-means scale sensitivity (P2g(iii)) — ISLP §12.4.1 explicit.
- Gradient boosting with squared loss = residual fitting (P2h(iii)) — ISLP §8.2.3.
- Logistic odds factor $e^{\beta\Delta}$ (P2e(i)) — ISLP §4.3.
- Confusion matrix arithmetic (P2f) — ISLP §4.4.3.
- AUC = 0.5 is random guess (P5c(ii)) — ISLP §4.4.3.
- "Boosting captures nonlinearities/interactions a linear model can't" (P4e(i)) — ISLP §8.2.3.
- B-too-large overfits boosting (P4e(ii)) — ISLP §8.2.3.
- Standardize before PCA (P2g(iii) for K-means, similar idea) — ISLP §12.2.
