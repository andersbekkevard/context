# Mock Exam 3 — Cheat-sheet candidates

## Question summary

- **Q1 (10 pts):** Fill-in-the-blank concept words (parametric vs nonparametric, bias-variance, shrinkage/sparsity, k-fold, AIC distributional assumptions, GAMs, PCA, dendrogram).
- **Q2 (28 pts):** T/F + short numeric grab-bag: (a) bias-variance + double descent, (b) CV variants incl. LOOCV hat-matrix shortcut + wrong-way CV, (c) NN parameter count (6→8→4→2 softmax) + sigmoid forward pass, (d) odds/log-odds + interaction trap, (e) splines dof + continuity, (f) PCA on 5 standardized vars + score on PC1, (g) bootstrap probability + (1-1/e), (h) random forest mtry defaults + OOB + B-not-tuned, (i) K-means.
- **Q3 (16 pts):** (a) LDA: state assumptions, derive δ_k from log(π_k f_k), numerical 2D boundary with Σ⁻¹ given, name QDA shape; (b) 4×4 hierarchical clustering by hand, single linkage; complete-linkage final height; cut at h=4.5; (c) bias-variance decomposition with expectations explicitly named, and identify double descent.
- **Q4 (22 pts):** Insurance/`charges` data analysis: (a) OLS with `bmi:smoker` interaction (the interaction trap), F-test for region, R² vs adj-R²; (b) ridge with 10-fold CV, 1-SE rule, bias-variance interpretation of OLS beating ridge; (c) GAM dof counting (smoothing spline df=4 + cubic bs with 3 knots + dummies); GAM additivity limitation; (d) gradient boosting hyperparameters (B, d, ν), why B is tuned in boosting but not RF, variable-importance + PDP for interpretation.
- **Q5 (24 pts):** Stroke classification: (a) logistic regression — `exp(β)` for age, plug-in patient prediction, sensitivity/specificity in plain English; (b) LDA vs QDA confusion matrices with cost asymmetry; (c) RF with mtry=3, OOB, variable-importance limits (no direction, no causality); (d) ROC + AUC plain-English interpretation, threshold lowering direction; (e) class imbalance: 5% naive baseline beats all classifiers; metric choice.

## High-priority surfacing

- **LDA discriminant derivation — explicit cancellation step** (filter A, D)
  - The prof's flagged "mathy" question. Starting from log(π_k f_k(x)) with shared Σ, the term −½ x'Σ⁻¹x is *k*-independent → drops in arg max → δ_k is linear. With class-specific Σ_k that term *survives* → QDA boundary is quadratic. The verbatim "where does the quadratic come from?" punchline.
  - Q3a(ii) demands this; Q3a(iv) requires naming QDA.
  - Source: `wiki/concepts/linear-discriminant-analysis.md:82-89`; `exam_analysis.md:203-204,300-309`.
  - Cheat-sheet form: `δ_k(x) = x'Σ⁻¹μ_k − ½ μ_k'Σ⁻¹μ_k + log π_k`. Quadratic-cancellation note: "x'Σ⁻¹x has no k → drops (LDA). With Σ_k it survives → QDA." LDA pooled: `Σ̂ = Σ_k [(n_k−1)/(n−K)] · S_k`.

- **LDA numerical boundary recipe (equate δ_0 = δ_1)** (filter A, D)
  - Steps: compute Σ⁻¹(μ_1−μ_0) → LHS direction; compute ½(μ_1'Σ⁻¹μ_1 − μ_0'Σ⁻¹μ_0) → RHS scalar; with equal priors log-priors cancel. Sanity check: midpoint of μ_0, μ_1 lies on the line.
  - Q3a(iii) — `2x_1 + 7x_2 = 25` (verified by midpoint (2,3)).
  - Source: `mock-exams/mock-exam-3-solution.tex:329-365`; `wiki/concepts/linear-discriminant-analysis.md:100-110`.
  - Cheat-sheet form: "Boundary: `x'Σ⁻¹(μ_1−μ_0) = ½(μ_1'Σ⁻¹μ_1 − μ_0'Σ⁻¹μ_0) + log(π_0/π_1)`. Sanity: midpoint lies on line iff equal priors."

- **Double descent — bias-variance decomposition is exact, just not U-shaped** (filter A, D)
  - Prof's hobbyhorse (~5 mentions). Decomposition `E[(y−f̂)²] = Bias² + Var + σ²` is an algebraic identity for *any* estimator. Past the interpolation point (p≈n) optimization shifts to "min ‖β‖₂ subject to interpolation" → minimum-norm interpolator → variance comes back down. Bias and variance still sum exactly.
  - Q2a(ii)/(iii), Q3c(ii) — name "double descent / benign overfitting." Profile B answer.
  - Source: `wiki/concepts/double-descent.md:55-78`; `exam_analysis.md:75-79,247-258`.
  - Cheat-sheet form: "Decomp exact for any f̂. Past p≈n: pseudoinverse/SGD picks min-norm interpolator → implicit ridge → variance ↓ again. Name: double descent / benign overfitting."

- **Bias-variance — what each expectation is over** (filter A, D)
  - Outer E[·] is jointly over training-set draws *and* test-point noise ε. Bias² uses E[f̂(x₀)] over training-set draws only; Var(f̂(x₀)) also training-set draws. σ² is the noise variance, independent of estimator.
  - Q3c(i) explicitly demands "what randomness the expectations are taken over."
  - Source: `mock-exams/mock-exam-3-solution.tex:442-458`; `exam_analysis.md:75-79`.
  - Cheat-sheet form: "Outer E: jointly over (training set, test noise). Inner Bias/Var: training set only."

- **LOOCV hat-matrix shortcut (OLS only)** (filter A, C)
  - `CV_n = (1/n) Σ ((y_i − ŷ_i)/(1 − h_ii))²` with H = X(X'X)⁻¹X'. Single fit, no refits. Works only for linear methods.
  - Q2b(iii) T/F directly tests it.
  - Source: `wiki/concepts/leave-one-out-cv.md:39-55`; `exam_analysis.md:210`.
  - Cheat-sheet form: "PRESS: `CV_n = (1/n)Σ((y_i−ŷ_i)/(1−h_ii))²`. OLS only. h_ii from H = X(X'X)⁻¹X'."

- **LOOCV bias/variance — direction trap** (filter B)
  - LOOCV: **low bias** (trains on n−1) but **high variance** (n training sets share n−2 obs → fold errors highly correlated → average is high-variance). k-fold (k=5,10) is the variance-favoring compromise. CE1 4b reverses both, false.
  - Q2b(i).
  - Source: `wiki/concepts/leave-one-out-cv.md:78-90`; `exam_analysis.md:162`.
  - Cheat-sheet form: "LOOCV: low bias, HIGH variance (correlated folds). k-fold: lower variance, slight bias."

- **Wrong-way CV / nested CV** (filter A, B)
  - Filtering predictors with y outside the CV loop biases test error toward 0 (the exercise 5.3 simulation: p=5000 random → ~0% with bad CV, true rate is 50%). Anything that uses y is training. Selection must live inside each fold.
  - Q2b(iv) — the canonical "permissible to use y to filter once" false statement.
  - Source: `wiki/concepts/nested-cv-and-cv-pitfalls.md:46-89`; `exam_analysis.md:273`.
  - Cheat-sheet form: "Anything using y = training. Filter inside each CV fold or estimate is biased downward (possibly to ~0%)."

- **1-SE rule — precise statement** (filter A, C, D)
  - Among λ with CV(λ) ≤ CV(λ̂_min) + SE(CV(λ̂_min)), pick the **largest** (= simplest / most-regularized). Motivation: CV-min is itself noisy (you optimized against the same folds); prefer parsimony. The "SE not quite valid" footnote (prof flagged): same folds used to select and estimate SE.
  - Q4b(iii).
  - Source: `wiki/concepts/one-standard-error-rule.md:21-56`; `exam_analysis.md:193,343-349`.
  - Cheat-sheet form: "1-SE: λ_1SE = max{λ : CV(λ) ≤ CV(λ̂_min) + SE}. Pick simplest within 1 SE."

- **Interaction trap: main-effect of `smoker_yes` at bmi=0 is meaningless** (filter B, A)
  - The prof's #1 flagged trap. `β_smoker_yes = −22.50` is the smoker contrast *only at bmi=0* (out-of-support). Actual smoker effect at bmi=30: `−22.50 + 1.45·30 = +21.0` — opposite sign! Hierarchical principle: main effects must accompany interactions.
  - Q4a(ii) ΔEUR per bmi unit: non-smoker = +100, smoker = +1550 (β_bmi + β_int = 1.55, ×1000 EUR). Q4a(iii) explanation.
  - Source: `wiki/concepts/categorical-encoding-and-interactions.md:75-105`; `exam_analysis.md:264,317`.
  - Cheat-sheet form: "With X:Z interaction, main β_Z is contrast at X=0 only. Slope for X under Z=1 is β_X + β_XZ. NEVER quote main effect as 'average.'"

- **Odds ↔ probability + interaction multiplier** (filter B, C)
  - `odds = p/(1−p)`, `p = odds/(1+odds)`. β_j is log(odds-ratio), unit-increase multiplier is `e^β_j`. With interaction `X:Z`, unit increase in X for Z=1 multiplies odds by `e^(β_X + β_XZ)`, not `e^β_X`.
  - Q2d(i) p=0.05 → 0.0526; (ii) odds=4 → p=0.80; (iii) smoker bmi unit → e^(0.02+0.10)=1.13 (half-credit for e^0.02 trap).
  - Source: `wiki/concepts/odds-and-log-odds.md:39-68`; `exam_analysis.md:174,313-321`.
  - Cheat-sheet form: "p↔odds: p=O/(1+O), O=p/(1−p). β = log(OR). Interaction: smoker bmi-unit → e^(β_bmi+β_int). 0.02+0.10 → 1.13."

- **Cubic vs natural spline dof counting** (filter A, B)
  - Cubic regression spline with K knots: K+4 params (or K+3 excluding intercept). Natural cubic spline: K+2 (or K, excluding intercept) — boundary linearity = 2 fewer params. Third derivative is *discontinuous* at knots in plain cubic spline (only value, 1st, 2nd derivatives match). Knots fixed in advance, not estimated.
  - Q2e (T/F), Q4c(i) GAM dof: bs with K=3 knots = K+3=6 (intercept removed).
  - Source: `wiki/concepts/regression-splines.md:39-67,100-108`; `exam_analysis.md:184`.
  - Cheat-sheet form: "Cubic spline: K+4 params (K+3 sans intercept). Natural: K+2 (K sans intercept). 3rd derivative DISCONTINUOUS at knots. Knots fixed, not fit."

- **GAM dof totaling — term-by-term** (filter A, C)
  - Global intercept (1) + smoothing-spline df (4) + cubic regression spline K+3 (6 for 3 knots) + binary smoker (1) + region 3-dummies (3) + children (1) = **16**.
  - Q4c(i).
  - Source: `mock-exams/mock-exam-3-solution.tex:580-597`; `wiki/concepts/regression-splines.md:124-128`.
  - Cheat-sheet form: "GAM dof = intercept + smooth-df + (K+3 cubic bs) + (K−1 dummies per K-level factor) + linear-term dof. Sum it term by term."

- **Bootstrap `1 − (1 − 1/n)^n` → `1 − 1/e ≈ 0.632`** (filter A)
  - For n=8: 1 − (7/8)^8 = 0.656. As n→∞: 1 − 1/e ≈ 0.632. The "with replacement" rule is non-negotiable (without replacement = useless permutation). One bootstrap sample says ~nothing. Bootstrap quantifies *variance*, not bias.
  - Q2g (all three parts), connects to OOB.
  - Source: `wiki/concepts/bootstrap.md:56-77,144-150`; `exam_analysis.md:208`.
  - Cheat-sheet form: "P(in boot) = 1−(1−1/n)^n → 1−1/e ≈ 0.632. With replacement only. Bootstrap = variance, not bias correction."

- **PCA score formula + standardization mandate** (filter A, C)
  - Score `z*_m = φ_m^T x*` (inner product of loading with observation). On standardized data, total variance = p (# variables). PCA is **not** scale-invariant — without standardization the largest-unit variable dominates PC1. Loadings unit-norm: `Σ φ_jm² = 1`. Eigenvalue of covariance = variance of that PC.
  - Q2f(i) total = 5; (iv) z*_1 = 1.025; (v) X_1 (|0.55| largest).
  - Source: `wiki/concepts/principal-component-analysis.md:37-50,85-92`; `exam_analysis.md:187-189`.
  - Cheat-sheet form: "Score z_m = φ_m'x. Standardize first (PCA NOT scale-invariant). Loadings unit-norm. Total var = p for standardized data. Eigenvalue = PC variance."

- **Random forest defaults + B-not-tuned argument** (filter B, C)
  - mtry: `√p` classification (with p=6: 3), `p/3` regression. B is *not* tuned by CV — RF test error monotonically non-increasing in B (averaging only ↓ variance); pick "enough" (~500-1000). Boosting: B *is* tuned because sequential fitting overfits noise. Trees grown deep, unpruned.
  - Q2h all four; Q5c(i); Q4d(ii).
  - Source: `exam_analysis.md:185-186`; mock solution `:242-254`.
  - Cheat-sheet form: "mtry: √p class / p/3 reg. B: RF not tuned (variance ↓ only), boosting tuned (sequential = overfits). RF trees grown deep, unpruned."

- **OOB error mechanics** (filter A)
  - Each bootstrap sample omits ~1−1/e ≈ 37% of obs. For each obs i, OOB prediction is the vote/mean over only trees whose bootstrap didn't include i. Near-unbiased proxy for test error, free (no CV, no holdout).
  - Q2h(iii), Q5c(iii).
  - Source: `wiki/concepts/bootstrap.md:67-69` (1/e link); `exam_analysis.md:186`.
  - Cheat-sheet form: "OOB: ~37% out per bootstrap. Predict each i from only trees that didn't see i. Near-unbiased test-error estimate, free."

- **Variable-importance limits** (filter B, C)
  - Shows total contribution magnitude (mean decrease in node impurity OR permutation importance — prof prefers permutation). Does NOT show **direction**, **shape**, or **causality**. For direction/shape: PDPs / ICE. The prof flagged Gini-vs-permutation preference: *"I would generally go with [permutation] because it makes more sense."*
  - Q5c(iv), Q4d(iii) — what additional diagnostic for interpretation? → variable importance + PDP.
  - Source: `exam_analysis.md:253` (prof preference); mock solution `:798-812`.
  - Cheat-sheet form: "VI = magnitude only. NO direction, NO shape, NO causality. Use PDP/ICE for direction. Prof prefers permutation over Gini."

- **AUC plain-English interpretation** (filter B, A)
  - AUC = P(score for random positive > score for random negative). 0.5 = chance; <0.5 = invert classifier (still informative). Independent of class prevalence (unlike accuracy).
  - Q5d(i) "beyond area-under-curve."
  - Source: `wiki/concepts/roc-auc.md:55-70`; `exam_analysis.md:211`.
  - Cheat-sheet form: "AUC = P(p̂(pos) > p̂(neg)) over random pair. 0.5 chance. <0.5 = flip predictions. Threshold-free."

- **Threshold lowering direction (ROC operating point)** (filter B)
  - Lower threshold → more positive predictions → TPR↑ (sensitivity↑) AND FPR↑ (specificity↓). On the ROC curve, operating point moves **up and to the right**.
  - Q5d(ii)(iii).
  - Source: `wiki/concepts/sensitivity-specificity.md:62-72`; `wiki/concepts/roc-auc.md:36-45`.
  - Cheat-sheet form: "Threshold ↓: sens ↑, spec ↓. On ROC: up-and-right. Threshold ↑: sens ↓, spec ↑. Down-and-left."

- **Class imbalance — naive baseline beats classifiers on error rate** (filter B)
  - With 5% positives, "always-no" classifier has 5% error rate — beating LDA (14%), QDA (24%), RF (15%). Test error rate is misleading. Use sens/spec pair, AUC (or PR-AUC for heavy imbalance), or "sens at fixed spec." Cost-sensitive: missed stroke ≫ false alarm.
  - Q5e(i)(ii).
  - Source: mock solution `:832-852`; `exam_analysis.md:370`.
  - Cheat-sheet form: "5% prevalence → naive 'always-no' = 5% error, beats real models. Use (sens, spec), AUC, PR-AUC. Don't use raw error on imbalanced data."

- **NN parameter count: bias on receiving layer** (filter A, C)
  - For each layer: `(prev_width + 1) × this_width`. Bias on receiving layer only, inputs have no bias. For 6→8→4→2: 7·8 + 9·4 + 5·2 = 56+36+10 = **102**. No-bias trap = 88.
  - Q2c(i).
  - Source: `wiki/concepts/nn-parameter-count.md:36-62`; `exam_analysis.md:214`.
  - Cheat-sheet form: "Params per layer: (in+1) × out. 6→8→4→2 = 56+36+10 = 102. Softmax adds nothing."

## Lower-priority but useful

- **Why standardize before ridge** (filter A): L2 penalty is scale-dependent; large-unit predictors get small β → barely penalized; small-unit get big β → over-penalized. Standardize so penalty acts on signal strength, not units. (Q4b(i); `exam_analysis.md:267`)

- **Ridge limit behavior** (filter C): λ→0 → OLS; λ→∞ → all β=0 (intercept conventionally unpenalized, fit collapses to ȳ). Ridge never gives exactly zero — lasso does (Q1 trap). (`exam_analysis.md:149`)

- **F-test for joint factor relevance** (filter C): For testing whether a K-level categorical jointly matters after controlling for other predictors. H₀: all K−1 dummy coefficients = 0. Individual t-tests are NOT enough — dummies can each be insignificant while group is. Note: prof flagged F-tests as low-priority overall but the *concept* (partial-F / ANOVA comparison full vs reduced) is fair game in this Q4a(iv) setup. (`exam_analysis.md:358`)

- **R² vs adjusted R² gap** (filter C): Small gap = essentially no penalty for over-parameterization, every predictor "earns its keep." R² always ↑ with more predictors; adj R² may ↓. (Q4a(v); `exam_analysis.md:195`)

- **Gradient boosting hyperparameters trade-off** (filter C): B too small = high bias; too large = overfit (unlike RF). ν small → need more B, generally better generalization (halving ν ~ doubles B). d (depth) sets interaction order: stumps (d=1) = additive ensemble. Typical ν ∈ [0.001, 0.1]. (Q4d(i); `exam_analysis.md:213`)

- **GAM additivity limitation** (filter C): GAM = Σ_j f_j(x_j) — no interactions unless added by hand. Cannot learn age × bmi effects. Trees capture interactions for free (every split is conditional on parent splits). The boosting-beats-GAM gap is the interaction signal. (Q4c(ii), Q4d(iii); `exam_analysis.md:158`)

- **Hierarchical clustering hand recipe + linkage formulas** (filter A, D): Single = min, complete = max, average = mean. The −1 per mistake convention. First two merges identical across linkages when only singletons fuse. Cut interpretation: above some heights, below others = how many clusters. (Q3b; `wiki/concepts/hierarchical-clustering.md:66-90`; `exam_analysis.md:287-298`)

- **K-means non-properties** (filter B): K is user-chosen (not auto). Random init → local minima → must run many times, pick lowest WCSS. K-means does NOT produce a hierarchy (that's hierarchical clustering). (Q2i; `exam_analysis.md:275`)

- **Validation-set ≠ 2-fold CV** (filter B): Validation = single split (one fit, one evaluation). 2-fold averages over both directions (fold A train + fold B test, then swap). (Q2b(ii))

- **AIC distrust / prof's preference for CV** (filter C): Prof: *"I don't trust them... making some assumption that probably won't hold."* — distributional assumptions on residuals. CV preferred. (Q1 blank 7; `exam_analysis.md:250`)

- **Sigmoid forward pass** (filter A): z = b + Σ w_j x_j; σ(z) = 1/(1+e^(−z)). For z=0.2 → 0.550. ReLU alternative: max(0, z) = 0.2.

## Things explicitly in ISLP (skipped)

- **Bias-variance decomposition formula** itself (ISLP 2.2): always in scope but well-covered in book.
- **Ridge objective** `Σ(y − x'β)² + λΣβ²` (ISLP 6.2): direct from book; cheat-sheet only if space.
- **Linear regression standard errors / t-statistics / R²** (ISLP 3.1): standard textbook output interpretation; no special prof framing.
- **Confusion matrix definitions, sensitivity = TP/(TP+FN), specificity = TN/(TN+FP)** (ISLP 4.4.2 Table 4.6): present in book; the *plain-English-for-this-problem* framing is the prof's emphasis (worth a one-line cheat note but the formulas are in ISLP).
- **PCA via eigendecomposition / PVE = λ_m / Σλ** (ISLP 12.2): direct book treatment.
- **K-means objective Σ_k Σ_{i∈C_k} ‖x_i − x̄_k‖²** (ISLP 12.4.1): book formula.
- **Hierarchical clustering algorithm 12.3** (ISLP 12.4.2): book contains the recipe verbatim.
- **Bootstrap algorithm** (ISLP 5.2): present in book; the `1−1/e` derivation and "anything using y is training" angle are the prof's specials.
- **Gradient boosting algorithm 8.2** (ISLP 8.2.3): in book; the prof avoids line-by-line memorization.
- **Logistic regression `logit(p) = β₀ + βx`** (ISLP 4.3.1): present in book; the *interaction-trap* application is the prof's special.
