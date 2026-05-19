# Mock Exam 1 — Cheat-sheet candidates

## Question summary

- **P1 (10p)** Fill-in-blank concept vocab: supervised/unsupervised, prediction/inference, irreducible error, regularization, generative/discriminative, PCA, hierarchical clustering. Pure vocab recall.
- **P2a (3p)** Bias–variance T/F (flexibility direction; high-σ² regime; irreducible NOT shrinkable by data).
- **P2b (3p)** Ridge vs Lasso T/F (ridge ≠ exact 0; L1 corners; ridge "less flexible" than OLS; standardize).
- **P2c (4p)** NN parameter count (5→6→1 with biases = 43) + ReLU forward pass (pre-activation -0.7 → ReLU = 0).
- **P2d (3p)** Odds↔probability (odds=2.5→p=5/7; p=0.3→odds=3/7; β=0.8 → e^0.8≈2.23).
- **P2e (3p)** Smoothing spline λ direction T/F (λ→0 wiggly, λ→∞ OLS line, df=tr(S), non-integer).
- **P2f (5p)** PCA: total variance = p; cumulative PVE; score z*₁ = φ₁ᵀx*; largest-loading variable.
- **P2g (3p)** Sensitivity 140/200=0.70, specificity 720/800=0.90; AUC T/F (A,C,D correct; B false).
- **P2h (2p)** Why nonlinear activations (universal approximator + collapse-to-affine).
- **P2i (2p)** LDA vs QDA T/F (shared Σ → linear; xᵀΣ⁻¹x cancels; small-n favors LDA; both generative).
- **P3a (8p)** MATHY: LS objective + assumptions; Gaussian log-likelihood; show MLE=LS by stripping constants; Laplace → L1 (not LS).
- **P3b (5p)** Hierarchical clustering by hand with complete linkage on 4×4 matrix; first-merge under single linkage; cut height.
- **P3c (3p)** Bias-variance decomposition (state 3 terms with names AND what expectations are over); critique "trade-off" framing (squared bias, regularization, double descent).
- **P4a (6p)** Read OLS table: 9 params (n−p−1 = 341 sanity check); 95% CI on fuel_electric (3.20 ± 1.96·1.10); name F-test for joint fuel test; mpg-collinearity interpretation.
- **P4b (3p)** Training vs test RSS when adding redundant cubic+rescaled-age terms; XᵀX singular under perfect collinearity; ridge λI saves uniqueness.
- **P4c (6p)** 10-fold CV-MSE pseudocode; choice of λ̂_min vs λ̂_1SE; OLS vs lasso comparison (no shrinkage gain ⇒ predictors mostly carry signal); 1-SE rule statement.
- **P4d (5p)** GBM hyperparameters B, d, η; why B "infinite" is wrong (overfits); test-MSE gain over OLS ⇒ nonlinearities/interactions ⇒ use VI + PDP for interpretation.
- **P4e (2p)** Cubic regression spline df with 4 internal knots = K+3 = 7; total GAM df = 26 (sum across components incl. categorical 2-dummy).
- **P5a (10p)** Logistic with `balance:sex` interaction: odds-factor for male = e^β_bal, female = e^(β_bal+β_int); explain why sex_female main effect = -0.18 is "diff at balance=0" (meaningless); sensitivity/specificity in problem-specific words; compute confusion-matrix metrics.
- **P5b (3p)** ROC axes (TPR vs FPR); AUC definition + AUC=0.5 chance; lower threshold → ↑sens, ↓spec, move up-right on ROC.
- **P5c (4p)** KNN procedure; sensitivity/spec/error from confusion matrix; mixed-units → Euclidean dominated by balance.
- **P5d (5p)** Tree competitor: pick RF, ntree=500, mtry=√9=3, terminal-node-size=1; compare to logistic (both ~tied → pick logistic for interpretability OR RF if missing-defaulter cost matters); VI plot shows magnitude not direction.
- **P5e (2p)** Class imbalance: naive accuracy = 0.80 (always predict majority); use sensitivity/specificity pair, AUC, or F1 / cost-sensitive metric instead.

---

## High-priority surfacing

### A1. **Single-hidden-layer NN parameter count** (filter A, C)
- One-liner: `params = (p+1)·M + (M+1)·C`; bias is on the **receiving** layer, never on input.
- Hit by: P2c(i). Canonical wrong answer = forgetting biases (-7 deduction equivalent).
- Repo: `wiki/concepts/nn-parameter-count.md:46-49`; pitfalls list at `:108-114`.
- Cheat-sheet form:
  ```
  1 hidden: params = (p+1)M + (M+1)C
  Multilayer: sum of (prev+1)·this for each non-input layer
  Trap: bias only on RECEIVING units; dropout does NOT reduce count
  ```

### A2. **MLE = LS under Gaussian errors — the canonical mathy slot** (filter D, A)
- One-liner: Write log-likelihood, drop θ-independent constants, MLE-argmax = LS-argmin. State the assumption explicitly. Laplace ⇒ L1.
- Hit by: P3a (all 4 parts). Prof verbatim ("I do generally like to keep one theory question … assume an additive Gaussian error model").
- Repo: `wiki/concepts/least-squares-and-mle.md:47-51` (the 3-line proof); `wiki/book-deltas/m03-linreg.md:144-164` (full reproduction, since ISLP does **not** prove this).
- Cheat-sheet form:
  ```
  log L(β,σ²) = -n/2·log(2πσ²) - (1/2σ²)·Σ(y_i - x_iᵀβ)²
  arg max_β log L = arg min_β Σ(y_i - x_iᵀβ)² = β̂_LS
  Laplace errors ⇒ MLE = LAD (min Σ|y_i - ŷ_i|), L1, robust to outliers
  Sign trap: maximize log L = minimize -log L = minimize RSS
  ```

### A3. **Bias–variance decomposition with named expectations** (filter D, A)
- One-liner: 3 terms with explicit "what is the expectation taken over." Outer E over training set AND noise; Var/E[f̂] are over training set only.
- Hit by: P3c. Prof has flagged FOUR separate times "definitely on exam."
- Repo: `wiki/concepts/bias-variance-tradeoff.md:38-43, 65-69`; derivation at `:77-93`.
- Cheat-sheet form:
  ```
  E[(y₀ - f̂(x₀))²] = (f(x₀) - E[f̂(x₀)])² + Var(f̂(x₀)) + σ²
                     ↑ squared bias        ↑ variance    ↑ irreducible
  Outer E: over training sample + noise ε₀.
  Inner E[f̂] and Var(f̂): over training sample only.
  Cross terms vanish: (1) E[ε]=0; (2) E[E[f̂] - f̂] = 0.
  Why "decomposition" > "trade-off": exact identity; regularization
    flattens variance without paying full bias cost (squared bias
    is forgiving); double descent allows both to drop together.
  ```

### A4. **Cubic-spline / natural-spline / GAM df counting (the intercept trap)** (filter A, C)
- One-liner: Cubic spline K knots = K+4 params incl. intercept (= K+3 excl.); natural cubic = K+2 incl. (= K+1 excl.). Smoothing spline df = tr(S_λ), non-integer.
- Hit by: P4e(i) [K=4 → K+3 = 7] and P4e(ii) [total GAM = 26].
- Repo: `wiki/book-deltas/m07-beyondlinear.md:312-325` (consolidated table + intercept-trap restatement); `wiki/concepts/regression-splines.md:57-59`.
- Cheat-sheet form:
  ```
  Polynomial deg d:           d+1 (incl. intercept)
  Step fn, K cutpoints:       K+1
  Linear spline, K knots:     K+2
  Cubic spline, K knots:      K+4
  Natural cubic, K int knots: K+2
  Smoothing spline:           tr(S_λ) ∈ [2, n], non-integer
  GAM total = 1 (intercept) + Σ (df of each component) + Σ (#dummies)
  Categorical with L levels contributes L-1 dummies
  Trap: when "df" is the R `df=` argument, intercept is ALREADY supplied
  ```

### A5. **Smoothing spline direction-of-effect** (filter B)
- One-liner: λ↑ ⇒ smoother / lower df. **Opposite** direction from polynomial degree. Penalty is on **second** derivative.
- Hit by: P2e (all 4 statements turn on this direction).
- Repo: `wiki/concepts/smoothing-splines.md:41-56`.
- Cheat-sheet form:
  ```
  λ → 0:  interpolating spline (df → n), wiggly
  λ → ∞:  OLS straight line (df → 2)
  Penalty = ∫ g''(t)² dt (NOT g'(t)²; 2023 exam trap)
  df_λ = tr(S_λ) — non-integer in general
  Same direction as ridge/lasso λ; opposite from poly degree, knot count
  ```

### A6. **Logistic regression interaction trap** (filter B)
- One-liner: With `x:z` interaction, odds-factor for unit increase in x is **e^(β_x + β_xz·z)**, not e^β_x. The main-effect on z alone is "diff at x=0," typically meaningless.
- Hit by: P5a(i), P5a(ii). Prof's #1 flagged interpretation trap.
- Repo: `wiki/concepts/odds-and-log-odds.md:62-68`; `wiki/concepts/categorical-encoding-and-interactions.md:85-95`.
- Cheat-sheet form:
  ```
  Logistic model: log(p/(1-p)) = β₀ + β_x x + β_z z + β_xz·x·z
  Odds-factor for Δx=1 at fixed z:  e^(β_x + β_xz·z)
    Male (z=0):    e^β_x
    Female (z=1):  e^(β_x + β_xz)
  Main-effect β_z = (z=1 vs z=0) diff in log-odds at x=0 only.
  Always state your reference coding. Flip reference → flip all signs.
  Main-effects rule: include x and z whenever you include x·z.
  ```

### A7. **Hierarchical clustering by hand — the −1-per-mistake question** (filter D)
- One-liner: Find smallest off-diagonal → merge → recompute via linkage rule (complete=max, single=min, average=mean). First merge is **identical across linkages** (it's still point-to-point).
- Hit by: P3b (all parts).
- Repo: `wiki/concepts/hierarchical-clustering.md:50-58` (linkage table); `:74-91` (recipe + worked example); prof's "-1 per mistake" rubric at `:65-72`.
- Cheat-sheet form:
  ```
  Complete: D(A∪B, C) = max(D(A,C), D(B,C))   — balanced clusters
  Single:   D(A∪B, C) = min(D(A,C), D(B,C))   — chains
  Average:  weighted mean over all pairs       — balanced
  First merge: linkage-INDEPENDENT (singletons, only one pair).
  Cut at height h: count # vertical lines crossed = # clusters.
  Horizontal leaf position = ARBITRARY; only fusion HEIGHT carries info.
  Sanity check: heights should be non-decreasing (no centroid inversions).
  ```

### A8. **One-standard-error rule** (filter C)
- One-liner: Pick the **largest λ** (simplest model) whose CV error ≤ min CV error + 1·SE. Prefers parsimony, accepts a small bias for stability.
- Hit by: P4c(iv).
- Repo: `wiki/concepts/one-standard-error-rule.md:42-45`; "not quite valid" footnote at `:55-56`.
- Cheat-sheet form:
  ```
  λ̂_1SE = max { λ : CV(λ) ≤ CV(λ̂_min) + ŜE(CV(λ̂_min)) }
  Why: CV minimum is itself noisy → choosing argmin overfits CV folds.
  Simpler model = more interpretable + more stable test perf.
  Footnote: ŜE not "clean" — same folds used for selection and SE.
  ```

### A9. **Perfect collinearity vs ridge regularization** (filter A, C)
- One-liner: Redundant predictor ⇒ XᵀX **singular** ⇒ OLS has no unique solution. Ridge replaces XᵀX with XᵀX + λI, always invertible, splits coefficient between collinear columns.
- Hit by: P4b(iii). Listed by prof as a key conceptual contrast.
- Repo: `wiki/book-deltas/m03-linreg.md:331-345` (collinearity blow-up of (XᵀX)⁻¹); ridge connection in `wiki/concepts/ridge-regression.md` (not loaded but established).
- Cheat-sheet form:
  ```
  Perfect collinearity → rank(X) < p+1 → XᵀX singular → OLS undefined
  Training RSS unchanged (redundant column adds no col-space info)
  Ridge fix: (XᵀX + λI)⁻¹ always exists for λ > 0
            ⇒ unique β̂; coefficient SHARED across collinear columns
  ```

### A10. **PCA score computation + standardization invariant** (filter B, C)
- One-liner: Score z_im = φ_m · x_i (dot product). Standardize first ⇒ Σ_j var(X_j) = p, so total variance = #variables.
- Hit by: P2f (all 5 parts).
- Repo: `wiki/concepts/principal-component-analysis.md:48-52, 85-93`.
- Cheat-sheet form:
  ```
  Loading vector: ‖φ_m‖² = 1 (unit norm)
  Score: z_im = φ_mᵀ x_i (works for any new obs at test time)
  Standardized data: total variance = p (=Σ eigenvalues)
  PVE_m = λ_m / Σλ_j; cumulative PVE picks "how many PCs to keep"
  Largest |φ_jm| ⇒ variable j contributes most to PC m
  PCA is NOT scale invariant — standardize or biggest-unit var wins
  Unique up to SIGN flip
  ```

### A11. **KNN scaling trap** (filter B, C)
- One-liner: Euclidean distance dominated by the largest-unit feature. Standardize before KNN (and ridge/lasso/PCA/k-means/hierarchical clustering).
- Hit by: P5c(iii). Universal-scaling cheat: same rule as PCA.
- Repo: `exam_analysis.md:165, 217`; `wiki/concepts/knn-classification.md` (linked from MOC).
- Cheat-sheet form:
  ```
  Standardize required for: KNN, Ridge, Lasso, PCA, K-means, hierarchical
  KNN: dist(x*, x_i) = Σ_j (x*_j - x_ij)² → dominated by large-unit feature
  Standardize: z_ij = (x_ij - x̄_j)/s_j; balance (×1000 EUR) vs pay_0 (integer)
  ```

### A12. **Random-forest mtry defaults — direction matters** (filter B, C)
- One-liner: Classification mtry = √p; regression mtry = p/3. NOT interchangeable.
- Hit by: P5d(i) — RF on classification with p=9 ⇒ mtry=3.
- Repo: `wiki/concepts/random-forest.md:22, 58-59, 102, 111`.
- Cheat-sheet form:
  ```
  Random forest defaults:
    Classification: mtry = √p     (e.g., p=9 → 3)
    Regression:     mtry = p/3   (e.g., p=12 → 4)
  ntree: NOT a tuning parameter — "large enough" (e.g. 500), variance shrinks
  Terminal node size: small (1 classif / 5 regr); rely on averaging not pruning
  mtry small ⇒ trees more decorrelated ⇒ Var = ρσ² + (1-ρ)σ²/B drops
  Bagging = RF with mtry = p
  ```

### A13. **Boosting B is a real tuning parameter — overfits!** (filter B, C)
- One-liner: Unlike RF (where larger B always reduces variance), boosting with too-large B fits noise residuals. Use CV / early stopping.
- Hit by: P4d(i,ii).
- Repo: `wiki/concepts/boosting.md:66-67, 93`.
- Cheat-sheet form:
  ```
  Boosting hyperparams: B (#trees), d (depth/interaction-order), η (shrinkage)
  B too small → underfit (bias); B too large → OVERFIT (unlike RF)
  η ∈ [0.001, 0.1]; halving η ≈ doubles required B
  d=1 (stump) = additive; d ∈ {1,2,4} typical (interactions up to order d)
  Boosting attacks BIAS via sequential weak learners; RF attacks VARIANCE
  ```

### A14. **Sensitivity/specificity definitions in problem-specific words** (filter B)
- One-liner: Sensitivity = TPR = "of true positives, what fraction did we catch." Specificity = TNR = "of true negatives, what fraction did we correctly clear." Lower threshold ⇒ ↑sens, ↓spec, move up-right on ROC.
- Hit by: P2g, P5a(iii,iv), P5b(iii), P5c(ii).
- Repo: `wiki/concepts/sensitivity-specificity.md`; ROC at `wiki/concepts/roc-auc.md:30-58`.
- Cheat-sheet form:
  ```
  Sens = TP/(TP+FN) = P(ŷ=1 | y=1)       ROC y-axis (= TPR)
  Spec = TN/(TN+FP) = P(ŷ=0 | y=0)       1-spec = FPR (ROC x-axis)
  Error = (FP+FN)/n;  Accuracy = (TP+TN)/n
  Threshold ↓: more flags ⇒ sens ↑, spec ↓; point moves UP-RIGHT on ROC
  AUC = P(score(positive) > score(negative)); 0.5 = chance, 1 = perfect
  AUC trap: equal AUC ≠ same curve shape (curves can integrate to same area)
  AUC stable under class imbalance, accuracy is not
  ```

### A15. **F-test naming — but not its mechanics** (filter A)
- One-liner: Joint test of categorical levels = **F-test** (partial F / ANOVA of reduced-vs-full). Prof said: don't compute the statistic; just name it and interpret the printout.
- Hit by: P4a(iii). NOTE: prof explicitly said "won't ask you to compute F" — so this is interpretation-only.
- Repo: `docs/scope.md:64` ("F-test mechanics" — out); `exam_analysis.md:359, 382`; `wiki/book-deltas/m03-linreg.md:418-424`.
- Cheat-sheet form:
  ```
  Joint test of all dummies for a K-level categorical (H₀: all βs = 0):
    Use the F-test (partial F / ANOVA test of reduced vs full model)
  Reject if p < α ⇒ "fuel is jointly associated with price, after
    controlling for other predictors"
  Marginal t-tests can each be insignificant while joint F still rejects
  (mechanics: known to prof as out-of-scope; you just NAME and INTERPRET)
  ```

### A16. **Class imbalance — accuracy is misleading** (filter B, C)
- One-liner: With P(Y=1)=0.20, always-predict-0 gets 80% accuracy. Use sens/spec pair, AUC, F1, or cost-sensitive metric.
- Hit by: P5e (both parts).
- Repo: `exam_analysis.md:209-211, 370`.
- Cheat-sheet form:
  ```
  Imbalanced classes: naive (always-majority) accuracy = max(π_k)
  Better metrics:
    (sens, spec) pair — separates per-class performance
    AUC — threshold-independent
    F1 = 2·prec·rec/(prec+rec) — balances precision & recall
    Cost-sensitive: pick threshold to hit target sens/spec given costs
  ```

### A17. **CV-MSE pseudocode** (filter C, D)
- One-liner: Partition into K folds; for each fold k, train on the other K-1, predict on fold k, get MSE_k; average. SE = sd(MSE_k)/√K (or per slide-deck convention, unscaled sd).
- Hit by: P4c(i).
- Repo: `wiki/concepts/k-fold-cv.md` (linked); also `exam_analysis.md:346-349`.
- Cheat-sheet form:
  ```
  K-fold CV(λ):
    Partition train into F_1,...,F_K (≈ equal size)
    For k = 1,...,K:
      Fit model on ⋃_{j≠k} F_j at λ
      MSE_k(λ) = (1/|F_k|) Σ_{i∈F_k} (y_i - ŷ_i^(-k))²
    CV(λ) = (1/K) Σ MSE_k(λ)
    SE = sd({MSE_k}) / √K
  K = 5 or 10 standard. LOOCV = K=n.
  Bias–var: validation set = high bias / low var; LOOCV = low bias / high var
  Sweet spot: K=5/10. (Per prof: "less variance is usually what wins.")
  ```

### A18. **CI shortcut from coefficient table** (filter C)
- One-liner: For "approximate 95% CI" with large df, just use β̂ ± 1.96·SE. Reading a coef table is high-yield exam template.
- Hit by: P4a(ii).
- Repo: `wiki/book-deltas/m03-linreg.md:380-385`.
- Cheat-sheet form:
  ```
  β̂_j ± t_{1-α/2, n-p-1} · SE(β̂_j)  ≈  β̂_j ± 1.96·SE  (large df)
  PI: same but variance has +1·σ²  (always wider than CI by σ-floor)
  Interpretation: "for an electric vs petrol (otherwise identical),
    price is on average β̂ thousand EUR higher, 95% CI [low, high]"
  ```

---

## Lower-priority but useful

### B1. **ReLU forward pass mechanics** (filter A)
- z = b + Σ w_j x_j, then max(0, z). Hit by P2c(ii). Simple but a common slip is forgetting bias in the pre-activation sum.

### B2. **Why nonlinear activations matter** (filter A)
- Without nonlinearity, composition of affine maps = affine. Universal approximation needs nonlinearity. Activations are NOT regularizers (P2h(iii) trap) and DO NOT make the loss convex (P2h(iv) trap).
- Repo: `wiki/concepts/activation-functions.md`, `wiki/concepts/universal-approximation.md`.

### B3. **LDA → linear because xᵀΣ⁻¹x cancels** (filter D)
- The prof flagged "where does the quadratic come from in QDA?" The answer is: QDA's Σ_k is class-specific, so xᵀΣ_k⁻¹x doesn't cancel across classes → quadratic. Hit by P2i.
- Repo: `exam_analysis.md:203-204, 299-309` (G2 procedural template).
- Compact form:
  ```
  LDA δ_k(x) = log π_k - ½μ_kᵀΣ⁻¹μ_k + xᵀΣ⁻¹μ_k  (linear in x — quadratic term cancels)
  QDA δ_k(x) = log π_k - ½log|Σ_k| - ½(x-μ_k)ᵀΣ_k⁻¹(x-μ_k)  (quadratic in x)
  Small n: LDA usually wins (QDA's class-specific Σ_k has too many params)
  Both GENERATIVE: model π_k and f_k(x), apply Bayes
  ```

### B4. **Variable importance vs partial-dependence — what each can/can't say** (filter C)
- Hit by P4d(iii), P5d(iii). VI = magnitude only; PDP = direction & shape. Prof prefers permutation VI over Gini (Mar 17).
- Repo: `wiki/concepts/variable-importance.md`, `wiki/concepts/partial-dependence-plots.md`.

### B5. **Reducible vs irreducible — σ² cannot be shrunk by more data** (filter B)
- Hit by P2a(iv). Common T/F trap.
- Repo: `wiki/concepts/reducible-vs-irreducible-error.md`.

### B6. **Ridge `less flexible` than OLS** (filter B)
- P2b(iii) trap. λ>0 ⇒ shrinkage ⇒ effective df < p+1 ⇒ less flexible than OLS, not more.
- Repo: `wiki/concepts/ridge-regression.md`.

### B7. **OLS test RSS can go either direction when adding terms** (filter B)
- P4b(ii). Training RSS monotone-down; test RSS depends on whether new feature carries signal vs. just adds variance.
- Repo: `exam_analysis.md:155`.

### B8. **Variance-of-bagged-average formula** (filter A)
- ρσ² + (1-ρ)σ²/B — does NOT → 0 if ρ > 0. Motivates RF's mtry decorrelation.
- Repo: `exam_analysis.md:192`; `wiki/concepts/bagging.md`, `wiki/concepts/random-forest.md`.

### B9. **OOB error ≈ 1/3 hold-out** (filter A)
- Free validation set; P(observation OOB for a given tree) → 1/e ≈ 0.368. Not directly tested here but standard fact.
- Repo: `exam_analysis.md:208`.

### B10. **Practical vs statistical significance** (filter B)
- P4a(iv): mpg p=0.135 not significant in this model, but collinearity with engine/hp may absorb effect. Tying conceptually to "significance is just sample size" (prof's framing).
- Repo: `wiki/book-deltas/m03-linreg.md:457-459`.

---

## Things explicitly in ISLP (skipped — find at exam time)

- Odds ↔ probability formulas (ISLP §4.3.1, eq 4.4) — though worth memorizing for speed.
- Confusion-matrix metric definitions (ISLP §4.4.2).
- ROC axes / AUC definition (ISLP §4.4.2, Fig 4.8).
- Ridge / Lasso objectives and L1-corner geometry (ISLP §6.2, Fig 6.7).
- Hierarchical clustering algorithm + linkage table (ISLP §12.4.2, Table 12.3).
- PCA eigendecomposition / scree / PVE (ISLP §12.2).
- K-fold CV definition (ISLP §5.1.3).
- LDA/QDA discriminant scores (ISLP §4.4.1–4.4.2).
- Tree/RF/boosting algorithms at high level (ISLP §8.1–8.2).
- GAM additive form (ISLP §7.7).
- Cubic-spline truncated-power basis (ISLP §7.4.3, eq 7.10).
- Bias-variance qualitative discussion (ISLP §2.2.2).
- One-standard-error rule (ISLP §6.1.3).
- F-statistic definition for joint hypothesis (ISLP §3.2.2, eq 3.23) — only the **name** is needed; the prof said he won't ask the mechanics.
