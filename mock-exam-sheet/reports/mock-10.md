# Mock Exam 10 — Cheat-sheet candidates

## Question summary

- **P1 (10%)** — 10 fill-in-the-blank concept terms (parametric, prediction/inference, irreducible error, double descent, generative, LDA, bagging, boosting, nested CV).
- **P2 (28%)** — T/F + short-numeric grab bag: (a) bias–variance & benign overfitting, (b) CV one-SE numeric + autocorrelation pitfall, (c) bootstrap pairs vs residuals + percentile CI, (d) backprop / SGD / learning rate, (e) collinearity & dummy-variable trap, (f) AdaBoost / GBM / XGBoost flavors, (g) logistic regression with interaction (group-specific odds factors, predicted prob, β-not-probability trap), (h) NN regularization (label smoothing numeric, dropout train/test), (i) PCA (PVE, score, standardization-not-invariant).
- **P3 (16%)** — Derivations: (a) LDA discriminant + boundary + worked plug-in + LDA→QDA gap; (b) k-fold CV with one-SE rule pseudocode; (c) backprop on sigmoid + BCE (∂L/∂z₂ = p̂ − y cancellation).
- **P4 (20%)** — Diabetes regression: OLS w/ quadratic + interaction + near-collinear (s1, s2) pair, lasso vs OLS interpretation, GBM (ν, d, M) grid, GAM df counting, random forest mtry/importance limitations.
- **P5 (26%)** — SA heart disease classification: logistic w/ ldl:famhist interaction, AdaBoost α derivation under exponential loss, NN parameter count + regularization mechanisms, confusion-matrix metrics + asymmetric costs, KNN scaling.

## High-priority surfacing

- **One-SE rule mechanics (numeric + decision)** (B/C) — Compute SE = s/√k from per-fold MSEs, then bound = CV(λ_min) + SE, then pick **largest** λ inside band. Q2b(i)(ii), Q3b(i), Q4b(iv) — mock-exam-10-solution.tex:102–117, 368–396. Compact: `SE = sd(fold MSEs)/√k; bound = CV_min + SE; λ* = max{λ : CV(λ) ≤ bound}`.

- **Why one-SE-SE is "not quite valid"** (A/B) — Per-fold MSEs are NOT independent (training sets overlap heavily) so s/√k under-estimates true SE. Q2b(iv) — mock-exam-10-solution.tex:129–133. Compact: "folds share most training data → positively correlated → s/√k is a stand-in, not real SE."

- **Random-folds CV breaks under temporal autocorrelation** (B) — Random partitioning of time-series data lets future leak into training; remedy is **rolling-origin / blocked folds**, NOT bigger k. Q2b(iii)B — mock-exam-10-solution.tex:122–127.

- **Paired vs residual bootstrap choice** (C) — With **random X / observational design**, prefer **paired (case)** bootstrap: residual bootstrap assumes correct functional form + i.i.d. exchangeable residuals (fails under heteroscedasticity / mis-specification). Q2c(i) — mock-exam-10-solution.tex:138–144.

- **Percentile CI vs normal-approx CI** (A/C) — Percentile 95% CI = `[θ*_(0.025), θ*_(0.975)]` (empirical quantiles), NOT `θ̂ ± 1.96·SE*`. Bootstrap does NOT auto-correct bias (need BCa). Q2c(ii)(iii)C — mock-exam-10-solution.tex:146–158.

- **Bootstrap B sizing** (A) — B ∈ [1000, 10000] for SE; more for quantile CIs. Q2c(iii)B — mock-exam-10-solution.tex:154–155.

- **Learning-rate sanity check** (B) — Standard NN LR range `10⁻⁴ to 10⁻²`; η=2 diverges, not "speeds up." Q2d(iii) — mock-exam-10-solution.tex:172–174.

- **Dummy-variable trap remedy** (B/C) — Standard fix is **drop one dummy (reference encoding) or drop the intercept** — NOT a ridge penalty (ridge stabilizes numerics but doesn't structurally identify K coefficients). Q2e(ii) — mock-exam-10-solution.tex:187–193.

- **Collinearity covariance cancellation in joint contribution** (B) — Individual `Var(β̂_j)` inflates, but `Var(β̂_1 X_1 + β̂_2 X_2)` includes large **negative covariance** that cancels much of it → predictions stay stable. Why F-tests can be significant while marginal t-tests are not. Q2e(iii), Q4a(ii)(iii) — mock-exam-10-solution.tex:194–199, 461–481.

- **Boosting trade `M · ν ≈ const`** (A/C) — Halving ν roughly doubles required M*. Verified in mock: ν: 0.10→0.05→0.01 ↔ M*: 700→1400→6500. Q2f(i), Q4c(ii) — mock-exam-10-solution.tex:208–211, 567–572.

- **GBM depth canonical range d ∈ {1,…,8}** (C) — Deep trees defeat "many weak learners"; stumps (d=1) are textbook default. d ∈ {10,…,50} is a TRAP. Q2f(ii) — mock-exam-10-solution.tex:212–215.

- **XGBoost's three lecture-named differences** (A) — (a) leaf-weight `γ|T| + ½λΣw_j²` penalty in split objective; (b) Newton/second-order Hessian split scoring; (c) row + column subsampling. Q2f(iii) — mock-exam-10-solution.tex:217–219.

- **Logistic-β-is-NOT-probability trap** (B) — β̂_j is change in **log-odds** per unit x_j; probability change is nonlinear (max near p̂=0.5). Q2g(iii), Q5a(i)(ii) — mock-exam-10-solution.tex:239–243.

- **Group-specific odds-ratio with x:group interaction** (B/C) — Odds factor for unit ↑ in x is `exp(β_x + β_{x:group_k})`, NOT `exp(β_x)` for all groups. Q2g(i), Q5a(i) — mock-exam-10-solution.tex:225–232, 638–647.

- **Label smoothing with C=2 binary case** (A) — Treat as one-hot (0,1); ε=0.10, off-target=ε/(C-1)=0.10, on-target=0.90 → (0.10, 0.90). Bounds gradient from mislabeled examples. Q2h(i), Q5c(v) — mock-exam-10-solution.tex:248–253, 782–792.

- **Dropout train-vs-test rescaling** (A/B) — At test time **no masking**, weights/activations rescaled so expected unit input matches train. Early stopping uses **validation** loss, NOT training loss. Q2h(ii), Q5c(iv) — mock-exam-10-solution.tex:255–262, 773–779.

- **PCA standardization is NOT scale-invariant** (B) — Without standardizing, large-variance variables dominate first PCs. λ_j = empirical variance of score z_j. Total variance of standardized data = p. Q2i(iii) — mock-exam-10-solution.tex:283–290.

- **LDA boundary linearity reason** (D) — Quadratic `x^T Σ^{-1} x` cancels between δ_A − δ_B **because Σ is shared**. For QDA, `Σ_k` differs → the `x^T(Σ_A^{-1} − Σ_B^{-1})x` term survives → quadratic boundary. Q3a(i)(ii)(iv) — mock-exam-10-solution.tex:305–361.

- **Sigmoid+BCE cancellation: ∂L/∂z₂ = p̂ − y** (A/D) — The `p̂(1−p̂)` in `∂L/∂p̂` cancels exactly with `σ′(z₂) = p̂(1−p̂)`. Why sigmoid+BCE is preferred over sigmoid+SE (no vanishing gradient at saturation). Q3c(i) — mock-exam-10-solution.tex:410–426.

- **GBM pseudocode (squared-error)** (A/C) — Init `f^(0) = mean(y)`; at step m fit tree to residuals r_i = y_i − f^(m−1)(x_i) (= neg-gradient under SSE); update `f^(m) = f^(m−1) + ν · h̃_m`. Q4c(i) — mock-exam-10-solution.tex:556–565.

- **GBM tunes M, but RF does NOT** (C) — Too-large M overfits in boosting (test error rises); random forests don't overfit in B (only variance reduces). "Set M=10000 to be safe" is unsound for GBM, fine for RF. Q4c(iv) — mock-exam-10-solution.tex:583–587.

- **Cubic regression spline df counting** (A) — `bs(x, knots=K)` cubic spline df = K + d + 1 = K + 4 basis functions; minus intercept → K + 3. For K=2: 5 df. Q4d(i) — mock-exam-10-solution.tex:593–596.

- **mtry default for regression RF = p/3** (vs classification √p) (A/C) — For p=9: mtry=3. Reason: decorrelate trees (bagging only reduces variance when trees are weakly correlated). Q4d(iii) — mock-exam-10-solution.tex:611–616.

- **RF importance plot: no sign, no SE, no functional form** (B) — Reports magnitude only; OLS coefficient supplies sign + SE + form. Q4d(iv) — mock-exam-10-solution.tex:618–624.

- **AdaBoost α derivation under exponential loss** (D) — Split sum into correct vs misclassified, differentiate `Q(α) = W_C e^{−α} + W_W e^{+α}`, set to 0 → `e^{2α} = W_C/W_W` → `α* = ½ log((1−err)/err)`. AdaBoost.M1 absorbs ½ into weight update. Q5b(i) — mock-exam-10-solution.tex:698–715.

- **AdaBoost α < 0 when err > 0.5** (B/C) — Base learner worse than random → α negative → its sign flipped in ensemble vote. Q5b(iii) — mock-exam-10-solution.tex:722–727.

- **NN parameter count includes biases** (A) — Layer `(n_in → n_out)`: `n_in·n_out + n_out`. For 7→20→10→1: 160 + 210 + 11 = 381. Forgetting biases gives 350 (trap). Q5c(i) — mock-exam-10-solution.tex:740–747.

- **Asymmetric-cost classifier selection** (B/C) — Costly FN screening → pick on **sensitivity** (= TP/(TP+FN)), not accuracy. Logistic 0.47 vs AdaBoost 0.66 → deploy AdaBoost despite lower specificity. Q5d(iii) — mock-exam-10-solution.tex:826–833.

- **Naive-classifier baseline = 1 − base rate** (B) — Predict-majority error = minority-class fraction. Reveals accuracy alone is poor when classes imbalanced (naive has 0% sensitivity but 66% accuracy here). Q5d(ii) — mock-exam-10-solution.tex:820–824.

## Lower-priority but useful

- **Benign overfitting / SGD min-norm bias** (A) — Implicit regularization of SGD steers over-parameterized nets toward minimum-norm interpolator. Q2a(iv), Q5c(vi) — mock-exam-10-solution.tex:89–93, 794–799.

- **σ² irreducible error doesn't depend on n** (B) — Doubling n shrinks `Var(f̂(x_0))` like 1/n but NOT σ². Q2a(ii) — mock-exam-10-solution.tex:78–82.

- **LOOCV: low bias, high variance because folds overlap** (A) — Q2b(iii)A — mock-exam-10-solution.tex:119–122.

- **Mini-batch SGD gradient: unbiased, variance ∝ 1/m** (A) — Q2d(ii) — mock-exam-10-solution.tex:166–171.

- **Lasso objective: intercept NOT penalized** (C) — Standard convention; intercept eliminated by centering response, X standardized for scale-invariance of L1. Q4b(i) — mock-exam-10-solution.tex:514–522.

- **Lasso path depends on dummy reference level** (B) — Penalty applied to contrast coefficients, not group-symmetric (need group lasso for that); a zeroed dummy means "level indistinguishable from reference," not "predictor irrelevant." Q4b(ii) — mock-exam-10-solution.tex:524–530.

- **Causal-vs-correlation prof framing** (B) — Prof's verbatim: "fancy correlations, not causal claims." Negative β_obesity could be confounding via adiposity/bmi/bp. Q5a(v) — mock-exam-10-solution.tex:685–692.

- **KNN with raw mixed-scale features dominated by largest-range variable** (B/C) — Standardize before KNN. Q5e(ii) — mock-exam-10-solution.tex:843–849.

- **Multiple p-values under collinearity = conditional tests** (B) — Each tests β_j=0 given other near-collinear partner already in model; can both be insignificant individually while joint F-test is significant. Q4a(iii) — mock-exam-10-solution.tex:473–481.

- **Forgetting BMI² term when computing Δprog** (B) — Going BMI=1 → BMI=2 contributes 3.8·(2²−1²) = 11.4 from quadratic, in addition to linear 24.0. Q4a(v) — mock-exam-10-solution.tex:498–509.

## Things explicitly in ISLP (skipped)

- Bias–variance decomposition (ISLP 2.2.2).
- LDA discriminant derivation up to the standard form (ISLP 4.4).
- Sigmoid σ′(z) = σ(z)(1−σ(z)) identity (ISLP 10).
- Lasso objective form and CV-min selection (ISLP 6.2).
- Random forest mtry concept (ISLP 8.2.2).
- Sensitivity / specificity / confusion matrix definitions (ISLP 4.4.2).
- KNN classifier definition and distance/scaling pitfall (ISLP 2.2.3 / 4.7.6).
- AdaBoost weight-update algorithm pseudocode (ISLP 8.2.3).
- Backprop chain rule structure (ISLP 10.7).
- Standard cubic spline basis function count (ISLP 7.4.3).
