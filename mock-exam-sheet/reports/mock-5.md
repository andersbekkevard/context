# Mock Exam 5 — Cheat-sheet candidates

## Question summary

- **P1** (10p): fill-in-blank vocab across modules (collinearity, shrinkage, PCs, dropout, early stopping, label smoothing, mini-batches, nested CV, AdaBoost, gradient boosting).
- **P2** (28p): T/F + short numeric on bias-variance / double descent (a), CV pitfalls (b), bootstrap & SE/CI (c), NN regularization (d), SGD & backprop (e), boosting tuning + α formula (f), logistic odds & interactions (g), PCA eigenvalues/PVE (h), LDA/QDA/spline d.f. (i).
- **P3** (16p): math/pseudocode — bias-variance derivation (a), k-fold CV + nested CV + screening-leak (b), bootstrap by hand + parametric-vs-nonparametric SE disagreement (c), complete-linkage hierarchical clustering by hand (d).
- **P4** (22p, regression — energy/cost): polynomial+interaction OLS table interp (a), collinearity diagnosis (b), lasso CV with λ_min vs λ_1SE (c), gradient-boosting pseudocode + tuning + PDP (d).
- **P5** (24p, classification — readmission): logistic with diabetes×insulin interaction + η/p̂ calc (a), confusion-matrix metrics + threshold direction (b), AdaBoost pseudocode + α/weight update by hand + model comparison + exp-loss derivation (c), NN param count + ReLU compute + 3-step backprop chain + regularization choices + SGD implicit reg (d).

## High-priority surfacing

- **AdaBoost α formula & weight update** (A,D) — α_m = log((1−err_m)/err_m); w_i ← w_i · exp(α_m · 1{misclass}) — Q2f(i), Q5c(i)–(ii) — `mock-exam-5.tex:114`, `mock-exam-5-solution.tex:168, 842-851` — `α=log((1-err)/err); err=0.20 ⇒ α=log4≈1.39; correct: ×1, wrong: ×e^α`.
- **AdaBoost as forward-stagewise on exponential loss** (A,D) — L(y,f)=e^{−yf}, y∈{−1,+1} — Q5c(iv) — `mock-exam-5-solution.tex:907-913` — `AdaBoost ⇔ forward stagewise additive, exp loss`.
- **Gradient = residual identity for squared loss** (D) — −∂L/∂f|_{f=f̂^{(m−1)}} = y_i − f̂^{(m−1)}(x_i) = r_i^{(m)} — Q2f(ii), Q4d(ii) — `mock-exam-5-solution.tex:176, 688` — `L=½(y−f)² ⇒ −∂L/∂f = r; boosting = fit tree to residual`.
- **Trees in boosting: (M, ν) coupled inversely** (C) — smaller ν ⇒ more trees needed; halving ν ≈ doubles required M — Q2f(iii)C, Q4d(iii) — `mock-exam-5-solution.tex:189-191, 702-704` — `ν↓ ⇒ M↑ (jointly tuned)`.
- **RF vs boosting on tree count** (C) — RF: B not a real tuning parameter, monotone non-increasing test error in B, "use enough" 500–1000. Boosting: M is real, overshoots ⇒ overfit — Q2f(iii)A-B, Q4d(iv) — `mock-exam-5-solution.tex:183-188, 709-714` — `RF: B↑ safe; Boost: needs CV or early stopping`.
- **XGBoost = 2nd-order + leaf L1/L2 penalty** (A) — Newton-style Hessian info + leaf-weight regularization + sparsity-aware splits — Q2f(iii)D — `mock-exam-5-solution.tex:192-195` — `XGBoost: gradient+Hessian, L1/L2 on leaves`.
- **One-SE rule precise statement** (C) — λ_1SE = max{λ : CV(λ) ≤ m* + SE*} where m* is CV-min and SE* is its CV-SE — Q4c(ii) — `mock-exam-5-solution.tex:630-639` — `λ_1SE = largest λ with CV-MSE within 1 SE of min`.
- **Lasso not invariant to dummy reference level** (B,C) — L1 penalty applied per-column; reparam ⇒ different sparsity. Use group lasso for factors — Q4c(i) — `mock-exam-5-solution.tex:617-627` — `lasso path depends on reference level; group lasso = clean fix`.
- **Hierarchical-principle trap: don't drop low-order if high-order kept** (B,C) — high p-value on β_age with β_age² significant ⇒ keep both; same for main effects under interaction — Q4a(iv), Q5a(iv) — `mock-exam-5-solution.tex:544-551, 794-801` — `interaction in ⇒ main effects stay; quad in ⇒ linear stays`.
- **Interaction main-effect conditional meaning** (B) — β_heatpump = effect at x̃_area=0, NOT marginal/average — Q4a(v) — `mock-exam-5-solution.tex:554-563` — `β_main = effect when interacting var = 0; total effect = β_main + β_int·x̃`.
- **Smoker per-year effect under interaction** (B) — for smoker, age effect = β_age + β_age:smoker (sum, not just interaction) — Q2g(ii) — `mock-exam-5-solution.tex:204-210` — `Δlogodds = β_main + β_int; trap = quoting e^{β_int}`.
- **Logistic β → probability is NOT additive** (B) — ∂p/∂x_j = β_j · p(1−p); near p=0.5 ≈ 0.25 β_j; saturates at extremes — Q2g(iii) — `mock-exam-5-solution.tex:213-218` — `Δp ≈ β·p(1−p); only β maps cleanly to odds, not probability`.
- **Confusion-matrix layout convention (prof)** (B) — rows = actual, columns = predicted; sens = TP/(TP+FN) = TP/row-sum-actual-pos — Q5b(i) — `mock-exam-5.tex:380-386`, `mock-exam-5-solution.tex:813-816` — `Sens=TP/P_actual; Spec=TN/N_actual; Err=(FP+FN)/n`.
- **Threshold-lowering direction** (B) — threshold ↓ ⇒ more positives ⇒ Sens↑, Spec↓ — Q5b(ii) — `mock-exam-5-solution.tex:820-823` — `t↓ ⇒ Sens↑, Spec↓ (monotone)`.
- **Random CV folds leak under temporal autocorrelation** (B,C) — random shuffling leaks past/future; k=10 doesn't help; need forward-chaining / rolling-window — Q2b(iii) — `mock-exam-5-solution.tex:97-100` — `Temporal data ⇒ rolling-window CV, not random k-fold`.
- **CV screening leak (filter outside fold)** (B,C) — selecting top-25 predictors via marginal corr on full data BEFORE CV ⇒ biased downward; fix: screen INSIDE each fold — Q3b(iii) — `mock-exam-5-solution.tex:390-399` — `Feature selection must be inside CV fold; else biased-low test error`.
- **Bootstrap inclusion probability** (D) — P(not in) = (1−1/n)^n → 1/e ≈ 0.368; for n=10: P(in) = 1 − (9/10)^{10} ≈ 0.651 — Q2c(i), Q3c(i) — `mock-exam-5-solution.tex:105-107, 403-407` — `P(included for n=10) ≈ 0.651; asymptotic P(out) → 1/e`.
- **Bootstrap percentile CI specifies order statistics** (D) — for B=1000, 95% CI = [θ*_(25), θ*_(975)] (2.5%/97.5% quantiles of sorted bootstrap replicates) — Q2c(ii) — `mock-exam-5-solution.tex:110-114` — `Percentile 95% CI: [q_{0.025}, q_{0.975}] of {θ*_b}`.
- **Bootstrap quantifies variance, not bias** (B) — bootstrap distribution centered at θ̂, not at θ; doesn't correct bias — Q2c(iii)B — `mock-exam-5-solution.tex:119-122` — `Boot SE = sd across θ*_b; doesn't correct bias`.
- **Parametric vs nonparametric SE disagreement** (B,A) — Fisher SE assumes model correct + asymptotic normality; bootstrap captures misspecification & dependence — Q3c(iii) — `mock-exam-5-solution.tex:426-433` — `Boot SE > Fisher SE ⇒ likely model misspec or dependence`.
- **Complete linkage dendrogram by hand** (D) — at merge, new row entries = max over merged rows; final-merge height = max pairwise across all members — Q3d — `mock-exam-5-solution.tex:439-461` — `Complete: d(A,B) = max_{i∈A,j∈B} d_{ij}; trap = using min (single)`.
- **Backprop ≠ optimizer** (B,A) — backprop only computes ∂L/∂θ via chain rule (reuses forward activations); SGD/Adam does the update — Q2e(i) — `mock-exam-5-solution.tex:151-156` — `Backprop = gradient computation; SGD = parameter update`.
- **Dropout is training-time only** (B) — at test the full network is used; weights scaled by keep-prob (or inverted-dropout during training) — Q2d(i) — `mock-exam-5-solution.tex:133-137` — `Dropout off at test; always-on dropout = stochastic predictions`.
- **L2 weight decay increases training error** (B) — penalty constrains optimum ⇒ training fit worse; purpose is reducing test error — Q2d(iii) — `mock-exam-5-solution.tex:142-146` — `Weight decay: trainErr↑, testErr↓ (variance trade)`.
- **NN parameter count formula** (D) — each layer: (#inputs + 1) × #outputs (+1 = bias). 7→10→1: 8·10 + 11·1 = 91 — Q5d(i) — `mock-exam-5-solution.tex:917-925` — `Params per layer = (n_in+1) × n_out`.
- **3-step backprop chain** (D) — ∂L/∂α_{kj} = −(y_i − f̂)·β_k·g′(v_{ik})·x_{ij} — Q5d(iii) — `mock-exam-5-solution.tex:957-963` — `output-err → β_k → g′(v) → x_{ij}`.
- **SGD implicit regularization (prof hobbyhorse)** (A,C) — mini-batch gradient noise biases toward flat minima ⇒ better generalization; explains over-param SGD without explicit decay — Q5d(v) — `mock-exam-5-solution.tex:992-999` — `SGD noise → flat minima → free regularization`.
- **Cubic spline parameter count: K + 4 (with intercept)** (D) — truncated-power basis: {1, x, x², x³} + K knot terms; equivalently 3 spline terms + K + intercept — Q2i(iii) — `mock-exam-5-solution.tex:258-264` — `Cubic spline: K interior knots ⇒ K+4 params incl. intercept`.
- **PCA on standardized data: total variance = p** (D) — Σλ_j = p; PVE_k = λ_k/p — Q2h(i) — `mock-exam-5-solution.tex:221-224` — `Standardized PCA: total var = p; PVE = λ/p`.
- **PCA is unsupervised: loadings don't see y** (A) — eigenvectors of X^T X / SVD of X — limitation that PCs may be uncorrelated with response (PLS = supervised alternative) — Q2h(iii) — `mock-exam-5-solution.tex:238-242` — `PCA blind to y; PLS = supervised version`.

## Lower-priority but useful

- **Adjusted R² barely moves under collinearity** (B) — bonus collinearity tell: 0.43 → 0.44 when "new" predictor adds same info — Q4b(i) — `mock-exam-5-solution.tex:573-577` — `Adj R² flat after adding collinear predictor ⇒ same info`.
- **VIF formula** (A) — VIF ≈ 1/(1−r²); r=0.92 ⇒ VIF ≈ 6.5 — Q4b(ii) — `mock-exam-5-solution.tex:583-585` — `VIF = 1/(1−R²_j); large ⇒ inflated Var(β̂)`.
- **Collinearity hurts inference, not prediction** (B,C) — sum β_area·area + β_rooms·rooms identified even if individual coefs aren't (assuming test stays in same collinear region) — Q4b(iii) — `mock-exam-5-solution.tex:587-593` — `Collinear preds: stable ŷ, unstable individual β̂`.
- **LOOCV bias/variance vs k-fold** (B) — LOOCV: lower bias (n−1 ≈ n), higher variance (correlated fold errors) — Q2b(i) — `mock-exam-5-solution.tex:89-92` — `LOOCV: bias↓, var↑ vs 10-fold`.
- **Quadratic parabola vertex computation** (D) — marginal age effect = −0.05 + 0.60·x̃ ⇒ minimum at x̃ ≈ 0.083 — Q4a(iii) — `mock-exam-5-solution.tex:536-541` — `For β₁x + β₂x²: ∂/∂x = β₁ + 2β₂x; vertex at −β₁/(2β₂)`.
- **Joint contribution table for 2-way binary interaction** (B) — (0,0):0, (1,0):β_a, (0,1):β_b, (1,1):β_a+β_b+β_ab; the (1,1) cell dominated by interaction in readmission case — Q5a(ii) — `mock-exam-5-solution.tex:748-761` — `4-cell table for 0/1 × 0/1 interaction`.
- **Logistic linear predictor + sigmoid by hand** (D) — η = β_0 + Σβ_j x_j; p̂ = 1/(1+e^{−η}) — Q5a(iii) — `mock-exam-5-solution.tex:776-788` — `η=0.38 ⇒ p̂ = σ(0.38) ≈ 0.594`.
- **Bias-variance cross-term vanishes by independence** (D) — ε ⊥ D_train + E[ε]=0 ⇒ E[ε·(f − f̂)] = 0 — Q3a(i) — `mock-exam-5-solution.tex:278-291` — `Cross term: ε ⊥ training ⇒ E[ε·(...)] = E[ε]·E[...] = 0`.
- **Double descent: identity holds, variance non-monotone** (A,B) — decomposition is an algebraic identity for ANY estimator; second descent = variance itself drops past interpolation — Q2a(ii), Q3a(iv) — `mock-exam-5-solution.tex:75-80, 350-354` — `Identity holds always; variance term can fall past interpolation peak`.
- **Lasso CV interpretation** (C) — small test-MSE gain + few dropped coefs ⇒ dropped predictors had little signal; their OLS estimates were noisy — Q4c(iii) — `mock-exam-5-solution.tex:643-648` — `lasso small gain + sparser ⇒ dropped preds were noise`.
- **Lasso vs OLS sanity** (C) — λ_1SE-lasso MSE (0.203) still beats OLS (0.205) — simplification doesn't regress past starting point — Q4c(iv) — `mock-exam-5-solution.tex:655-660` — `Even λ_1SE-lasso ≤ OLS test MSE`.
- **Boosting beats additive linear ⇒ interactions present** (C) — gradient boosting test MSE << OLS/lasso ⇒ underlying relationship has interactions a linear-additive model can't capture — Q4d(v) — `mock-exam-5-solution.tex:717-722` — `GB >> OLS/lasso gap ⇒ interactions; use PDP/var-importance for interpretability`.
- **Label smoothing target shape** (A) — replace (0,0,1,0) with (ε/3, ε/3, 1−ε, ε/3) — Q2d(ii) — `mock-exam-5.tex:99`, `mock-exam-5-solution.tex:138-141` — `One-hot → softened; ε small (~0.05); reduces over-confidence`.
- **Mini-batch SGD unbiased gradient** (D) — uniform random batch ⇒ E[batch gradient] = full-data gradient — Q2e(ii) — `mock-exam-5-solution.tex:157-159` — `Random mini-batch ⇒ unbiased estimator of full gradient`.
- **Bootstrap SE formula** (D) — SE_boot = sqrt((1/(B−1))·Σ(θ*_b − θ̄*)²) — Q3c(ii) — `mock-exam-5-solution.tex:417-418` — `SE_boot = sd across bootstrap replicates`.

## Things explicitly in ISLP (skipped)

- Logistic odds factor exp(β·Δx) — direct ISLP §4.3.
- Cross-validation basic mechanics / k-fold split — ISLP §5.1.
- Bias-variance core formula — ISLP §2.2.
- Ridge regression formula and bias-variance — ISLP §6.2.
- Lasso L1 penalty, sparsity, geometry — ISLP §6.2.
- LDA linear / QDA quadratic boundary distinction — ISLP §4.4.
- Random-forest tree-count "use enough" heuristic — ISLP §8.2 (still worth flagging as contrast vs boosting).
- PCA eigenvalue / PVE / cumulative scree — ISLP §12.2.
- Gradient boosting pseudocode (B, ν, d) — ISLP §8.2.3.
- Standard k-fold-CV pseudocode — ISLP §5.1.
- Standard backprop chain rule for one hidden layer — ISLP §10.7.
- Standard confusion-matrix definitions — ISLP §4.4.3.
- Bootstrap basic procedure — ISLP §5.2.
