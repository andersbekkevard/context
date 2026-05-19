# Mock Exam 8 — Cheat-sheet candidates

## Question summary

- **P1 (10 pt)** — Fill-in-the-blank concepts: discriminative/generative, LDA boundary linearity & pooled Σ, collinearity, ridge, RF random subset of m, OOB (~37%), backprop, label smoothing, bootstrap.
- **P2 (28 pt)** — T/F + short numeric across nine themes: bias–variance of shrunken mean (with double-descent identity claim), LOOCV vs k-fold + closed form, bootstrap CI/percentile/paired, mini-batch SGD (η, ReLU vs sigmoid), AdaBoost α_m sign with err>0.5, gradient-boosting (M,ν) coupling, XGBoost specifics, dropout + label smoothing soft target, sigmoid+intercept arithmetic, logistic interaction odds (treatment), LDA/QDA + prior direction trap, naïve Bayes = diagonal LDA, PCA cumulative PVE + score + standardization trap.
- **P3 (16 pt)** — (a) Full LDA discriminant derivation + 2D concrete numerics + boundary solve + prior shift direction; (b) one-epoch SGD pseudocode + implicit-regularization framing; (c) k-fold CV pseudocode with SE, one-SE rule with explicit "simpler" direction per method, and band selection on a table.
- **P4 (22 pt)** — Bike-share OLS with poly + interaction + collinear pair (temp/feel_temp): parameter count, collinearity diagnosis (prediction vs interpretation), quadratic optimum, interaction-aware delta, hierarchy principle, heteroscedasticity, CI vs PI; ridge vs lasso under collinearity (geometry); one-SE rule; PCR naming + sparsity cost; gradient boosting (M, d, ν) trade-off table with overfitting trend and RF contrast; GAM df accounting + additivity limitation.
- **P5 (24 pt)** — Loan-default logistic with continuous–continuous interaction (DTI×credit_score): interaction-aware odds factors, "main effect at credit_score=0" trap, full linear-predictor calculation, threshold sensibility at 18% base rate, fancy-correlation/not-causal mantra, confusion matrix + sens/spec direction under threshold lowering; AdaBoost-from-exponential-loss derivation (weighted err reduction, closed-form α_m, weight update); NN parameter count + ReLU pre-activation + backprop chain rule for the three layers; dropout/early-stopping/label-smoothing flags.

## High-priority surfacing

- **Mini-batch SGD η range** (A/C) — Prof's *practical* learning-rate range is η ∈ [10⁻³, 10⁻¹] with 0.01–0.1 typical; η=0.1 is *not* too small, η≈2 would diverge — mock-exam-8-solution.tex:163-165 — *Q2d(ii) trap*. Cheat: "η typical ≈ 1e-3…1e-1; 0.1 is fine, 2 diverges."
- **Dropout course default ≈ 0.2** (A/C) — Prof recommendation is ≈0.2, with 0.4+ regarded as too aggressive (over-drops → underfit, bias↑) — mock-exam-8-solution.tex:213, 1064-1069 — *Q2f(i)(C), Q5c(v)*. Cheat: "dropout 0.2 standard; 0.5 underfits."
- **Label smoothing soft target (0.95, 0.05) for ε=0.05** (A) — Binary hard (1,0) → (1−ε, ε); only enters training loss, not inference — mock-exam-8-solution.tex:215-216, 1056-1060 — *Q2f(ii), Q5c(iv)*. Compact: "ε=0.05 ⇒ (0.95, 0.05)."
- **Early stopping monitors VALIDATION loss not training loss** (B) — Training loss decreases monotonically, so it would never stop; the prof-flagged inversion — mock-exam-8-solution.tex:211-212 — *Q2f(i)(B)*. Compact: "early-stop = val loss, not train."
- **L2 weight decay raises training error** (B) — Constrained feasible region cannot have a lower min; weight decay trades training↑ for test↓ — mock-exam-8-solution.tex:216-220 — *Q2f(iii)*. Compact: "weight decay → train err ↑, test err ↓."
- **AdaBoost α_m sign-flip when err > 0.5** (B) — err=0.70 ⇒ α=log(0.3/0.7)≈−0.85; negative α *inverts* that classifier's vote in the sign-sum (still constructive: "wrong with structure") — mock-exam-8.tex:111, mock-exam-8-solution.tex:181-186 — *Q2e(i)*. Cheat: "α<0 when err>0.5 ⇒ inverted vote."
- **(M, ν) coupling in gradient boosting** (C) — Halving ν roughly *doubles* (not halves) trees needed; classic direction-of-effect inversion — mock-exam-8-solution.tex:190-194, 745-751 — *Q2e(iii), Q4c(iii)*. Cheat: "ν↓ ⇒ M↑ (proportional)."
- **XGBoost extras over vanilla GBM** (A) — (i) explicit L1 *and* L2 on per-leaf weights, (ii) second-order Taylor: splits use gradient g_i AND Hessian h_i — mock-exam-8-solution.tex:196-200 — *Q2e(iv)*. Cheat: "XGBoost = GBM + L1/L2 on leaves + Newton (g, h) splits."
- **LDA prior direction trap** (B) — Increasing π̂_k moves boundary *away from* class k and labels MORE area as k (sol calls (C) False; the question phrasing "toward class k → less area" is the inversion) — mock-exam-8-solution.tex:256-258, 401-405 — *Q2h(i)(C), Q3a(iv)*. Cheat: "π_k↑ ⇒ boundary moves *away* from class k; more area labeled k."
- **PCA is NOT scale-invariant** (B) — Predictor in large native units dominates PC1; standardization is what makes PCA "scale-invariant" — mock-exam-8-solution.tex:285-289 — *Q2i(iii)*. Cheat: "PCA scale-dependent; standardize first."
- **LOOCV-for-OLS shortcut formula** (A/D) — CV_(n) = (1/n) Σ ((y_i − ŷ_i)/(1 − h_ii))² where h_ii are hat-matrix diagonals — mock-exam-8-solution.tex:110-113 — *Q2b(ii)*. Cheat: write this verbatim; "(1−h_ii)² leverage adjustment."
- **CV-SE is "strictly speaking, not quite valid"** (A) — Per-fold MSEs are NOT independent (shared training observations across folds); SE formula = sd_k / √k is used anyway, one-SE rule robust — mock-exam-8.tex:88, mock-exam-8-solution.tex:118-123 — *Q2b(iv)*. Prof's exact wording flagged.
- **One-SE rule "simpler direction" per method** (C) — ridge: larger λ; polynomial: smaller degree; KNN: larger k. Then pick the simplest θ within m*+SE* of the minimum — mock-exam-8-solution.tex:507-518 — *Q3c(ii)*. Compact table:  λ↑, deg↓, k↑.
- **One-SE rule asymmetric (picks LARGEST λ in band)** (B/C) — Band [m*, m*+SE*]; common trap is picking the other tail (λ=0.01) instead of the most-regularized in-band value (λ=1) — mock-exam-8-solution.tex:521-544 — *Q3c(iii)*. Cheat: "1-SE always pushes toward *simpler*, not symmetric."
- **Hierarchy principle (interaction in ⇒ keep main effect)** (B/C) — Even when main-effect p-value is large (0.074), do NOT drop it if its interaction is in the model; the main effect's p-value is the test "at the other variable = 0" only — mock-exam-8-solution.tex:641-647 — *Q4a(v), Q5a(ii)*. Cheat: "hierarchy: keep mains under their interactions."
- **Collinearity hurts interpretation, NOT prediction** (B) — Diagnostic fingerprint: test MSE stable when one collinear partner dropped, but β-estimate swings massively; (X^TX)⁻¹ has huge diagonal entries in collinear direction — mock-exam-8-solution.tex:563-583 — *Q4a(ii)*. Cheat: "Collinearity: prediction stable, interpretation unstable."
- **Continuous×continuous interaction: main effect = "effect at the other variable = 0"** (B) — β̂₂=0.25 on DTI is NOT the average DTI effect; it's the DTI effect at credit_score=0 (outside data range!). Centering credit_score fixes the interpretation — mock-exam-8-solution.tex:830-836 — *Q5a(ii)*. Cheat: "main effect ≠ avg effect when interaction present; it's the effect at *the other = 0*."
- **Forgetting interaction in odds-multiplier calc** (B) — Standard trap: report exp(β̂₂) regardless of credit_score; correct is exp(β̂₂ + β̂_int · credit_score), so factor changes from 1.073 (score 600) to 1.010 (score 800) — mock-exam-8-solution.tex:803-822 — *Q5a(i)*. Cheat: "interaction-aware odds: exp(β_main + β_int · x_other)."
- **0.5 threshold inappropriate at 18% base rate** (C) — Sensitivity 0.50 at p=0.5 is poor in screening; cost asymmetry calls for *lower* threshold. Sens↑/spec↓ when threshold lowered (ROC up-and-to-the-right) — mock-exam-8-solution.tex:865-872, 893-897 — *Q5a(iv,vi)*. Cheat: "thresh↓ ⇒ sens↑, spec↓."
- **Ridge "splits" vs lasso "picks one" under collinearity** (A) — L2 ball geometry: smooth, optimum has both coords nonzero ≈ half OLS each. L1 diamond: corner-on-axis ⇒ zeros one of the pair. The structural geometry argument — mock-exam-8-solution.tex:678-691 — *Q4b(ii)*. Cheat: "L2 ball → splits; L1 diamond → corner → picks one."
- **PCR vs lasso: structural cost** (C) — PCR (PCA-then-regress) cannot produce a sparse model in *original* predictors — every original var enters every PC score; only lasso gives genuine variable selection — mock-exam-8-solution.tex:704-708 — *Q4b(iv)*. Cheat: "PCR = no original-predictor sparsity."
- **GBM CV-curve goes back UP at large M; RF does NOT** (B) — GBM trees sum (additively cumulative ⇒ overfit possible); RF averages over bootstrap ⇒ "too many trees" harmless — mock-exam-8-solution.tex:738-744 — *Q4c(ii)*. Cheat: "GBM: U-shape in M; RF: flat-then-flat (averaging)."
- **GAM additivity limitation** (C) — GAM = β₀ + Σ f_j(x_j) can't capture interactions natively; depth-d≥2 trees in GBM do. Patch in GAM needs explicit tensor-product term — mock-exam-8-solution.tex:776-783 — *Q4d(iii)*. Cheat: "GAM additive ⇒ no interactions; need s(x,y) tensor product."
- **B-spline df = K + p (cubic, intercept removed)** (A/D) — bs() with K=3 knots, degree 3, no intercept → 3 + 3 = 6 basis functions — mock-exam-8.tex:346, mock-exam-8-solution.tex:756-760 — *Q4d(i)*. Cheat: "cubic bs: df = K + 3 (intercept-out)."
- **AdaBoost derivation: split-by-correct → weighted err** (D) — Inner min reduces to G_m = argmin_G Σ w_i^(m) 1[y_i ≠ G(x_i)] because (e^α − e^{−α}) > 0 multiplies only the misclassified mass — mock-exam-8-solution.tex:904-923 — *Q5b(i)*. The prof flagged AT LEAST one derivation on 2026 exam — this is the canonical one.
- **AdaBoost α_m closed form derivation** (D) — Set dJ/dα=0 → e^{2α}=(1−err)/err → α_m = ½ log((1−err)/err). The factor of 2 differs from slide convention but absorbed into sign(·) — mock-exam-8-solution.tex:926-942 — *Q5b(ii)*. Memorize the derivative-set-to-zero step.
- **Backprop with sigmoid + cross-entropy: ∂L/∂η = p̂ − y** (D) — The p̂(1−p̂) factors cancel cleanly; "canonical output pair." Then ∂L/∂β_k = (p̂−y)h_k; ∂L/∂α_kj = (p̂−y)β_k g'(z_k) x_j — mock-exam-8-solution.tex:1004-1037 — *Q5c(iii)*. Cheat: write the chain ∂L/∂η · ∂η/∂h_k · ∂h_k/∂z_k · ∂z_k/∂α_kj = (p̂−y)·β_k·g'(z_k)·x_j.
- **LDA discriminant: which terms drop and why** (D) — Drop (a) constants free of k, (b) −½log|Σ| BECAUSE Σ shared, (c) −½ x^T Σ⁻¹ x BECAUSE Σ shared (term depends on x but not k). Under QDA all three k-dependent terms survive ⇒ boundary becomes a conic — mock-exam-8-solution.tex:322-342 — *Q3a(i)*. Cheat: "Σ shared kills x^T Σ⁻¹ x ⇒ linear boundary."
- **Paired bootstrap for model comparison** (A) — Joint resampling of test pairs preserves Cov(A,B) (hard pairs inflate both losses); independent resampling destroys cov and inflates Var(Δ̂) = Var(A)+Var(B)−2Cov(A,B) — mock-exam-8-solution.tex:143-149 — *Q2c(iii)*. Cheat: "paired bootstrap → preserve Cov ⇒ tighter SE on diff."
- **Random k-fold CV invalid for time series** (B) — Random partition leaks "next-day" info into training; bias is *downward*. Correct: forward-chaining CV (train [1,...,t], test [t+1,...,t+h]) — mock-exam-8-solution.tex:114-118 — *Q2b(iii)*. Cheat: "time series ⇒ forward-chaining CV, never random k-fold."

## Lower-priority but useful

- **Bias–variance decomposition is an ALGEBRAIC identity (holds in double-descent regime)** — The identity Bias²+Var+σ² is exact; double descent is a non-monotone *shape* of Var(p), not a failure of the identity — mock-exam-8-solution.tex:88-92 — *Q2a(iii)*.
- **MSE-optimal shrinkage c* < 1 only when noise non-negligible** — Whether shrinkage dominates OLS depends on SNR μ²·n/σ²; trap is "shrinkage always helps" — mock-exam-8-solution.tex:84-87 — *Q2a(ii)*.
- **Mini-batch gradient variance scales as 1/m, not m** — Doubling m halves (not doubles) variance — mock-exam-8-solution.tex:447-449 — *Q3b(ii)(B)*.
- **Total variance of standardized data = p** — Σ λ_j = p when variables standardized; check λ-sum first — mock-exam-8-solution.tex:271 — *Q2i(i)*.
- **Naive Bayes ≡ diagonal-Σ LDA** — Conditional independence given class ⇔ diagonal within-class covariance — mock-exam-8-solution.tex:259-262 — *Q2h(ii)*.
- **Bootstrap is WITH replacement** — ~37% of rows are OOB; common confusion with without-replacement — mock-exam-8-solution.tex:136-138 — *Q2c(ii)*.
- **B for bootstrap CIs: 1k–10k typical** — Prof's recommendation; B≈1000 for SE alone — mock-exam-8-solution.tex:139-141 — *Q2c(ii)(C)*.
- **CI for E[y|x₀] vs PI for new y at x₀: PI is wider** — PI adds extra σ² (irreducible noise) on top of the βˆ-uncertainty σ² x₀^T(X^TX)⁻¹x₀ — mock-exam-8-solution.tex:656-660 — *Q4a(vii)*.
- **Heteroscedasticity remedy: log-transform y or WLS** — Fanning-out in resid-vs-fitted ⇒ violates constant-variance assumption — mock-exam-8-solution.tex:650-653 — *Q4a(vi)*.
- **Quadratic-temp inverted-U for bike-share** — Cold → few cyclists; hot → uncomfortable; peak at moderate warmth — mock-exam-8-solution.tex:601-604 — *Q4a(iii)*.
- **"Fancy correlation, not causal" prof mantra** — Logistic β̂'s reflect conditional association on observational data; raising credit score by intervention ≠ predicted effect — mock-exam-8-solution.tex:874-881 — *Q5a(v)*.
- **NN parameter count: don't forget biases** — (p×M + M) + (M×1 + 1); for p=7, M=12: 96 + 13 = 109 — mock-exam-8-solution.tex:984-989 — *Q5c(i)*.
- **"Dying ReLU"** — g'(z) = 1[z>0]; once z_k≤0 the gradient through that unit is exactly 0, can stay dead — mock-exam-8-solution.tex:1038-1040 — *Q5c(ii,iii)*.
- **Ridge does NOT penalize the intercept** — Penalty sum is j=1..p only; the intercept is unconstrained — mock-exam-8-solution.tex:664-668 — *Q4b(i)*.
- **Workday × temperature interaction: amplification on workdays** — β̂_int = +0.40 means temperature drives demand *more* on workdays (commuter decision) than weekends (recreational, less weather-sensitive) — mock-exam-8-solution.tex:629-634 — *Q4a(iv)*.

## Things explicitly in ISLP (skipped)

- LDA Gaussian density formula, derivation of discriminant in single-predictor case (ISLP §4.4)
- OLS X^T X invertibility, hat matrix definition (ISLP §3.2)
- Ridge / lasso objectives, constraint-region pictures (ISLP §6.2, Fig 6.7 — though geometry intuition flagged above is *prof framing*)
- Confusion-matrix metrics: sensitivity/specificity formulas (ISLP §4.4.3)
- Sigmoid identity σ'(η) = σ(η)(1−σ(η)) and logistic linear predictor (ISLP §4.3)
- PCA loadings, scores, PVE definitions (ISLP §12.2)
- Bootstrap mechanics (with replacement, σ̂ from B replicates) (ISLP §5.2)
- k-fold CV mechanics (ISLP §5.1)
- GAM general form (ISLP §7.7)
- Random forest bagging + m<p split sampling (ISLP §8.2)
- Squared-error gradient boosting on residuals (ISLP §8.2.3)
- Backprop generalities (ISLP §10.7)
- R²/Adjusted R² interpretation, residual d.f. (ISLP §3.1.3)
