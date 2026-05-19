# Mock Exam 4 — Cheat-sheet candidates

## Question summary

- **P1 (10pt)**: Fill-in-blank vocab — collinearity, ridge, k-fold CV, nested CV, mini-batch SGD, backprop, dropout, early stopping, label smoothing, gradient boosting.
- **P2 (28pt)**: T/F + short numeric across bias-variance/double descent, CV (wrong-way trap, LOOCV-OLS shortcut), NN param count + forward pass (ReLU, $p=4,M_1=5,M_2=3,C=1$ → 47), backprop seed sign, mini-batch SGD as implicit regularizer, dropout (train/test asymmetry, 20% prof default, NOT 50%), AdaBoost vs GB vs XGBoost (XGB = 2nd-order + L1/L2 on leaf weights), collinearity (perfect vs near; predictions can still be fine; software drops one column), PCA (cum-var, score, scale-sensitivity), hierarchical (first-merge is linkage-independent), K-means multistart.
- **P3 (16pt)**: Bias-variance derivation in 5 parts (cross-term killing, add-subtract trick, "decomposition not trade-off"); nested k-fold pseudocode + winner's-curse downward bias; bootstrap SE pseudocode, $1-(1-1/n)^n$ at $n=6$ → 0.665 and $\to 1-1/e\approx 0.632$, percentile-CI.
- **P4 (22pt)**: Apartment-rent OLS (collinearity in area/rooms/area_per_room, polynomial age-minimum at 88.9y, SD-scale interaction with balcony, $R^2$ vs adj $R^2$, F-test on 3-level factor); lasso path with $\lambda_{\min}$ vs $\lambda_{1\text{SE}}$; gradient-boost pseudocode + variable-importance/PDP; GAM df counting ($K+3$ for cubic spline with $K=3$ knots → 6 df, total 20).
- **P5 (24pt)**: Logistic with size:age interaction (odds factor depends on age — main effect alone is age-0 extrapolation), NN param count (6→32→16→1 = 769), regularization menu + "iron rule" of NN training, AdaBoost weight-update, $\text{err}=0.5 \Rightarrow \alpha=0$, AdaBoost-as-gradient-boost-under-exponential-loss, class-imbalance / threshold tuning (lower threshold → sensitivity↑ specificity↓, NW→NE on ROC).

## High-priority surfacing

- **NN parameter-count formula** (A, D)
  - $\text{params} = (p+1)M_1 + \sum_{\ell\ge 2}(M_{\ell-1}+1)M_\ell + (M_L+1)C$. The $+1$ sits on the *receiving* layer (one bias per receiving unit). Forgetting biases is the prof's canonical distractor.
  - Tested twice: Q2c.i (47), Q5b.i (769). Past exams 2023/2024/2025 used "no-bias" as distractor.
  - Source: wiki/book-deltas/m11-nnet.md:220-259 (NOT in ISLP, only implicit via MNIST diagram).
  - Cheat-sheet: `params = Σ_ℓ (in_ℓ + 1)·out_ℓ`. Quick check: $4{\to}5{\to}3{\to}1 = 25+18+4 = 47$.

- **Backprop output-seed sign** (B, D)
  - $\delta^{\text{out}}_i = \partial R_i / \partial f(x_i) = -(y_i - \hat f(x_i))$ — the **negative** residual. Sign trap from $\partial_f \tfrac12(y-f)^2 = -(y-f)$.
  - Q2d.iv flags exactly this trap.
  - Source: wiki/book-deltas/m11-nnet.md:60-66 ("Watch this on the exam"); ISLP §10.7.1 doesn't introduce $\delta$ notation at all.
  - Cheat-sheet: `δ^out = -(y - ŷ)`. Multi-class with softmax+CE: $\delta^{\text{out}}_c = f_c - y_c$.

- **Mini-batch SGD = implicit L2 / min-norm interpolator** (A, B)
  - Headline NN fact. Unbiased estimator $\mathbb{E}[\widehat{\nabla L}_{\mathcal B}] = \nabla L$; larger batch → lower variance → **less** implicit regularization. ISLP §10.7.2 only footnotes this as research; prof says "proven."
  - Q2d.iii, Q5b.ii.
  - Source: wiki/book-deltas/m11-nnet.md:312-318.
  - Cheat-sheet: bigger batch → smoother gradient → weaker implicit regularizer.

- **Dropout: prof's 20% rule, NEVER 50% on tabular** (A, C)
  - "Drop-out rates may be chosen between 0.2 and 0.5. … 20% is very common, never use 50%." Active only at training; test time uses full network with appropriate scaling (inverted dropout).
  - Q2e.i, Q2e.ii, Q5b.iv directly use this prescription.
  - Source: wiki/book-deltas/m11-nnet.md:358-367 (rate prescription NOT in ISLP).
  - Cheat-sheet: `p_drop ∈ [0.2, 0.5]; 20% default tabular, 50% under-fits signal-poor data; off at test`.

- **Label smoothing — NOT in ISLP** (A)
  - Replace one-hot $\mathbf y$ with $\tilde y_c = 1-\varepsilon$ on true class, $\varepsilon/(C-1)$ elsewhere. Motivation per slides: "training data may contain errors in the responses recorded."
  - Q1(9), Q2e.iv, Q5b.iii.
  - Source: wiki/book-deltas/m11-nnet.md:344-351.
  - Cheat-sheet: `ŷ_c = 1-ε on true class, ε/(C-1) else`; motivated by label noise.

- **Early stopping monitors VALIDATION (not training) loss** (B)
  - Common distractor: "stop when training loss plateaus." Wrong — defeats the purpose. Return params from the epoch with min val loss.
  - Q2e.iii.
  - Source: wiki/book-deltas/m11-nnet.md:352-357.

- **Wrong-way CV (filter-on-full-data trap)** (A, B)
  - Pre-selecting predictors with $y$ on the full data leaks labels into every fold → CV underestimates. Fix: selection must be **inside** each fold (or use nested CV). $n=50, p=5000$ noise gives CV ≈ 0% wrong vs 50% correct. ESL §7.10, not ISLP Ch.5.
  - Q1(4), Q2b.ii, Q2b.iii, Q3b.ii.
  - Source: wiki/book-deltas/m05-resample.md:121-163.
  - Cheat-sheet: anything that uses $y$ (correlation filter, supervised PCA) must live inside CV loop.

- **Nested CV: inner selects, outer assesses** (A, C)
  - Single-CV-then-report-the-min is a winner's-curse (downward-biased). Outer fold gives honest assessment of the selection-inclusive procedure, not of any single $\lambda$.
  - Q3b.i, Q3b.ii.
  - Source: wiki/book-deltas/m05-resample.md:93-119.
  - Cheat-sheet: pick λ via inner k-fold per outer fold; aggregate outer scores. Same-CV-minimum-as-both is biased ↓.

- **LOOCV OLS PRESS shortcut** (A, D)
  - $\text{LOOCV} = \tfrac1n \sum_i \bigl(\tfrac{y_i - \hat y_i}{1-h_{ii}}\bigr)^2$ — one fit + hat-matrix diagonal. ISLP §5.2 eq. 5.2.
  - Q2b.iv (T/F: "can be computed without refitting $n$ times").
  - Source: ISLP eq 5.2 (in ISLP); flagged here because the prof verbatim-tested it.
  - Cheat-sheet: `LOOCV_OLS = (1/n) Σ ((y_i−ŷ_i)/(1−h_ii))²`.

- **Collinearity: perfect ≠ inflated SE** (B)
  - Q2g.iii nails the common mistake: `age + age/10` makes $X^\top X$ **exactly singular**; OLS estimates are **not defined**, NOT "well-defined with infinite SE." Software silently drops a column.
  - Q2g, Q4a.ii-iii.
  - Source: wiki/concepts/collinearity.md:21-23, 89-92.
  - Cheat-sheet: near-collinearity → SEs inflate; perfect → no solution exists, software drops a column.

- **Collinearity: predictions OK, coefficients NOT** (B)
  - Q2g.ii false: collinearity inflates *coefficient* variance; predictions $\hat y = X\hat\beta$ can still be accurate (sum is well-determined, individual splits are not).
  - Source: wiki/concepts/collinearity.md:89-91.

- **Quadratic-term turning point** (D)
  - For $\hat y = \cdots + \beta_1 x + \beta_2 x^2$, partial effect minimized at $x^* = -\beta_1/(2\beta_2)$ (min if $\beta_2 > 0$).
  - Q4a.iv: $-(-0.080)/(2\cdot 0.00045) = 88.9$ years.
  - Cheat-sheet: `x* = -β_age/(2 β_age²); min if β_age² > 0`.

- **Interaction direction-of-effect on standardized predictor** (B)
  - With `distance:has_balcony`: 1-SD distance effect is $\beta_{\text{dist}}$ (no balcony) vs $\beta_{\text{dist}} + \beta_{\text{dist:bal}}$ (with balcony). Both negative here ($-1.95$ vs $-1.40$) — balcony **attenuates** the rent loss.
  - Q4a.v, also Q5a.i (logistic odds factor $= \exp(\beta_{\text{size}} + \beta_{\text{size:age}}\cdot \text{age})$ depends on age).
  - Cheat-sheet: with interaction `x:z`, partial effect of $x$ = `β_x + β_{xz}·z`. Main-effect alone summarizes only when $z=0$.

- **1-SE rule** (A, C)
  - Pick the **simplest** $\lambda$ whose CV-MSE is within one SE of the minimum CV-MSE. Slide-only / not in ISLP Ch.5 (§6.1.3 mentions briefly).
  - Q4b.iv.
  - Source: wiki/book-deltas/m05-resample.md:33-51.
  - Cheat-sheet: `λ* = max{λ : CV(λ) ≤ CV(λ̂) + SE(CV(λ̂))}` (simplest within 1-SE).

- **Bootstrap inclusion probability** (D)
  - $P(X_1 \in \text{boot}) = 1 - (1-1/n)^n$. At $n=6$: 0.665. As $n\to\infty$: $1 - 1/e \approx 0.632$.
  - Q3c.ii.
  - Source: ISLP exercise 5.2; OOB derivation wiki/book-deltas/m05-resample.md:255-272.
  - Cheat-sheet: `P(in) = 1 − (1−1/n)^n → 1 − 1/e ≈ 0.632`. (OOB fraction ≈ 0.368.)

- **Lasso objective: intercept NOT penalized** (D)
  - $\hat\beta^{\text{lasso}} = \arg\min \tfrac1{2n}\|y - \beta_0 - X\beta\|^2 + \lambda\|\beta\|_1$. Intercept excluded so solution doesn't depend on units of $y$.
  - Q4b.i.
  - Cheat-sheet: `(1/2n)·RSS + λ·Σ|β_j|`; intercept and biases unpenalized.

- **AdaBoost $\alpha_m = 0$ at err=0.5** (B, D)
  - $\alpha_m = \log((1-\text{err}_m)/\text{err}_m)$. At err=0.5 → $\log 1 = 0$, classifier contributes nothing. err > 0.5 → $\alpha < 0$, vote is auto-flipped.
  - Q5c.ii.
  - Source: wiki/book-deltas/m09-boosting.md:78-86.
  - Cheat-sheet: `α_m = log((1−err_m)/err_m)`; err=0.5⇒α=0; err>0.5⇒α<0 (auto-flip).

- **AdaBoost = gradient boost under exponential loss** (A)
  - $L(y, f) = \exp(-yf)$ with $y\in\{-1,+1\}$. Equivalence not in ISLP §8.2.3.
  - Q5c.iv.
  - Source: wiki/book-deltas/m09-boosting.md:100-120.

- **Gradient boosting: residual = negative gradient under MSE** (D)
  - $L = \tfrac12(y-f)^2 \Rightarrow -\partial L/\partial f = y - f$ = residual. Fitting next tree to residuals ≡ gradient descent in function space.
  - Q2f.ii, Q4c.ii.
  - Cheat-sheet: `r_i = y_i − F_{m-1}(x_i) = -∂L/∂f`; update `F_m = F_{m-1} + ν·T_m`.

- **GB shrinkage $\nu$ ↔ $M$ coupling** (B)
  - Smaller $\nu$ ⇒ each tree contributes **less** ⇒ **more** trees needed. Q2f.iii distractor reverses this.
  - Cheat-sheet: halving $\nu$ roughly doubles required $M$.

- **XGBoost = 2nd-order + L1/L2 on leaf weights** (A)
  - Distinguishing features beyond vanilla GB: (a) second-order Taylor expansion (gradient + Hessian), (b) explicit $L_1/L_2$ penalty on leaf weights.
  - Q2f.iv.
  - Source: wiki/book-deltas/m09-boosting.md (XGBoost section).

- **PCA needs standardization on differently-scaled variables** (B)
  - PCA operates on the covariance matrix → scale-sensitive. Without standardization, large-scale variables dominate PC1. Q2h.iii false claim that PCA is "invariant under rescaling."
  - Cheat-sheet: standardize before PCA when scales differ.

- **PVE / cum-var calculation with standardized vars** (D)
  - For standardized vars, $\sum_j \lambda_j = p$, so $\text{PVE}_j = \lambda_j / p$. Cumulative until ≥ 0.80.
  - Q2h.i: $(2.5+1.5+1.1+0.8)/7 = 0.843 \Rightarrow 4$ PCs.
  - Cheat-sheet: `PVE_j = λ_j / Σ λ_k = λ_j / p` (standardized).

- **First merge in hierarchical: linkage doesn't matter** (B)
  - All three linkages reduce to smallest pairwise dissimilarity when every cluster is a singleton; linkage rule only kicks in from the 2nd merge.
  - Q2i.i.

- **Cubic regression spline df: $K + 4$ incl intercept, $K + 3$ excl** (A, D)
  - `bs(x, knots={c1,...,cK})` cubic spline uses $K+3$ basis functions when global intercept supplied separately, $K+4$ when not.
  - Q4d.i: $K=3 \Rightarrow K+3 = 6$ df. Q4d.ii total 20 df.
  - Source: wiki/book-deltas/m07-beyondlinear.md:101-103, 318-325.
  - Cheat-sheet: `cubic spline df = K + 4 (incl intercept) = K + 3 (excl)`; natural cubic = $K+2$ / $K+1$.

- **GAM structural limitation: no interactions** (C)
  - Additive sum of 1-D smooths → cannot represent predictor-pair interactions unless explicitly added (e.g. tensor product). GB trees of depth $d>1$ get interactions for free.
  - Q4d.iii.

- **Threshold tuning: direction on ROC** (B)
  - Lower threshold (e.g. 0.5 → 0.3) ⇒ more positive predictions ⇒ **sensitivity ↑, specificity ↓**, point moves **up-and-right** on the ROC (higher TPR, higher FPR).
  - Q5d.iii.
  - Source: wiki/concepts/sensitivity-specificity.md:64-71.
  - Cheat-sheet: lower threshold → more flags → TPR↑, FPR↑ (NE on ROC).

- **Screening tools weigh sensitivity more (medical context)** (B, C)
  - For dermatology screening: missed melanoma = potentially fatal; false positive = follow-up biopsy. Tune threshold down to push sensitivity up.
  - Q5a.v, Q5c.iii.

- **Class-imbalance: naive accuracy** (B)
  - $20\%$ malignant ⇒ "always benign" classifier has accuracy 0.80, **sensitivity 0**, specificity 1. Headline accuracy is misleading without sens/spec.
  - Q5d.i-ii.

## Lower-priority but useful

- **Bias-variance "decomposition not trade-off"** — prof's preferred terminology because regularizers (ridge/lasso/dropout) can reduce variance without raising bias; double descent can lower both. Q2a, Q3a.iv. Wiki: concepts/bias-variance-tradeoff.md.
- **F-test for joint significance of categorical** — $H_0:\beta_{\text{type\_studio}}=\beta_{\text{type\_loft}}=0$. Q4a.vii.
- **Adding nested term: training $R^2$ monotone ↑, adjusted $R^2$ can fall** — Q4a.vi.
- **Variance-of-sum identity** — explains why LOOCV variance ≠ $\sigma^2/n$ (correlated folds). Prof slide formula. Wiki: book-deltas/m05-resample.md:54-69.
- **Bootstrap percentile CI** — $[\hat\theta^*_{(2.5\%)}, \hat\theta^*_{(97.5\%)}]$ from $B$ resamples. Q3c.iii.
- **Lasso geometry: $\ell_1$ ball has corners on axes** — produces exact zeros vs ridge's smooth ball. Q4b.ii.
- **Variable-importance + PDP** — interpretability recovery from tree ensembles. Q4c.iii.

## Things explicitly in ISLP (skipped)

- Lasso/ridge basics — ISLP §6.2 covers fully.
- ReLU/sigmoid/softmax forward-pass mechanics — ISLP §10.1 (computed: $z = -3.5$, ReLU $\Rightarrow 0$ in Q2c.ii).
- Linear predictor sum + sigmoid step for logistic regression — ISLP §4.3 (Q5a.iii).
- K-means multistart recommendation — ISLP §12.4.1 (Q2i.ii).
- Standard bootstrap pseudocode — ISLP §5.2 eq. 5.8 (Q3c.i).
- OLS coefficient interpretation, residual-df check — ISLP §3.
- Confusion-matrix construction, sensitivity/specificity definitions — ISLP §4.4.2.
- Polynomial regression mechanics — ISLP §7.1.
