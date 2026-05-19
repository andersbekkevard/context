# Mock-exam + past-exam patterns — cheat-sheet candidates

Synthesis of recurring patterns across mock-exams 1–10, real exams 2023/24/25, and the Apr 28 scope review. Tiered for a 2-page A4 cheat sheet, focusing on what RESCUES you under time pressure (ISLP is open-book — don't burn lines on what's already there).

---

## MUST (recurring every exam, high-rescue value)

### Output-table reading (regression / logistic)

- **Param count = rows of coefficient table.** Verify against printed residual d.f. = `n_train − p`. Always do this sanity check first; it catches mis-reads of categorical dummies.
- **Dummy coding.** $K$-level categorical → $K-1$ dummy columns. Reference level = the one missing from the table. Adding $K$ dummies + intercept ⇒ rank-deficient design (dummy-variable trap). Ridge can rescue identifiability; OLS cannot.
- **CI from output.** $\hat\beta \pm 1.96\cdot \mathrm{SE}$. Reject $H_0$ at 5% iff CI excludes 0 iff $|t|>1.96$.
- **Unit gotcha.** If the variable is in 1000s of EUR/USD/kg, a "1-unit" coefficient = effect per 1000. State units before computing $\Delta$.
- **Standardized predictors.** Effect of "1-SD increase" is just $\hat\beta$ (since SD=1 after standardization). Mention this when interpreting standardized output.

### Interaction interpretation (THE most common trap)

> Recipe for `x:group`:
> - At reference group: effect of $x$ = $\hat\beta_x$
> - At non-reference group: effect of $x$ = $\hat\beta_x + \hat\beta_{x:\text{group}}$
> - Effect of group depends on $x$: $\hat\beta_{\text{group}} + \hat\beta_{x:\text{group}}\cdot x$

**Trap.** A non-significant main effect (e.g. $p_\text{group}=0.18$) does NOT mean drop the variable when the interaction is in the model — the main effect is just "group effect at $x=0$" (often outside data range, especially after standardization). Also: never drop a main effect while keeping its interaction (hierarchical principle).

### Odds ↔ probability (every single exam)

- $p = \frac{\text{odds}}{1+\text{odds}}$, $\text{odds} = \frac{p}{1-p}$
- Logistic: $\hat\eta = \beta_0 + \sum \beta_j x_j$; $\hat p = \sigma(\hat\eta) = \frac{1}{1+e^{-\hat\eta}}$
- **Odds factor for $\Delta x_j$ = 1: $e^{\hat\beta_j}$.** For $\Delta x_j = k$: $e^{k\hat\beta_j}$. For interaction `x:D=1`: factor = $e^{\hat\beta_x + \hat\beta_{x:D}}$.
- **Trap.** $\hat\beta_j$ is NOT the change in probability — it's change in log-odds. "Probability change ≈ $\hat\beta$" is FALSE.
- **Reference relabel.** Flipping the reference flips the sign of the dummy coefficient; predicted probabilities are unchanged.

### Confusion-matrix metrics (every exam, often twice)

Layout convention: rows = actual, cols = predicted (TP top-left for the positive class).

- **Sensitivity = TPR = TP/(TP+FN)** — share of true positives caught.
- **Specificity = TN/(TN+FP)** — share of true negatives correctly cleared.
- **Error rate = (FP+FN)/n**; **Accuracy = 1 − error**.
- **Precision = TP/(TP+FP)** (positive predictive value).
- Always state the metric in problem-specific words ("fraction of true defaulters caught", "fraction of healthy patients correctly cleared").
- **Baseline trap (class imbalance).** Always-majority classifier gets error = minority rate (e.g. 20%, 5%). Test error alone is misleading; report sensitivity+specificity or AUC.

### ROC / AUC

- Axes: TPR (y) vs FPR=1−specificity (x). Threshold sweep.
- AUC = P(rank random positive above random negative). AUC=0.5 ⇔ random; 1 ⇔ perfect.
- **Lowering threshold (e.g. 0.5 → 0.3): sensitivity↑, specificity↓; operating point moves up-and-right.**
- Same AUC ≠ same ROC shape (false equivalence trap).

### Bias-variance decomposition (asked in nearly every exam, often the "mathy" question)

$$\mathbb{E}[(y_0-\hat f(x_0))^2] = \underbrace{(f(x_0)-\mathbb{E}[\hat f(x_0)])^2}_{\text{Bias}^2} + \underbrace{\mathrm{Var}(\hat f(x_0))}_{\text{variance}} + \underbrace{\sigma^2}_{\text{irreducible}}$$

- Bias is **squared**.
- Expectation over **training set draws AND test noise $\varepsilon_0$**; inner $\mathrm{Var}/\mathbb{E}[\hat f]$ over training draws only.
- $\sigma^2$ is irreducible — more data does NOT shrink it.
- **It's an exact identity** (algebraic), not a constraint. "Decomposition" preferred over "trade-off" because regularization / over-parameterized regime / double descent can reduce both bias² and variance simultaneously.
- Derivation cross-terms vanish because (a) $\varepsilon_0 \perp \hat f$ and $\mathbb{E}\varepsilon_0=0$; (b) inner cross-term is $2(f-\mathbb{E}\hat f)(\mathbb{E}\hat f - \mathbb{E}\hat f)=0$.
- **Direction rule.** Flexibility↑ ⇒ bias²↓, variance↑. High-noise $\sigma^2$ ⇒ prefer LESS flexible.

### Hierarchical clustering by hand

1. Find min off-diagonal entry of $D$ → merge those two; height = that entry.
2. Recompute distances from new cluster to remaining points:
   - **Single linkage**: min over pairs.
   - **Complete linkage**: max over pairs.
   - **Average linkage**: mean.
3. Repeat until one cluster.

**Trap.** First merge is identical under all linkages (only one pair to compare). Linkage only matters from merge #2 onward.

Cut at height $h$ → number of clusters = lines crossed in the dendrogram.

### CV recipes & pseudocode

**$k$-fold CV pseudocode (always testable):**
```
Partition n obs into K equal folds F_1,...,F_K (random)
For each hyperparam λ ∈ Λ:
    For k=1..K: fit on ∪_{j≠k}F_j → ĥf_λ^(-k); MSE_k(λ)=avg loss on F_k
    CV(λ) = mean_k MSE_k(λ); SE(λ) = sd_k(MSE_k)/√K
Pick λ* via min OR 1-SE rule (see below)
Refit on FULL training set at λ*
```

**One-SE rule.** $\hat\lambda_{1\text{SE}} = \max\{\lambda : \mathrm{CV}(\lambda) \le \mathrm{CV}(\hat\lambda_{\min}) + \mathrm{SE}(\hat\lambda_{\min})\}$. Picks the SIMPLEST model statistically indistinguishable from the best. Preferred for parsimony / stability / "CV minimum is itself noisy" / overfitting-the-folds.

**LOOCV.** Higher variance than 5/10-fold (training sets overlap a lot, errors highly correlated). Lower bias. OLS shortcut: $\mathrm{CV}_\text{LOO}=\frac{1}{n}\sum\left(\frac{y_i-\hat y_i}{1-h_{ii}}\right)^2$ where $h_{ii}$ = hat-matrix diagonal.

**Wrong-way CV (high-impact trap).** Filtering predictors by correlation with $y$ on FULL data, THEN running CV inside, leaks the response → CV biased DOWNWARD. Fix: do feature selection INSIDE each CV fold (or use nested CV).

**Nested CV.** Outer folds estimate generalization; INNER folds tune λ. Outer hyperparam can differ per outer fold — that's correct, you're estimating the procedure not a single model.

**Time-series.** Naive random k-fold is BIASED when observations are temporally autocorrelated.

### Ridge vs Lasso (recurs across exams)

- Ridge: $\min \|y-X\beta\|^2 + \lambda\sum\beta_j^2$; **intercept NOT penalized**; shrinks toward 0, never exactly 0.
- Lasso: $\lambda\sum|\beta_j|$; **sparse** solutions because $\ell_1$ ball has corners on axes.
- **Standardize predictors before either** (penalty is scale-dependent).
- $\lambda \to 0$: → OLS; $\lambda \to \infty$: → intercept-only.
- Ridge fixes singular $X^\top X$ (perfect collinearity, rank-deficient design): $X^\top X + \lambda I$ is always PD.
- Direction: increasing $\lambda$ ⇒ bias↑, variance↓.
- **Comparison trap.** If shrinkage doesn't beat OLS on test MSE, conclusion = "most predictors carry signal; bias added by penalty exceeds variance saved."

### Bootstrap essentials

- Sample WITH replacement, same size $n$.
- $P(i \notin \text{boot sample}) = (1-1/n)^n \to 1/e \approx 0.368$, so $P(i \in) \to 1-1/e \approx 0.632$.
- **OOB error in RF/bagging is "free CV"** — each obs is OOB for ~1/3 of trees.
- Bootstrap SE: $\widehat{\mathrm{SE}} = \mathrm{sd}(\hat\theta^*_1,\dots,\hat\theta^*_B)$.
- **Percentile 95% CI**: $[q^*_{0.025},\, q^*_{0.975}]$ of the bootstrap dist.
- **Trap.** Bootstrap does NOT auto-correct bias of $\hat\theta$. Typical $B \in [1000, 10000]$.

### PCA hand calculation (recurs)

- Standardized $\Rightarrow$ total variance = $p$ (number of variables). Sanity-check: $\sum \lambda_j$ should equal $p$.
- $\mathrm{PVE}_j = \lambda_j / \sum_\ell \lambda_\ell$. Cumulative PVE for "≥ X%" decisions.
- Score on PC$j$: $z^*_j = \phi_j^\top x^*$. Unit-norm constraint: $\|\phi_j\|=1$.
- "Most contributing variable to PC1" = max $|\phi_{1k}|$.
- **Trap.** PCA is NOT scale-invariant — failing to standardize lets the largest-variance variable dominate. (False to say "loadings unchanged without standardization.")

### Spline degrees of freedom (recurs every exam)

- **Cubic regression spline, $K$ interior knots**: $K+4$ basis functions (intercept counted as 1) ⇒ $K+3$ excluding global intercept.
- **Natural cubic spline, $K$ knots**: $K$ basis functions (intercept absorbed), so $K+1$ if you count an intercept; fewer than plain cubic because linearity-outside-boundary constraints subtract 4. **Cheat note: ISLP-edition default is "$K$" total for natural with intercept absorbed; verify per question.**
- **Cubic spline continuity**: continuous in value, 1st, 2nd derivative; 3rd derivative jumps.
- Smoothing spline: $\mathrm{df}_\lambda = \mathrm{tr}(S_\lambda)$; non-integer; $\lambda\to 0$ ⇒ interpolating, $\lambda\to\infty$ ⇒ straight line; $\lambda\uparrow$ ⇒ df↓ (smoother fit).
- **GAM total d.f.** = intercept + Σ (spline term basis count) + Σ (linear term: 1) + Σ (categorical: $K-1$ dummies). Worked example: intercept (1) + 3 cubic-spline terms ($K+3$ each) + 2 linear (1+1) + 3-level categorical (2) = formulaic sum.

### Neural-network parameter counting

Layer with $a$ inputs → $b$ neurons (with bias): $(a+1)\cdot b$ params. Sum across layers including output.

**Forward pass.** Pre-activation $z = b + \sum w_j x_j$; output = activation$(z)$. ReLU = $\max(0,z)$. Sigmoid = $1/(1+e^{-z})$.

**Why nonlinear activations.** Without them, any stack of layers collapses to a single affine map. Convexity is LOST in multi-layer nets (nonlinearity is not a regularizer — that's a separate mechanism).

### Tree ensembles

- **Bagging variance formula** (mathy candidate): if $B$ predictions have variance $\sigma^2$ and pairwise correlation $\rho$, then
$$\mathrm{Var}(\bar f_\text{bag}) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2 \xrightarrow{B\to\infty} \rho\sigma^2$$
The floor $\rho\sigma^2$ is what motivates **random forests**: at each split sample $m<p$ predictors to decorrelate trees, lowering $\rho$.
- **RF defaults**: $m=\lfloor\sqrt p\rfloor$ (classification), $m\approx p/3$ (regression). $B \approx 500$ standard. RF does NOT overfit in $B$ (just plateaus).
- Trees grown deep, unpruned in RF — variance handled by averaging.
- **Boosting differs**: $B$ IS a tuning parameter; can overfit if $B$ too large. Halving learning rate $\nu$ ≈ doubles required $B$. $(B, \nu)$ tuned jointly. Small $\nu \in [0.001, 0.1]$.
- **Gradient boosting = residual fitting** for squared loss; equivalently fits negative gradient. **Interaction depth $d$**: $d=1$ stump = additive; larger $d$ = higher-order interactions (more variance).

### LDA / QDA (asked in most exams, often the mathy one)

**LDA discriminant** (shared $\Sigma$):
$$\delta_k(x) = x^\top \Sigma^{-1}\mu_k - \tfrac{1}{2}\mu_k^\top\Sigma^{-1}\mu_k + \log\pi_k$$
Linear in $x$ because the quadratic term $-\tfrac{1}{2}x^\top\Sigma^{-1}x$ is class-independent (cancels in $\delta_k - \delta_\ell$).

**1D LDA boundary** ($\delta_0=\delta_1$): $x_\text{cut} = \frac{\mu_0+\mu_1}{2} - \frac{\sigma^2}{\mu_1-\mu_0}\log\frac{\pi_1}{\pi_0}$.

**Prior shift trap.** Increasing $\pi_k$ moves the boundary AWAY from class $k$ (more area classified as $k$). Direction is intuitive once you stare at the $+\log\pi_k$ term.

**QDA**: relax shared-$\Sigma$ → $x^\top\Sigma_k^{-1}x$ term survives ⇒ quadratic boundary. Many more parameters ⇒ high-variance; LDA preferred for small $n$ per class.

Both LDA & QDA are **generative**: model $P(X|Y=k)$ + $\pi_k$, then Bayes' rule. Logistic regression is **discriminative**.

### Method-comparison framing (closing question in nearly every exam)

Template answer:
1. **Quote both models' relevant test metrics** (MSE / sens / spec / err).
2. **Direction of difference** (which one wins on which metric).
3. **Tie to bias-variance / interpretability / domain cost** (FP vs FN cost).
4. **Pick one and name the criterion**.

Example: "Logistic and RF give similar sens (0.45 vs 0.42) and spec; logistic wins on interpretability via odds ratios and CIs — criterion: equal accuracy, prefer interpretable."

---

## NICE (appeared multiple times; covers a corner)

### AdaBoost mechanics (recurs in mocks 5/6/7/8/10)

```
Init w_i = 1/N
For m = 1..M:
    Fit G_m on weighted data; err_m = Σ w_i 1[y_i ≠ G_m(x_i)] / Σ w_i
    α_m = log((1-err_m)/err_m)            # negative if err > 0.5, flips vote
    w_i ← w_i · exp(α_m · 1[y_i ≠ G_m(x_i)]); renormalize
Output: G(x) = sign(Σ α_m G_m(x))
```
Misclassified obs get up-weighted next round — that's the whole point.

### Backprop in a tiny net

- Sigmoid output + binary cross-entropy: $\partial L/\partial z_\text{out} = \hat p - y$ (residual). Why sigmoid+CE is paired (avoids vanishing-gradient issues).
- ReLU derivative = $\mathbb{1}[z>0]$.
- Chain rule template for one hidden unit: $\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial \hat y}\cdot \frac{\partial \hat y}{\partial h}\cdot \frac{\partial h}{\partial z_1}\cdot \frac{\partial z_1}{\partial w_1}$.

### Mini-batch SGD facts

- Mini-batch gradient is **unbiased**; variance scales as $1/m$.
- Smaller $m$ ⇒ stronger implicit regularization.
- Typical $\eta$ on the order of $10^{-3}$ to $10^{-1}$ (NOT 2).

### NN regularization

- **Dropout**: train-only random masking, ~20% typical rate; rescale at test (or during train).
- **Early stopping**: stop when VALIDATION loss stops improving (not training loss — trap).
- **Weight decay** = $L_2$ penalty.
- **Label smoothing** ($\varepsilon$, $C$ classes): one-hot $(0,\dots,1,\dots,0)$ → true class gets $1-\varepsilon$, others get $\varepsilon/(C-1)$ each. **Or**: $1-\varepsilon$ on true, $\varepsilon/C$ uniform — both conventions appear; show your formula.

### Collinearity diagnostics

Symptoms in the regression table:
- Large coefficient ESTIMATES that flip sign across model perturbations.
- Inflated SEs (e.g. 5× larger than other rows).
- Individually insignificant $p$-values despite large estimates.

**Joint test (F-test) for categorical with multiple levels**: $F$-stat on $(K-1, n-p)$ d.f. tests $H_0$: all dummies = 0 vs reference. Use when individual dummies are non-significant but joint pattern may be.

### Collinearity fix toolkit
- Drop one of the redundant predictors.
- Ridge (keeps both, splits coefficient).
- PCR ("compression" vs ridge "shrinkage"; PCR is discrete projection onto top-$M$ PCs, ridge is continuous).

### K-means
- Iterates assign-then-update; objective monotonically non-increasing; converges to **local** min — run from multiple inits.
- $K$ is NOT chosen by the algorithm.
- Going from $K=4$ → $K=5$ does NOT just split one cluster — global re-assignment.
- Standardize before running (else largest-scale variable dominates).

### KNN
- Distance-based; predict majority class of $K$ nearest.
- **Must standardize** (especially when units differ: balance in thousands vs pay_0 small integers).
- Tuning $K$: small $K$ = low bias, high variance; large $K$ = smoother.

### Subset selection counts
- Best subset on $p$ predictors: $2^p$ models.
- Forward stepwise: $1 + \sum_{k=0}^{p-1}(p-k) = 1 + p(p+1)/2$ models.
- Backward stepwise: **cannot start when $p > n$** (full OLS undefined).

### Bayes' rule for generative classifiers
$P(Y=k|X=x) = \frac{\pi_k f_k(x)}{\sum_\ell \pi_\ell f_\ell(x)}$. Naive Bayes = LDA with diagonal $\Sigma$ (conditional independence).

### Variable importance plot
- Shows: mean decrease in node impurity OR permutation OOB increase per predictor.
- Cannot give: **direction** or **magnitude** of effect → for that, use partial dependence plots or a parametric model.

---

## CUT (skip — open-book ISLP covers it, or it's out of scope)

- AIC / BIC / $C_p$ algebra — out of scope per prof.
- F-test mechanics (computation) — out of scope.
- Bayesian interp of Ridge/Lasso (Gaussian/Laplace priors) — out of scope.
- Detailed boosting line-by-line pseudocode memorization — concept only.
- Multiple-testing corrections (Bonferroni / FDR), survival analysis, SVM, multi-class logistic, probit — all OUT.
- VIF, Shapiro-Wilk — out.
- CNN/RNN architecture details (LSTM gates, BPTT, ResNet, transformers) — high-level only.
- Optimizers beyond plain SGD (momentum, Adam internals) — out.
- Universal approximation proof — out.
- R/Python function names, package names, syntax — explicitly excluded for 2026.
- Pseudoinverse details, spectral decomposition — out.
- Bezier history, K-means++, Ward linkage formula, gap statistic — out.
- Boosting pseudocode (line-by-line) — concept only.
- Spectral covariance decomposition — out.
- ISLP-only formulas already at your fingertips in the open book — don't waste cheat-sheet space; index ISLP page numbers if anything.

---

## Cheat-sheet allocation suggestion

**Highest-rescue lines to commit to A5 paper** (the open book has everything else):
1. Confusion-matrix + odds/probability conversions (compact box).
2. Bias-variance formula + cross-term-vanishes note.
3. CV pseudocode (15 lines) + 1-SE rule wording.
4. Bagging variance formula + RF $m$ defaults.
5. LDA discriminant + boundary, prior-shift direction.
6. Spline / GAM d.f. counting recipe.
7. NN parameter count formula, ReLU/sigmoid.
8. Interaction-with-categorical recipe (the trap).
9. AdaBoost weights/α update (3 lines).
10. PCA: standardized ⇒ total var = $p$; score = $\phi^\top x$.
11. Dummy coding / collinearity symptom checklist.
12. KNN must-standardize note.
13. Bootstrap percentile CI + 0.632/0.368 numbers.
14. ISLP page-pointer mini-index for: ridge/lasso §6.2, GAM §7.7, trees ch.8, NN ch.10, LDA §4.4, PCA §12.2.
