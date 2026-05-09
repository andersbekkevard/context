# Concepts manifest

Deterministic list of concept atoms for the TMA4268 wiki. Source of truth for the concepts fan-out pass, every downstream atom-writing agent reads its module slice from here. Slugs, ownership, and cross-cutting flags are fixed at this layer; do not invent new atoms or rename slugs in the fan-out pass.

Built by reading all 27 lecture pages, the 12 slide decks, the 10 recommended-exercise sheets (Exercise2 through Exercise11), the two compulsory exercises, `exam_analysis.md` (cross-check, not canonical), and `docs/lectures-manifest.md`. The `notes/` folder, `book/`, and `exams/` were not read (off-limits or out of scope at this stage per the brief).

Operating principle: **the prof's scope rule**, slides + lectures + exercises = in scope; book-only = out. Exercise coverage carries the highest exam-relevance weight ("especially the exercises"). When in doubt about granularity, the agent leaned toward **fewer, broader atoms**, with sub-concepts (e.g. studentized residuals, hat matrix's role in leverage, single-vs-complete linkage choice) living as sections inside a parent atom rather than as standalone files.

## Atoms by owning module

### Module 01: Introduction (`01-intro`)

```yaml
- slug: statistical-learning
  canonical: Statistical learning
  lectures: [L01, L02]
  isl-ref: null
  exercises: []
  one-liner: prof's framing of the field — "statisticians getting in on machine learning," misspecified-model regime; the supervised vs unsupervised + prediction vs inference orthogonal axes
- slug: prediction-vs-inference
  canonical: Prediction vs inference
  lectures: [L01, L02, L03]
  isl-ref: null
  exercises:
    - Exercise2.1 — describe a real classification and a real regression task, identify response/predictors, decide prediction or inference
  one-liner: same model, two uses; finance vs science framing; same data → very different design choices; the prof returns to it whenever model design is on the table
- slug: supervised-vs-unsupervised
  canonical: Supervised vs unsupervised learning
  lectures: [L01, L02, L21]
  isl-ref: null
  exercises: []
  one-liner: have a Y to aim at vs not; LLM example as "supervised in disguise"; unsupervised is "dangerous statistics" without a downstream check
```

### Module 02: Statistical Learning (`02-statlearn`)

```yaml
- slug: parametric-vs-nonparametric
  canonical: Parametric vs nonparametric methods
  lectures: [L03]
  isl-ref: null
  exercises: []
  one-liner: linear regression vs KNN — assume a form and estimate parameters vs interpolate the data; trade flexibility for assumption-freedom
- slug: reducible-vs-irreducible-error
  canonical: Reducible vs irreducible error
  lectures: [L03, L04]
  isl-ref: null
  exercises:
    - CE1 problem 1b — derive the bias-variance decomposition, separate reducible and irreducible terms
    - CE1 problem 1c — interpret the three terms in words
  one-liner: E[(Y − Ŷ)²] = (f − f̂)² + Var(ε); the cross term vanishes because E[ε] = 0; the irreducible term is the noise floor
- slug: flexibility-overfitting-underfitting
  canonical: Flexibility, overfitting, underfitting, and the train/test MSE U-shape
  lectures: [L03, L04, L17]
  isl-ref: null
  exercises:
    - Exercise2.2 — flexible vs rigid methods, test error, overfit/underfit relation to bias-variance
    - Exercise2.5 — full polynomial-regression simulation: train MSE always falls, test MSE is U-shaped
  one-liner: training MSE always falls with flexibility, test MSE is U-shaped because flexible models eventually chase noise; KNN-K and polynomial degree are the standard knobs; *also* see double-descent for what happens past the U
- slug: knn-classification
  canonical: K-nearest neighbors (classification)
  lectures: [L03, L07, L08]
  isl-ref: null
  exercises:
    - Exercise4.1 — Euclidean distances by hand, predict class for K=1,4,7, why K=7 is bad
    - Exercise4.1c — relate best K to whether Bayes boundary is highly nonlinear
    - Exercise4.6g — fit KNN on Weekly with K=1
    - Exercise4.6h — sweep K=1..30, pick best K from error curve
  one-liner: non-parametric majority-vote classifier; small K → wiggly islands (overfit), big K → over-smoothed; killed by curse of dimensionality
- slug: knn-regression
  canonical: K-nearest neighbors (regression)
  lectures: [L10]
  isl-ref: null
  exercises:
    - CE1 problem 4a — write 10-fold CV pseudocode for KNN regression with MSE
    - CE1 problem 1e — read bias-variance plot for KNN over K
  one-liner: average the K nearest training y's; same K-as-flexibility story as KNN classification; the standard CV target for picking K
- slug: random-vector-and-covariance
  canonical: Random vectors, covariance and correlation matrices
  lectures: [L04, L05, L09]
  isl-ref: null
  exercises:
    - Exercise2.3g — compute correlations from the covariance matrix, compare to cor()
    - Exercise2.4 — simulate from multivariate normal with mvrnorm, four covariance settings
    - CE1 problem 1f — read correlation off a 2×2 covariance matrix
  one-liner: p-vector of random variables with covariance matrix Σ (variances on diagonal, covariances off); correlation matrix = V⁻¹ΣV⁻¹; expectation rules E(AXB) = A·E(X)·B; covariance is a *linear* notion (zero ≠ independent in general)
- slug: contrasts
  canonical: Contrasts (linear combinations)
  lectures: [L04]
  isl-ref: null
  exercises: []
  one-liner: a linear combination CX of the covariates (e.g. N−S on cork data); transformed via E(CX)=C·E(X), Cov(CX)=CΣCᵀ
```

### Module 03: Linear Regression (`03-linreg`)

```yaml
- slug: linear-regression
  canonical: Linear regression (simple and multiple)
  lectures: [L05, L06]
  isl-ref: null
  exercises:
    - Exercise3.1c — fit lm(mpg~.), interpret summary
    - Exercise3.1d — test importance of factor predictor with multiple levels
    - CE1 problem 2b — scatterplot, decide whether linearity makes sense, transform if not
    - CE1 problem 2c — fit additive lm with continuous + factor predictor, write three group equations
  one-liner: y = β₀ + β₁x₁ + … + βₚxₚ + ε; fit by least squares; closed-form β̂ = (XᵀX)⁻¹Xᵀy; the body-fat / BMI worked example; the simplest model with the deepest theoretical plumbing
- slug: least-squares-and-mle
  canonical: Least squares and the MLE equivalence
  lectures: [L05, L06, L12, L27]
  isl-ref: null
  exercises:
    - Exercise6.1a — derive β̂ = (XᵀX)⁻¹Xᵀy by differentiating RSS
    - Exercise6.1b — show MLE equals LS for the multiple linear regression model under Gaussian errors
  one-liner: minimize Σ(yᵢ − ŷᵢ)²; under εᵢ ~ N(0,σ²) i.i.d., maximizing the log-likelihood ⇔ minimizing SSE; *the* mathy theory question the prof flagged for the exam (Legendre got minimization, Gauss the distribution)
- slug: gaussian-error-assumptions
  canonical: Gaussian-error assumptions
  lectures: [L05, L06, L08]
  isl-ref: null
  exercises:
    - Exercise3.1e — autoplot diagnostic plots, comment on outliers and leverage
    - Exercise3.1h — try X transformations to fix residual issues
    - Exercise3.2e — error vs residual definitions and properties
  one-liner: εᵢ normal, mean zero, common variance, independent of X, independent of each other; the prof flagged independence (4 + 5) as the dangerous ones — "violations ruin everything"; spatial / temporal correlation is the canonical failure mode
- slug: design-matrix-and-hat-matrix
  canonical: Design matrix, normal equations, and the hat matrix
  lectures: [L06, L08, L10]
  isl-ref: null
  exercises:
    - Exercise6.1a — derive normal equations / closed form
  one-liner: X is n×(p+1) with intercept column; β̂ = (XᵀX)⁻¹Xᵀy from setting ∂RSS/∂β = 0; H = X(XᵀX)⁻¹Xᵀ "puts hats on Y" — its diagonal is leverage, it appears in the LOOCV shortcut, and (XᵀX)⁻¹ blowing up is what collinearity does
- slug: sampling-distribution-of-beta
  canonical: Sampling distribution of β̂
  lectures: [L05, L06]
  isl-ref: null
  exercises:
    - Exercise3.2a — derive distribution of β̂ in matrix form, find Var(β̂ⱼ)
  one-liner: under Gaussian assumptions β̂ ~ N(β, σ²(XᵀX)⁻¹); SE²(β̂₁) = σ²/Σ(xᵢ−x̄)² in the simple case; tells you to design experiments with bigger n and wider X-spread
- slug: confidence-and-prediction-intervals
  canonical: Confidence and prediction intervals
  lectures: [L05, L06]
  isl-ref: null
  exercises:
    - Exercise3.2b — simulate to demonstrate the frequentist CI interpretation
    - Exercise3.2c — simulate the PI at a fixed x₀
    - Exercise3.2d — CI for x₀ᵀβ vs PI for Y at x₀
  one-liner: CI = where the *mean response* lies (β̂ ± t·SE); PI = where a *future observation* lands; PI always wider because it adds σ²; both narrowest where data is dense
- slug: t-test-and-significance
  canonical: t-test, p-values, statistical vs practical significance
  lectures: [L05, L06]
  isl-ref: null
  exercises:
    - CE1 problem 2g — true/false on what a p-value does and does not mean
  one-liner: t = β̂ⱼ / SE(β̂ⱼ), df = n − p − 1; large n inflates significance for trivial slopes ("significance is just sample size"); engineering vs biology emphasis; "ideally you want both"
- slug: r-squared
  canonical: R² and adjusted R²
  lectures: [L05, L06]
  isl-ref: null
  exercises: []
  one-liner: R² = 1 − RSS/TSS, fraction of variance explained; never decreases as parameters added; adjusted R² penalizes parameter count via (n−1)/(n−p−1); prof distrusts both, prefers test-set error
- slug: residual-diagnostics
  canonical: Residual diagnostics (residual plot, QQ plot, leverage, studentized residuals)
  lectures: [L06, L08]
  isl-ref: null
  exercises:
    - Exercise3.1e — autoplot diagnostic plots, comment on residuals and leverage
    - Exercise3.1f — simulate normal data, see how "wonky" QQ plots can look even under correct assumptions
    - CE1 problem 2e — autoplot residual analysis with vs without transformations
    - CE1 problem 2f — why residual analysis matters, what to do under violations
  one-liner: residuals vs fitted (constant variance, structure), QQ plot (normality — "the kind of thing I would put on a test"), leverage = hᵢᵢ ("fat kid on the seesaw" — high leverage *and* large residual is the dangerous combination), studentized residuals rescale to ~constant variance
- slug: collinearity
  canonical: Collinearity / multicollinearity
  lectures: [L06, L08, L14, L15]
  isl-ref: null
  exercises: []
  one-liner: correlated predictors → XᵀX near-singular → β's trade off, SEs explode, fits unstable; fixes are dropping a variable, ridge, or PCA/PCR; pathological in p > n
- slug: f-test
  canonical: F-test for regression
  lectures: [L06]
  isl-ref: null
  exercises: []
  one-liner: tests H₀: all βⱼ = 0 (or a subset); generalizes the t-test, reduces to it for p=1; **prof said he won't ask you to compute it**, only why you'd use it (Q1 of the four important questions)
- slug: categorical-encoding-and-interactions
  canonical: Categorical predictors, dummy coding, and interactions
  lectures: [L05, L06]
  isl-ref: null
  exercises:
    - Exercise3.1d — work with factor variable origin (3 levels)
    - Exercise3.1g — fit lm with year × origin interaction, interpret
    - CE1 problem 2c — fit lm with continuous + factor, write three group equations
    - CE1 problem 2d — test whether MAGENUMF × Gattung interaction matters
  one-liner: K levels need K−1 dummies + reference category; never code 0/1/2 (imposes ordering); main-effects rule: if you include X·Z you must include X and Z; main-effect coefficient on Z only describes Z=1 *at X=0* — the canonical interaction trap (and a flagged exam question)
- slug: polynomial-regression
  canonical: Polynomial regression
  lectures: [L06, L16]
  isl-ref: null
  exercises:
    - Exercise2.5 — full polynomial-regression simulation for bias-variance
    - Exercise7.1 — fit polynomials of degree 1..4 to mpg ~ horsepower, plot test error vs degree
  one-liner: still linear regression — linear in the parameters β even though x is quadratic / cubic / ...; the standard simulation playground for bias-variance and double descent
```

### Module 04: Classification (`04-classif`)

```yaml
- slug: classification-setup
  canonical: Classification setup, Bayes classifier, and Bayes error rate
  lectures: [L07, L08, L09]
  isl-ref: null
  exercises: []
  one-liner: response Y is categorical; estimate P(Y|X), assign argmax_k (or threshold for binary); the Bayes classifier is provably optimal — its error floor (Bayes error rate) is the classification analogue of irreducible error
- slug: logistic-regression
  canonical: Logistic regression
  lectures: [L07, L08, L09, L17, L27]
  isl-ref: null
  exercises:
    - Exercise4.4 — given β₀, β₁, β₂ for hours-studied/GPA, compute P(A) for one student, then required hours for P=0.5
    - Exercise4.6b — fit glm(Direction~., binomial), interpret summary
    - Exercise4.6c — confusion matrix for full-data logistic regression
    - Exercise4.6d — train/test split logistic on Lag2, confusion matrix on held-out
    - CE1 problem 3a — show logit(p) is linear in covariates
    - CE1 problem 3b — interpret β₁ for an additional ace
    - CE1 problem 3c — fit logistic with ACEdiff/UFEdiff, derive class boundary, plot, sensitivity/specificity
  one-liner: Bernoulli GLM, log(p/(1−p)) = β₀ + βᵀx; sigmoid σ(η) = 1/(1+e^−η) keeps p ∈ (0,1); fit by MLE via Newton (no closed form); same independence/collinearity assumptions as OLS; coefficient = log odds-ratio
- slug: odds-and-log-odds
  canonical: Odds, log-odds, and odds-ratio interpretation
  lectures: [L07, L09, L27]
  isl-ref: null
  exercises:
    - Exercise4.3a — given odds 0.37, find P(default)
    - Exercise4.3b — given P=0.16, find odds
  one-liner: odds = p/(1−p); a one-unit increase in xⱼ multiplies the odds by e^βⱼ; not the probability — this is the easy interaction-trap question on the exam (units of x_j matter; reference-class flip flips signs)
- slug: linear-discriminant-analysis
  canonical: Linear discriminant analysis (LDA)
  lectures: [L08, L09]
  isl-ref: null
  exercises:
    - Exercise4.2a — pooled covariance estimator across two groups
    - Exercise4.2b — write the LDA classification rule for new observation
    - Exercise4.2c — classify a length=214/diagonal=140.4 bank note via LDA
    - Exercise4.6e — repeat lag-based classification with LDA
    - CE1 problem 3d — interpret π_k, μ_k, Σ, f_k(x) in LDA
    - CE1 problem 3e — derive δ_k(x), solve for class boundary, plot
    - CE1 problem 3f — perform LDA in R, confusion matrix, sensitivity/specificity
  one-liner: model P(X|Y=k) as Gaussian with class means μ_k and *pooled* Σ; flip via Bayes; throw away k-independent terms → discriminant linear in x; pooled covariance = Σ_k (n_k−1)/(n−K) · S_k
- slug: quadratic-discriminant-analysis
  canonical: Quadratic discriminant analysis (QDA)
  lectures: [L09]
  isl-ref: null
  exercises:
    - Exercise4.2d — derive QDA rule, classify the bank note, compare to LDA result
    - Exercise4.6f — repeat lag-based classification with QDA
    - CE1 problem 3g — perform QDA, confusion matrix, sensitivity/specificity
    - CE1 problem 3h — compare LDA, QDA, logistic decision boundaries
  one-liner: drop the pooled-Σ assumption — each class gets Σ_k → the xᵀΣ_k⁻¹x term no longer cancels → boundary is **quadratic**; "where does the quadratic come from?" is exam-flagged; more flexible, more parameters, more prone to overfit
- slug: discriminant-score-and-decision-boundary
  canonical: Discriminant score and decision-boundary derivation
  lectures: [L09, L27]
  isl-ref: null
  exercises:
    - CE1 problem 3e — derive δ_k(x) from Bayes' rule, solve δ_0 = δ_1 for the boundary
  one-liner: δ_k(x) = xᵀΣ⁻¹μ_k − ½μ_kᵀΣ⁻¹μ_k + log π_k (LDA); set δ_k(x) = δ_l(x) and solve for x → boundary is a hyperplane; "this would be a typical exam question" said twice; worked example μ_A=(1,1), μ_B=(3,3), Σ=2I → x₁ + x₂ = 4
- slug: naive-bayes
  canonical: Naive Bayes
  lectures: [L09]
  isl-ref: null
  exercises: []
  one-liner: assume Σ is diagonal (predictors conditionally independent given class); fewer parameters → preferred when p is large; same Bayesian-discriminant machinery as LDA/QDA
- slug: confusion-matrix
  canonical: Confusion matrix
  lectures: [L09, L10, L18]
  isl-ref: null
  exercises:
    - Exercise4.6c — compute confusion matrix and overall correct fraction
    - Exercise4.6d — confusion matrix on test set
    - CE1 problem 3c — confusion matrix from logistic regression with 0.5 cutoff
    - CE1 problem 3f — confusion matrix from LDA
    - CE1 problem 3g — confusion matrix from QDA
  one-liner: 2×2 (or K×K) of TP/FP/FN/TN; read sensitivity/specificity/accuracy from it; in-sample matrices always flatter the model — that's overfitting
- slug: sensitivity-specificity
  canonical: Sensitivity and specificity
  lectures: [L09, L10, L27]
  isl-ref: null
  exercises:
    - Exercise4.5a — define sensitivity and specificity for a disease/non-disease classifier
    - CE1 problem 3c — calculate sensitivity and specificity from a confusion matrix
    - CE1 problem 3f — same for LDA
    - CE1 problem 3g — same for QDA
  one-liner: sens = TP/(TP+FN), spec = TN/(TN+FP); justice-system analogy for the trade-off; on the exam, *write the formula* even if you don't plug in numbers
- slug: roc-auc
  canonical: ROC curve and AUC
  lectures: [L09, L10, L27]
  isl-ref: null
  exercises:
    - Exercise4.5b — explain how to construct a ROC curve and why
    - Exercise4.5c — define AUC, choose between p(x) (AUC=0.6) and q(x) (AUC=0.7)
    - Exercise4.6j — plot ROC and compute AUC for glm/lda/qda/knn on Weekly
  one-liner: sweep classifier threshold, plot (1−spec, sens); AUC summarizes; diagonal = chance, below diagonal = invert your classifier
- slug: curse-of-dimensionality
  canonical: Curse of dimensionality
  lectures: [L07, L08, L15]
  isl-ref: null
  exercises: []
  one-liner: in high-d, all pairwise distances become nearly equal → "nearest neighbor" stops meaning anything; kills KNN; the high-dim multicollinearity story for OLS
- slug: diagnostic-vs-sampling-paradigm
  canonical: Diagnostic vs sampling (generative) paradigm
  lectures: [L07, L09]
  isl-ref: null
  exercises: []
  one-liner: model P(Y|X) directly (logistic, KNN) vs model class-conditionals f_k(x) and priors π_k then flip via Bayes (LDA, QDA, Naive Bayes); the conceptual divider in module 4
```

### Module 05: Resampling (`05-resample`)

```yaml
- slug: training-validation-test-split
  canonical: Training / validation / test split
  lectures: [L10]
  isl-ref: null
  exercises: []
  one-liner: three partitions with three jobs — fit, select, assess; reusing test for selection makes you "too optimistic"; foundational for everything in modules 5–11
- slug: validation-set-approach
  canonical: Validation set approach
  lectures: [L10, L11]
  isl-ref: null
  exercises: []
  one-liner: random ~50/50 split, fit on one half, evaluate on the other; high variance across splits, biased upward; conservative and easy to explain
- slug: leave-one-out-cv
  canonical: Leave-one-out cross-validation (LOOCV)
  lectures: [L10, L11]
  isl-ref: null
  exercises:
    - CE1 problem 4b — true/false on bias and variance of LOOCV vs k-fold
  one-liner: train on n−1, hold out 1, repeat n times; low bias, high variance (folds nearly identical → highly correlated); for OLS the hat-matrix shortcut CV = (1/n)Σ((yᵢ − ŷᵢ)/(1−hᵢᵢ))² needs only one fit
- slug: k-fold-cv
  canonical: k-fold cross-validation
  lectures: [L10, L11, L18, L19]
  isl-ref: null
  exercises:
    - Exercise5.1 — describe the k-fold CV algorithm with figure, aggregation, polynomial / KNN examples
    - Exercise5.2 — compare k-fold to validation-set and LOOCV (bias / variance / compute)
    - Exercise6.3b — wrap CV around best-subset selection on Credit
    - Exercise8.2c — cv.tree() to find optimal pruning size for regression tree
    - Exercise8.3e — cv.tree() with prune.misclass for classification spam tree
    - CE1 problem 4a — write 10-fold CV pseudocode for KNN regression with MSE
    - CE1 problem 4b — true/false on k-fold vs LOOCV
  one-liner: split into k blocks, hold each out once; k=5 or 10 is the bias-variance compromise; the prof's preferred everything-tuner
- slug: one-standard-error-rule
  canonical: One-standard-error rule
  lectures: [L10]
  isl-ref: null
  exercises: []
  one-liner: among models within 1 SE of the CV minimum, pick the simplest; prof's preferred selection criterion; SE strictly speaking "not quite valid" because you reused the held-out folds for both selection and SE estimation
- slug: nested-cv-and-cv-pitfalls
  canonical: Nested CV and the wrong-way CV trap
  lectures: [L11]
  isl-ref: null
  exercises:
    - Exercise5.3 — simulate p=5000 random predictors with random labels; show wrong-way CV gives ~0% misclass while right-way gives ~50% (the truth)
  one-liner: outer folds = honest assessment, inner folds = selection; variable selection with the labels lives *inside* the CV loop — doing it once outside leaks holdout into selection ("wrong way"); misclassification can drop to zero on pure noise — flagged as a canonical "lying with statistics" trap
- slug: aic-bic-conceptual
  canonical: Penalty criteria (AIC / BIC / Cp) — conceptual only
  lectures: [L10, L12, L13]
  isl-ref: null
  exercises:
    - Exercise6.3a — pick best subset using Cp, BIC, adjusted R² on Credit (cross-check against CV)
  one-liner: training-error penalties for model complexity; **derivations and formulas explicitly out of scope** ("I really don't think I'm going to ask any questions about this"); know they exist and what they're for; prof distrusts them, prefers CV
- slug: bootstrap
  canonical: Bootstrap
  lectures: [L11]
  isl-ref: null
  exercises:
    - Exercise5.4 — compute probability that observation i is in a bootstrap sample, derive 1 − 1/e ≈ 0.632
    - Exercise5.5 — describe bootstrap algorithm for SE / CI of a regression coefficient
    - Exercise5.6 — implement bootstrap for SE of an OLS coefficient (for-loop and boot package), compare to (XᵀX)⁻¹σ̂²
    - CE1 problem 4d — B=1000 bootstrap to estimate SE and 95% CI of a logistic-regression-derived probability
  one-liner: resample with replacement, compute statistic on each, histogram the distribution; "your best model for the world is the data itself"; B = 1000–10000; works for arbitrary statistics where no closed-form distribution exists
- slug: bagging
  canonical: Bagging (bootstrap aggregating)
  lectures: [L11, L18, L19]
  isl-ref: null
  exercises:
    - Exercise8.2d — bagging with 500 trees on Carseats, compute test MSE, importance()
    - Exercise8.3f — bagging on spam with B=500
  one-liner: fit B models on B bootstrap samples, average (regression) or vote (classification); variance reduction with floor ρσ² + (1−ρ)σ²/B due to non-IID samples; precursor to random forests
- slug: out-of-bag-error
  canonical: Out-of-bag error
  lectures: [L18, L19]
  isl-ref: null
  exercises:
    - Exercise8.1d — explain OOB sample, what fraction of observations are OOB
  one-liner: ~1/3 of obs are excluded from each bootstrap sample (1 − 1/e ≈ 0.368) → free per-tree validation set; aggregate across trees → no separate test set required
```

### Module 06: Model Selection and Regularization (`06-modelsel`)

```yaml
- slug: subset-selection
  canonical: Subset selection (best, forward, backward, hybrid)
  lectures: [L12, L13]
  isl-ref: null
  exercises:
    - Exercise6.3a — best-subset on Credit via regsubsets, pick model by Cp/BIC/adjR²
    - Exercise6.3b — best-subset selection scored by 10-fold CV
    - Exercise6.3c — compare CV vs penalty-criterion choices
    - Exercise6.4a — forward, backward, hybrid stepwise on Credit
    - Exercise6.4b — compare with best-subset
  one-liner: best-subset = try all 2^p (combinatorially infeasible past modest p); forward = greedy add; backward = greedy remove (requires n > p); hybrid = forward + backward interleaved (can recover best-subset answer that pure forward misses)
- slug: ridge-regression
  canonical: Ridge regression (L2)
  lectures: [L04, L12, L13, L14, L15, L24]
  isl-ref: null
  exercises:
    - Exercise6.5 — apply ridge to Credit, compare with OLS
  one-liner: minimize RSS + λΣβⱼ²; smooth shrinkage that *never* hits zero; closed-form, works when p > n; **standardize predictors first**; tug-of-war between RSS pulling β's away from zero and penalty pulling them back
- slug: lasso
  canonical: Lasso (L1)
  lectures: [L13, L14, L15, L24, L26]
  isl-ref: null
  exercises:
    - Exercise6.6 — apply lasso to Credit, compare with ridge and OLS
  one-liner: minimize RSS + λΣ|βⱼ|; L1's corners on the axes drive coefficients to *exactly* zero → variable selection for free; "capitalist" regularizer
- slug: ridge-vs-lasso-geometry
  canonical: Ridge vs lasso geometric interpretation
  lectures: [L13, L14]
  isl-ref: null
  exercises: []
  one-liner: RSS ellipses meet the L1 diamond at corners (sparsity) vs the L2 ball at smooth interior points (no zeros); L2 averages over correlated parameters ("socialist"), L1 picks one ("capitalist"); worth learning to interpret per the prof
- slug: elastic-net
  canonical: Elastic net
  lectures: [L13, L14]
  isl-ref: null
  exercises: []
  one-liner: combine L1 + L2 penalties; sparsity from lasso plus correlated-variable averaging from ridge; "probably the one people use the most"
- slug: principal-component-regression
  canonical: Principal components regression (PCR)
  lectures: [L08, L14, L15]
  isl-ref: null
  exercises:
    - Exercise6.7 — how many PCs to use for Credit
    - Exercise6.8 — apply PCR on Credit, compare to other module 6 methods
  one-liner: standardize → PCA → regress y on first M PCs → back-transform to β; orthogonal Z's kill multicollinearity; **PCR ≈ a discretized ridge**; unsupervised compression so it can miss directions Y depends on
- slug: partial-least-squares
  canonical: Partial least squares (PLS)
  lectures: [L14, L15]
  isl-ref: null
  exercises:
    - Exercise6.9 — apply PLS on Credit, compare to PCR and other methods
  one-liner: like PCR but choose components to maximize Cov(Z, Y) instead of Var(Z); supervised dimensionality reduction; "no better than ridge or PCR but it's Swedish"
- slug: high-dimensional-regression
  canonical: High-dimensional regression (p > n)
  lectures: [L15]
  isl-ref: null
  exercises: []
  one-liner: p > n breaks OLS — XᵀX singular, R² = 1 always, σ̂² unreliable; AIC/BIC/Cp don't save you; multicollinearity is pathological; regularization makes the problem well-posed and is *the* reason this module exists
```

### Module 07: Moving Beyond Linearity (`07-beyondlinear`)

```yaml
- slug: basis-functions
  canonical: Basis functions
  lectures: [L16, L17]
  isl-ref: null
  exercises: []
  one-liner: replace x with b_j(x) and fit linear regression on the transformed columns; polynomial / step / spline / GAM all share this structure; *linear in the parameters* even when curvy in x
- slug: step-functions
  canonical: Step functions
  lectures: [L16]
  isl-ref: null
  exercises: []
  one-liner: piecewise-constant fit on chosen intervals; "stupid but actually pretty common"; no derivatives, jumps at cutpoints
- slug: regression-splines
  canonical: Regression splines (cubic and natural)
  lectures: [L16, L17, L26]
  isl-ref: null
  exercises:
    - Exercise7.3 — derive design matrix for a natural cubic spline with one internal knot
    - Exercise7.4 — build the design matrix by hand using truncated cubic basis, verify gam() gives same fit
  one-liner: piecewise cubics joined with continuous 0/1/2-derivatives at knots; basis = (x, x², x³, (x−c_j)³₊); natural splines force linear extrapolation past boundary knots; param count = K + d + 1
- slug: smoothing-splines
  canonical: Smoothing splines and effective degrees of freedom
  lectures: [L16, L17, L27]
  isl-ref: null
  exercises:
    - Exercise7.5 — fit GAM with smoothing-spline component (s(acceleration, df=3))
  one-liner: minimize Σ(yᵢ − g(xᵢ))² + λ∫g''(t)²dt over functions g; **λ ↑ → smoother (straight line at ∞), λ = 0 → wiggly fit** — easy T/F trap (opposite direction from polynomial degree); choose λ by LOOCV; effective df = trace(S) where ŷ = Sy (non-integer)
- slug: local-regression
  canonical: Local regression (LOESS)
  lectures: [L16, L17]
  isl-ref: null
  exercises:
    - Exercise7.5 — GAM uses lo() for local-linear component
  one-liner: fit a local linear regression at every x₀ weighted by a Gaussian kernel around x₀; "smoothed K-nearest-neighbors"; Gaussian width plays the role of K
- slug: generalized-additive-models
  canonical: Generalized additive models (GAMs)
  lectures: [L16, L17, L27]
  isl-ref: null
  exercises:
    - Exercise7.5 — fit gam() with cubic-spline displacement, polynomial horsepower, linear weight, smoothing-spline acceleration, factor origin
  one-liner: y = β₀ + Σⱼ fⱼ(xⱼ) + ε with each fⱼ chosen freely (poly, spline, LOESS, indicator); additive — no interactions; logistic GAM for binary Y is the same trick on the log-odds
```

### Module 08: Tree-Based Methods (`08-trees`)

```yaml
- slug: regression-tree
  canonical: Regression tree
  lectures: [L17, L18, L19]
  isl-ref: null
  exercises:
    - Exercise8.1a — explain regression-tree fitting algorithm; classification differences
    - Exercise8.2b — fit regression tree to Carseats Sales, plot, interpret, test MSE
  one-liner: chunk predictor space into rectangles R₁..R_J via greedy recursive binary splitting (CART); predict region mean for each; loss = Σ_m Σ_{i∈R_m}(yᵢ − ȳ_{R_m})²; captures interactions naturally — the prof's first algorithmic model
- slug: classification-tree
  canonical: Classification tree (Gini, cross-entropy)
  lectures: [L18]
  isl-ref: null
  exercises:
    - Exercise8.3c — fit tree() on spam, summary, plot, count terminal nodes
    - Exercise8.3d — predict on test set, misclassification rate
  one-liner: same algorithm, predict majority class per region (or class proportions); split on **Gini** Σ_k p̂_{mk}(1 − p̂_{mk}) or **cross-entropy** −Σ p̂ log p̂ (both purity-sensitive, differentiable) not misclassification; **prune on misclassification** — train one criterion, prune on another
- slug: cost-complexity-pruning
  canonical: Cost-complexity pruning
  lectures: [L17, L18]
  isl-ref: null
  exercises:
    - Exercise8.2c — cv.tree() then pick best size, compare pruned vs unpruned MSE
    - Exercise8.3e — cv.tree() with prune.misclass on spam
  one-liner: minimize Σ_m Σ_{i∈R_m}(yᵢ − ŷ_{R_m})² + α|T|; build out fully then prune backward via weakest-link; choose α by k-fold CV; **early stopping is wrong** because bad-looking splits enable good later ones — grow then prune
- slug: random-forest
  canonical: Random forest
  lectures: [L18, L19, L27]
  isl-ref: null
  exercises:
    - Exercise8.2e — randomForest with mtry=3, ntree=500; importance plot
    - Exercise8.3g — random forest on spam with subset predictors per split
  one-liner: bagging + restrict each split to m random predictors → decorrelates trees → reduces the ρσ² floor; defaults m = √p classification, m = p/3 regression; B is "use enough" (not tuned); the variance-reduction win over plain bagging
- slug: variable-importance
  canonical: Variable importance
  lectures: [L18, L19, L27]
  isl-ref: null
  exercises:
    - Exercise8.2d — importance() output for bagged Carseats
    - Exercise8.2e — importance() for RF
    - Exercise8.3g — importance() for RF on spam
    - Exercise8.2f — sweep ntree, plot test MSE vs ntree for bagging vs RF
  one-liner: impurity-based (sum decrease in Gini/RSS across splits) vs randomization-based (permute predictor on OOB and measure drop); prof prefers randomization "because it makes more sense"; both valid
```

### Module 09: Boosting and Additive Trees (`09-boosting`)

```yaml
- slug: boosting
  canonical: Boosting (forward stagewise additive modeling)
  lectures: [L19, L20, L21, L27]
  isl-ref: null
  exercises:
    - Exercise9.1 — what is an ensemble; bagging vs boosting differences
    - Exercise9.2 — explain L(.), f_m(.), M, J_m in the gradient tree boosting algorithm
  one-liner: sequentially fit weak learners that each correct the previous fit; sum-of-trees model f_M(x) = Σ_m T(x;Θ_m); concept-only mechanics (prof: "you don't need to memorize the pseudocode")
- slug: adaboost
  canonical: AdaBoost
  lectures: [L19, L20]
  isl-ref: null
  exercises: []
  one-liner: sequentially fit weak classifiers to *re-weighted* training data (misclassified points get weight e^{α_m}); final = sign of α-weighted vote; α_m = ½ log((1−err_m)/err_m); special case of gradient boosting under exponential loss
- slug: gradient-boosting
  canonical: Gradient boosting
  lectures: [L19, L20, L21]
  isl-ref: null
  exercises:
    - Exercise9.4a — fit a basic gbm() on simulated genomic data
    - Exercise9.4b — stochastic GBMs via h2o, hyperparameter grid
    - Exercise9.4c — XGBoost via xgb.cv on the same data
    - Exercise9.4d — find the best model, fit on full training data
  one-liner: fit each new tree to the *negative gradient* of the loss; for squared-error this is the residuals; algorithm 10.3 generalizes to any differentiable loss; the GBM family
- slug: weak-learner-and-learning-rate
  canonical: Weak learners and the learning rate ν
  lectures: [L19, L20]
  isl-ref: null
  exercises:
    - Exercise9.3 — interpretation of ν, how to choose it, where it enters the algorithm
  one-liner: deliberately small / shallow tree (depth 1 = stump, depth 4–8 typical) so the ensemble averages cleanly; ν ≤ 0.1 typical; smaller ν → need larger M; tune by early stopping on a validation set
- slug: stochastic-gradient-boosting
  canonical: Stochastic gradient boosting
  lectures: [L20]
  isl-ref: null
  exercises:
    - Exercise9.4b — h2o stochastic GBM with row/column subsampling
  one-liner: subsample rows (without replacement) and/or columns before each tree → diversity → variance reduction; bag.fraction ≤ 1; same idea as random forests applied to boosting
- slug: xgboost
  canonical: XGBoost
  lectures: [L20, L21]
  isl-ref: null
  exercises:
    - Exercise9.4c — xgb.cv on simulated data
    - Exercise9.4d — tune XGBoost hyperparameters
  one-liner: 2nd-order (Newton) gradients + parallelism + L1/L2 leaf-weight regularization + dropout; "the one everyone uses"; concept matters, internals don't
- slug: boosting-loss-functions
  canonical: Boosting loss functions (quadratic / absolute / Huber / deviance)
  lectures: [L20]
  isl-ref: null
  exercises: []
  one-liner: any differentiable loss plugs into gradient boosting; squared-error is sensitive to outliers and won't sparsify; Huber is the smooth compromise (quadratic near 0, linear far out); binomial / multinomial deviance for classification (multinomial → K trees per round)
- slug: partial-dependence-plots
  canonical: Partial dependence plots
  lectures: [L21, L27]
  isl-ref: null
  exercises: []
  one-liner: average f̂(x_j, X_{−j}) over the data with x_j fixed → claw back interpretability from tree ensembles; fair-game on the exam to ask what they show
```

### Module 10: Unsupervised Learning (`10-unsuper`)

```yaml
- slug: principal-component-analysis
  canonical: Principal component analysis (PCA)
  lectures: [L14, L15, L21]
  isl-ref: null
  exercises:
    - Exercise6.7 — how many PCs for the Credit dataset
    - Exercise10.1 — biplot, PVE plot, cumulative PVE on the NYT stories data
  one-liner: find unit-norm linear combination φ_m of standardized X with maximal variance, orthogonal to previous PCs; eigenvectors of the sample covariance, eigenvalues = PC variances; loadings φ_{jm} interpret PCs in terms of original variables; **standardize first** — not scale invariant
- slug: explained-variance-and-scree-plot
  canonical: PCA explained variance and scree plot
  lectures: [L14, L15, L21]
  isl-ref: null
  exercises:
    - Exercise10.1 — produce PVE and cumulative PVE plots, decide where to cut
  one-liner: PVE_m = λ_m / Σ λ_k; cumulative-variance threshold (90/95/99%) tells you how many PCs to keep; the standard chop-here heuristic
- slug: k-means-clustering
  canonical: K-means clustering
  lectures: [L22]
  isl-ref: null
  exercises:
    - Exercise10.2 — show the k-means algorithm decreases the objective at each step
    - Exercise10.3 — perform k-means on NYT stories
  one-liner: minimize within-cluster Σ pairwise squared Euclidean distances; iterate centroid → assign → centroid; **converges to local minima** — rerun with multiple inits, "random init can really screw you"; K is a hyperparameter you must pick
- slug: hierarchical-clustering
  canonical: Hierarchical (agglomerative) clustering and linkage
  lectures: [L22, L27]
  isl-ref: null
  exercises:
    - Exercise10.4 — perform hierarchical clustering on NYT stories
  one-liner: each obs starts as its own cluster, repeatedly merge closest pair, draw a dendrogram (cut height = number of clusters); linkage = single (min)/complete (max)/average (mean) of inter-cluster pairwise distances — average and complete give balanced clusters; **"−1 point per mistake"** if asked to do by hand on the exam
- slug: dimensionality-reduction
  canonical: Dimensionality reduction
  lectures: [L08, L09, L14, L15, L21]
  isl-ref: null
  exercises: []
  one-liner: collapse p predictors into M < p composite ones for visualization or downstream regression; PCA, PCR, PLS, LDA-as-projection, NN-feature-extractor all fit this frame
- slug: distance-metrics
  canonical: Distance metrics for clustering
  lectures: [L22]
  isl-ref: null
  exercises: []
  one-liner: Euclidean groups infrequent shoppers; correlation (1−ρ) groups people with similar preferences; cosine, Manhattan, Wasserstein etc. exist; metric choice is the headline hyperparameter — different distances, different clusterings
```

### Module 11: Neural Networks (`11-nnet`)

```yaml
- slug: feedforward-network
  canonical: Feedforward neural network
  lectures: [L22, L23, L24]
  isl-ref: null
  exercises:
    - Exercise11.1a — write the input/output equation for a given network with general activation functions
    - Exercise11.1b — interpret the architecture of an illustrated network
    - Exercise11.1c — compare a feedforward NN with linear hidden + sigmoid output to logistic regression
    - Exercise11.1d — how can a 10000-weight model fit on 1000 obs
    - Exercise11.2a — given the formula, identify the architecture and count parameters (1 hidden, ReLU)
    - Exercise11.2b — same for a 2-hidden-layer ReLU + sigmoid output
    - Exercise11.2c — compare GAM vs FNN approach to multi-covariate regression
    - Exercise11.3 — fit a Keras feedforward NN to Boston housing, compare to linear regression
  one-liner: input → hidden(s) → output, no loops; nested z_m = a(α₀_m + Σⱼα_jm xⱼ) → ŷ = f(β₀ + Σ_m β_m z_m); param count (P+1)M + (M+1)C; M is the engineering hyperparameter
- slug: activation-functions
  canonical: Activation functions (sigmoid, ReLU, GELU, softmax)
  lectures: [L22, L23]
  isl-ref: null
  exercises:
    - Exercise11.2a — compute the output of a single ReLU neuron given weights, inputs, bias
  one-liner: nonlinearity is what makes the network expressive — without it, the whole thing collapses to linear regression; sigmoid (historical, brain-inspired), ReLU/GELU (modern hidden, nicer gradients), softmax e^{z_c}/Σe^{z_k} (multi-class output)
- slug: nn-parameter-count
  canonical: Neural network parameter count
  lectures: [L23]
  isl-ref: null
  exercises:
    - Exercise11.2a — count parameters in a 1-hidden-layer ReLU network
    - Exercise11.2b — count parameters in a 2-hidden-layer ReLU + sigmoid network
  one-liner: per layer, weights between layers + bias for each receiving unit; one-hidden: M(p+1) + C(M+1); **forgetting bias is the canonical wrong answer**; explicit prof-flagged exam-likely calculation
- slug: universal-approximation
  canonical: Universal approximation
  lectures: [L23]
  isl-ref: null
  exercises: []
  one-liner: a single hidden layer of squashing units with enough width can approximate any Borel-measurable function; existence result, not a recipe; **proof out of scope** (measure theory)
- slug: gradient-descent-and-sgd
  canonical: Gradient descent, SGD, and mini-batch
  lectures: [L23, L24, L26]
  isl-ref: null
  exercises: []
  one-liner: θ ← θ − λ∇_θL; mini-batch of m ≪ N gives unbiased gradient estimate; powers-of-2 batch sizes for hardware; speeds training and provides **implicit L2 regularization** (chooses min-norm solution among interpolators) — the prof's headline regularization fact
- slug: backpropagation
  canonical: Backpropagation
  lectures: [L23, L24]
  isl-ref: null
  exercises: []
  one-liner: chain rule applied so intermediates from the forward pass are reused in the backward pass; δ propagates from output to input one layer at a time; only works for feedforward / acyclic architectures (why RNNs are harder)
- slug: nn-loss-functions
  canonical: NN loss functions
  lectures: [L23]
  isl-ref: null
  exercises: []
  one-liner: MSE for regression, binary cross-entropy for binary, categorical cross-entropy + softmax for multi-class; same shape as the GLM losses, applied to the network's output
- slug: nn-regularization
  canonical: Neural-network regularization (L1/L2, augmentation, label smoothing, early stopping, dropout, transfer learning)
  lectures: [L24, L26]
  isl-ref: null
  exercises:
    - Exercise11.4.2 — apply rotation/shift/flip data augmentation to CIFAR-10 CNN
  one-liner: never train a NN without regularization; menu = L1/L2 weight decay, data augmentation (rotate/flip/shift/noise), label smoothing, early stopping (stop at validation minimum), dropout (zero 20–50% of nodes per training step — never 50%), transfer learning (warm-start from a pretrained model)
- slug: convolutional-neural-network
  canonical: Convolutional neural network (CNN)
  lectures: [L24]
  isl-ref: null
  exercises:
    - Exercise11.4.1 — build a CNN for CIFAR-10 (conv → maxpool → conv → maxpool → flatten → dense → softmax)
    - Exercise11.4.1b — compute misclassification from CIFAR confusion matrix
    - Exercise11.5 — 1D-CNN for Wafer time-series classification, compare to logistic regression
  one-liner: feedforward with shared learned local filters convolved across spatial / temporal dims; pooling (typically max-pool) shrinks spatial extent and preserves peaks; just feedforward → backprop drops in unchanged; **architecture details out of scope** (per L27)
- slug: recurrent-neural-network
  canonical: Recurrent neural network (RNN)
  lectures: [L26]
  isl-ref: null
  exercises: []
  one-liner: hidden state A_t carries info across the sequence: A_t = σ(b + W X_t + U A_{t−1}); same weights at every step (weight sharing); precursor to language models; **architecture details out of scope** (LSTM/GRU gates, BPTT)
```

### Module 12: Final / Exam Review (`12-final`)

Module 12 holds no atoms of its own — it's the exam-review session. All exam-relevant content lives in the atoms above (cross-referenced from `wiki/lectures/L27-summary.md`). The MOC for module 12 will route to L27, the canonical scope rule, and the Specials atoms (especially `bias-variance-tradeoff`, `cross-validation`, and `regularization`).

## Specials (cross-cutting atoms with no single owning module)

These concepts genuinely span multiple modules and have no natural single owning module. The fan-out's specials agent writes these. Each entry uses the **plural `modules:`** field to signal "specials" (regular atoms use singular `module:`).

```yaml
- slug: bias-variance-tradeoff
  canonical: Bias-variance tradeoff (decomposition)
  modules: [02-statlearn, 03-linreg, 04-classif, 05-resample, 06-modelsel, 07-beyondlinear, 08-trees, 09-boosting, 11-nnet]
  lectures: [L02, L03, L04, L05, L09, L10, L11, L12, L13, L14, L18, L20, L23, L24, L26, L27]
  isl-ref: null
  exercises:
    - Exercise2.2 — bias-variance trade-off in flexible vs rigid methods, over/underfit
    - Exercise2.5 — full polynomial-regression simulation: trainMSE, testMSE, decompose into bias², variance, irreducible
    - Exercise5.2 — bias / variance / compute trade-offs across CV schemes
    - Exercise8.1c — role of bagging / random forest in attacking variance
    - CE1 problem 1a — write down expected test MSE at x₀
    - CE1 problem 1b — derive the bias-variance decomposition (3 terms)
    - CE1 problem 1c — interpret the three terms in words
    - CE1 problem 1d — true/false on bias-variance tradeoff and prediction
    - CE1 problem 1e — read the bias-variance plot for KNN
  one-liner: course-running theme; E[(y−f̂)²] = σ² + Bias² + Var; prof said *multiple times* "definitely going to be a question"; he resists the "trade-off" framing because regularization can reduce both — see double-descent; derivation expected on the exam
- slug: regularization
  canonical: Regularization
  modules: [05-resample, 06-modelsel, 07-beyondlinear, 09-boosting, 11-nnet]
  lectures: [L11, L12, L13, L14, L15, L16, L20, L23, L24, L26]
  isl-ref: null
  exercises:
    - Exercise6.5 — ridge regression on Credit
    - Exercise6.6 — lasso on Credit
    - Exercise11.1d — how can a 10000-weight NN fit on 1000 obs (regularization)
  one-liner: prof's framing is "the most important variant of model selection"; explicit (ridge L2, lasso L1, elastic net, weight decay, dropout, smoothing-spline λ, tree pruning α) and implicit (mini-batch SGD, bagging, model averaging); central trick of modern ML
- slug: cross-validation
  canonical: Cross-validation
  modules: [04-classif, 05-resample, 06-modelsel, 07-beyondlinear, 08-trees, 09-boosting, 11-nnet]
  lectures: [L09, L10, L11, L12, L13, L14, L15, L16, L18, L19, L20, L23, L24, L26, L27]
  isl-ref: null
  exercises:
    - Exercise5.1 — describe k-fold CV with figure, aggregation, regression and classification examples
    - Exercise5.2 — k-fold vs validation-set vs LOOCV (bias / variance / compute)
    - Exercise5.3 — wrong-vs-right way to do CV with feature pre-selection
    - Exercise6.3b — best-subset selection scored by 10-fold CV
    - Exercise8.2c — cv.tree() for choosing pruning size
    - Exercise8.3e — cv.tree with prune.misclass for classification
    - Exercise9.4a — gbm() with cv.folds
    - Exercise9.4c — xgb.cv on simulated data
    - CE1 problem 4a — write 10-fold CV pseudocode for KNN regression
    - CE1 problem 4b — true/false comparing CV variants
  one-liner: the prof's preferred way to choose any hyperparameter (anti-AIC/BIC because their assumptions don't hold); k=5/10 standard; LOOCV shortcut for OLS; nested CV for selection + assessment; right-vs-wrong way is a flagged trap
- slug: standardization
  canonical: Standardization (z-score)
  modules: [06-modelsel, 08-trees, 10-unsuper, 11-nnet]
  lectures: [L12, L13, L14, L15, L21, L22, L24]
  isl-ref: null
  exercises:
    - Exercise11.3 — preprocess Boston housing with mean/sd before NN
  one-liner: subtract mean, divide by sd per column; **mandatory** before ridge, lasso, PCA, PCR, k-means, hierarchical clustering, KNN, NNs — every method that's not scale-invariant; one-line diagnosis when results look weird
- slug: multivariate-normal
  canonical: Multivariate normal distribution
  modules: [02-statlearn, 03-linreg, 04-classif]
  lectures: [L04, L05, L09]
  isl-ref: null
  exercises:
    - Exercise2.4 — simulate multivariate normal with mvrnorm under different Σ
    - CE1 problem 1g — match contour plot to a given covariance matrix
  one-liner: f(x) = (2π)^{-p/2} |Σ|^{-1/2} exp(−½(x−μ)ᵀΣ⁻¹(x−μ)); foundation for OLS sampling distribution and for LDA/QDA's class-conditionals; ellipsoidal contours; zero covariance ⇒ independence (only under normality)
- slug: double-descent
  canonical: Double descent / benign overfitting
  modules: [02-statlearn, 05-resample, 06-modelsel, 11-nnet]
  lectures: [L04, L11, L13, L24, L26]
  isl-ref: null
  exercises: []
  one-liner: prof's hobbyhorse — past the interpolation point (#params ≈ #samples), test error comes back down; doesn't break bias-variance, just changes its shape; explained as "minimum-norm solution among infinite interpolators"; why he resists the word "tradeoff"
```

## Out of scope (per module)

The atoms below are *not* getting written. Each is something covered in ISLR or in the slides only in passing, that the prof explicitly excluded — mostly via L27's scope rule and the verbatim quotes from earlier lectures. These belong in each module's MOC under `## Out of scope`, sourced to a verbatim prof signal.

### Module 02: Statistical Learning

- **Pseudo-inverse mathematics.** L04 prof: "we'll get to it tomorrow" but he never did the formal derivation; in scope as a *concept* (the minimum-norm interpolator that explains double descent, captured in `double-descent`), out of scope as a derivation.
- **Spectral / eigen-decomposition theory of covariance matrices.** L04 verbatim: "we don't talk about spectral decomposition", deferred to Linear Statistical Models. Eigenvalue = PC variance is captured inside `principal-component-analysis`/`explained-variance-and-scree-plot`; the full spectral theory is out.

### Module 03: Linear Regression

- **F-test mechanics / formulas.** L06 verbatim: "I probably won't ask any questions about an F-test… too boring for this class." The `f-test` atom is conceptual only.
- **Variance Inflation Factor (VIF).** L08: marked self-study. No atom.
- **Moore-Penrose pseudoinverse details.** L08: explicitly bracketed off. No atom.
- **Heteroscedasticity tests / Shapiro-Wilk normality tests.** L08: "we're not going to talk about it." No atom.

### Module 04: Classification

- **Multi-class logistic regression.** L07/L08: "we're not going to talk about… mostly because LDA and KNN can deal with this case." No atom (binary logistic is in scope).
- **GLM link-function theory beyond logit; probit, complementary log-log.** L07: "outside scope of this course… if you took the GLM course." No atom.
- **Imbalanced-class detail / asymmetric ROC analysis.** L07: "I don't think the book talks much about that." Sensitivity/specificity in scope; deeper treatment is not.

### Module 05: Resampling

- **Cp, AIC, BIC, adjusted-R² derivations and formulas.** L12/L13 verbatim: "I'm not going to ask you to use these. I'm not going to ask you to derive them." The `aic-bic-conceptual` atom captures the "they exist, here's why, prof distrusts them" bullet point only.

### Module 06: Model Selection and Regularization

- **Bayesian interpretation of ridge / lasso (Gaussian / Laplace priors).** L14 verbatim: "I really don't think I'd put this on the test, just because it kind of assumes a lot of knowledge that maybe you don't have." No atom.
- **L0 norm / "Optimal Brain Damage."** L14: "we won't go into it because it's not used in practice." No atom.
- **Detailed PLS history and chemometrics-specific tuning.** L14/L15 mentioned only briefly; the PLS atom captures the algorithm and the "no better than ridge but Swedish" verdict, no separate history atom.

### Module 07: Moving Beyond Linearity

- **Natural-spline boundary-knot derivation.** L16 verbatim: "in other courses they go through the math of what these natural splines are. The book doesn't, so I won't either." Linear-extrapolation concept is captured in `regression-splines`.
- **B-spline basis machinery (the `bs` label).** L16: "I don't know why they call it BS." Cosmetic; not on the exam.
- **Bezier / shipbuilding history of splines.** L16: pedagogical context only.

### Module 08: Tree-Based Methods

- **NP-completeness proof / computational complexity of trees.** L17 mentioned in passing only.

### Module 09: Tree Boosting

- **Detailed boosting pseudocode line-by-line.** L20/L27 verbatim: "I won't have you memorize the names of the R functions, of course, but you should know what tree boosting is." The `boosting` and `gradient-boosting` atoms cover concept and high-level algorithm; line-by-line is not on the exam.
- **CatBoost, LightGBM as separate algorithms.** L21: name-checked only, no atoms.

### Module 10: Unsupervised Learning

- **Non-negative matrix factorization (NMF) / "parts" version of eigenfaces.** L21: mentioned only as a contrast; no atom.
- **K-means++ initialization, weighted KNN, Ward linkage formula, gap statistic.** L22: mentioned as alternatives only.
- **Wasserstein, cosine distance derivations.** L22: name-checked only; the *idea* "distance choice matters" is captured in `distance-metrics`, the metric internals aren't.

### Module 11: Neural Networks

- **CNN architecture details (filter math, padding, pooling variants, modern architectures like ResNet/Transformer).** L24/L27: high-level concept only, the `convolutional-neural-network` atom names max-pool and convolution but doesn't go deeper.
- **RNN architecture details (LSTM/GRU gates, BPTT).** L26/L27 verbatim: "deferred and never examined; only the high-level 'hidden state propagates' idea is in scope." Captured in the RNN atom one-liner.
- **Vanishing/exploding gradients, batch normalization, weight initialization (Xavier/He), Adam optimizer internals.** L24: "not discussed in any depth." No atoms.
- **Skip connections / intra-layer connections.** L27: explicitly out.
- **History of NNs (McCulloch & Pitts, Rosenblatt, AI winters, AlexNet).** L22 verbatim: "I'm not going to ask you a history question on the test." High-level "key ingredients = algorithms + compute + data" is fair game; proper-noun history is not.
- **Universal approximation proof.** L23 verbatim: "to actually understand this, you have to go into measure theory and all sorts of hard math that we don't talk about in this class." The atom states the result; proof excluded.
- **Shapley values, explainable AI machinery.** L26: brief mention only.

### Whole-course exclusions

- **Support Vector Machines (SVM, ISLR ch. 9 entirely).** L22 verbatim: "I was going to talk about it, but then we didn't, and it's fine. I don't think it's that interesting." Confirmed in `exam_analysis.md` §5. No SVM atoms anywhere — no hyperplanes, margins, kernels, slack variables, hinge loss, support vectors. The `modules/9SVM` folder is dead.
- **Survival analysis (Kaplan-Meier, Cox PH, censored data).** Not in any lecture or exercise.
- **Multiple-testing corrections (Bonferroni, FDR).** Not in any lecture or exercise.
- **Time-series modeling (ARIMA etc.).** The L26 NYSE example uses RNNs and treats time-series structure conceptually only.
- **R/Python package names, function syntax, executable code.** L27 verbatim: "There's going to be no language… no language-specific coding or anything of that sort." Everywhere.
- **Long proofs / measure theory.** L23: explicitly out of scope.
- **Long history-of-the-field essay questions.** L22/L27: "I'm not going to ask a history question."

## Notes on edge cases

A handful of judgment calls Anders should know about before fan-out launches:

1. **Total atom count: 81** (74 module-owned + 7 Specials). Above the brief's 40–70 estimate; the overage comes from honoring the prof's explicit weight on exercises (every recommended-exercise problem maps to an atom, which fragments module 4 and module 11 in particular). If Anders prefers fewer, the easy consolidation candidates are: `weak-learner-and-learning-rate` ⊆ `gradient-boosting`, `boosting-loss-functions` ⊆ `gradient-boosting`, `dimensionality-reduction` ⊆ `principal-component-analysis` (or kept as a thin pointer), `distance-metrics` ⊆ `clustering` atoms.

2. **Specials count = 6** (bias-variance, regularization, cross-validation, standardization, multivariate-normal, double-descent). Right in the brief's "5–10" range. The two judgment calls inside Specials:
   - `double-descent` kept *separate* from `bias-variance-tradeoff` even though they're conceptually related — the prof returned to it five times with distinctive treatment (L04 simulation, L11 bagging connection, L13 callback, L24/L26 "this is why ML works"). It's the canonical "explain double descent" exam-style question, distinct from "derive bias-variance."
   - `multivariate-normal` kept as a Special because it's load-bearing for both module 3 (β̂ sampling distribution) and module 4 (LDA/QDA class-conditionals) and is exam-flagged in CE1.1g (match contour to Σ). If demoted, it'd live in module 2.
   - `maximum-likelihood` was *considered* as a Special but folded into `least-squares-and-mle` (module 3) since the only exam-relevant cross-cutting fact is "MLE = LS under Gaussian errors," which the prof gave as the L27 mathy-question template. The other lectures that touch MLE (L07, L23) refer back to that derivation.

3. **`logistic-regression` carries a heavy exercise load (8+ exercise refs across modules 4 + CE1).** It's the biggest atom in module 4 by exercise count, lecture count, and exam relevance (Q3 + Q7 of CE1, plus L27 walkthrough). The atom *will* push toward the upper end of the 80–250 line bound — possibly even split-worthy if it grows past 250.

4. **Module 12 has zero owned atoms.** This is intentional — it's the exam-review session, all real content lives in earlier modules. The MOC for module 12 will route to L27, the scope-rule callout, and the Specials atoms. Mentioned here so the fan-out agent for module 12 doesn't go searching for missing concepts.

5. **`elastic-net`, `naive-bayes`, `partial-dependence-plots`, `step-functions` have no exercise instances.** Kept as thin atoms because each has explicit lecture treatment and is named in `exam_analysis.md`. Likely-thin atoms (50–80 lines).

6. **Backprop derivation gets its own atom** even though no exercise / exam problem asks you to derive it. The lectures (L23, L24) spend significant time on the chain-rule mechanics and the prof emphasized "if you have loops you're screwed" → important conceptual content for understanding why feedforward / CNNs work but RNNs are harder.

7. **CE2 (compulsory exercise 2) is an open project, no fixed problem-to-atom mapping.** I haven't tagged any atom with `CE2` references — the project is "use methods from at least two different modules" so any atom could in principle be invoked. The fan-out agents should not feel they need to add CE2 references; CE1 references are precise and well-mapped.

8. **No atom for "interpretability" as a standalone concept.** Treated within `partial-dependence-plots`, `variable-importance`, the `convolutional-neural-network` discussion, and the L26 "when to use deep learning?" lecture. If a downstream user keeps asking "what does the prof say about interpretability?" we can split it out, but for now it's distributed.

9. **Several module-3 atoms were merged from a more granular initial draft.** Specifically: `simple-linear-regression` + `multiple-linear-regression` + a separate `least-squares` were folded into `linear-regression` + `least-squares-and-mle`; `confidence-interval` + `prediction-interval` merged; `r-squared` + `adjusted-r-squared` merged; `hat-matrix` + `design-matrix` + `normal-equations` merged into `design-matrix-and-hat-matrix`; `residual-diagnostics` + `qq-plot` + `studentized-residuals` + `leverage` merged into `residual-diagnostics`; `categorical-encoding` + `reference-category` + `interactions` merged. Done because these are sub-concepts of the same exam question, not separately askable. If Anders wants finer atoms anywhere, the merges are easy to undo.

10. **Subset selection (best, forward, backward, hybrid) is one atom.** Same logic, as Anders won't ask "explain forward stepwise" without also asking about best-subset, and the prof treats them as one slide deck. The four sub-algorithms become four sections inside one atom.
