# TMA4268 Exam Analysis: May 18, 2026

Synthesized from 27 lecture transcripts (Jan 5 – Apr 28, 2026). Two subagent passes: first wave (7 thematic agents) covered explicit exam mentions; second wave (13 agents) deep-read the silent middle of the semester (Jan 19 – Mar 17) for implicit signals.

The single most important meta-fact: **this year's exam is fundamentally different from previous years.** The instructor (new this year) has redesigned it. Old exams are useful study material *only* if you re-interpret the coding tasks as interpretation tasks. Read this whole document before going back to past papers.

---

## 1. Logistics: what, when, where

| Item | Value |
|---|---|
| Date / time | **May 18, 2026, 09:00** |
| Duration | **4 hours** (instructor noted he'd consider shorter; assume 4h) |
| Format | **Open book**, digital (Inspera) |
| Answers written | **On paper**, then scanned (NOT typed into Inspera buttons) |
| Prereq | Pass both compulsory exercises with ≥60% |
| Retake | Possibly **oral** |

### What you can bring
- **The textbook (ISLP)**: physical or as the searchable PDF supplied in Inspera. Post-it tabs OK; no writing in the book.
- **One A5 sheet, handwritten notes only**: the instructor was uncertain whether iPad-printed counts; play safe and handwrite.
- **A calculator**: any normal one. "Won't need anything fancy."
- **The textbook PDF inside Inspera** is searchable.

### What you cannot do
- No computer, no R/Python, no internet.
- You will not write executable code.

---

## 2. Programming: R or Python?

**Both are fine for the course. Neither is required on the exam.**

- Jan 13: *"Python will also be okay. I'm going to make sure that the exam questions are not overly R-specific."*
- Apr 28 (definitive): *"There's going to be no language. You don't have to memorize the different R packages, the names of different R packages or Python packages. There's no language-specific coding or anything of that sort."*

### What this means concretely
- **No memorizing function names or package names** (no `glm()`, `gbm()`, `glmnet`, etc. required). Common functions appear *in* exam stimuli (output tables) but you don't have to recall them.
- **Code questions become interpretation questions.** Instead of "fit a model and compute MSE," you'll be shown the fitted model's output and asked to interpret/compute from it.
- **Pseudocode is acceptable.** If a question asks "how would you compute test MSE," write it in math, English, or pseudocode, anything that conveys the idea.

### Course-level (not exam) policy
- Compulsory exercises default to R Markdown (.Rmd + PDF). Python submissions appear to be tolerated if you ask, but R is the path of least resistance for the *coursework*.

---

## 3. Exam format: question types

Drawn from Apr 28, where the instructor walked through old papers and showed how he'd rephrase them.

1. **Multiple choice / fill-in-the-blank**: concept checks. Single letter answer is fine.
2. **True/False with optional explanation**: write just T/F, OR explain your reasoning for partial credit if uncertain.
3. **Output interpretation**: given a regression / GLM / cross-validation / ROC / dendrogram / scree plot output, explain what it says (coefficients, significance, interactions, AUC, optimal λ, etc.).
4. **Hand calculations**: odds ↔ probability, degrees of freedom counting, MSE from a small table, PCA variance explained, hierarchical clustering distances by hand.
5. **Pseudocode / equation-writing**: "write out how you'd compute X." Math notation, plain English, or pseudocode all OK.
6. **At least one mathy theory question**: *"something mathy but not incredible, no weird spaces or fancy proofs."* The bias-variance derivation is the standout candidate.
7. **Method comparison**: given two models' results, which is better and why? Tie answer to bias-variance / overfitting / interpretability.

### Grading mechanics
- **Show your work for partial credit.** Empty is worse than wrong-with-reasoning.
- **No negative marking.** Always answer.
- **If a question feels broken or trick**: write a short note explaining your interpretation; you'll likely get the points.
- Percentile-based letter grades; instructor said he gives a small upward nudge but never adjusts down.

---

## 4. What's IN scope

The rule from Apr 28: **"If it was covered in the slides or the exercises, it's fair game. If it's only in the book and we didn't talk about it in class or exercises, it won't be on the test."**

### Tier 1: guaranteed / very heavy (study first)

- **Bias-variance tradeoff.** Explicit, repeated guarantee. From Apr 28: *"Definitely going to be a question about bias variance in the test because I think that concept is the kind of the running theme through the course in all the models we looked at."* And **separately confirmed Feb 24** mid-semester: *"this is definitely going to be on the exam. I mean, just this concept... it's deceptively confusing. It seems really simple. It's not."* Read the bias-variance section of ISLP cover-to-cover. Be able to:
  - State and derive the decomposition E[(y − f̂)²] = Bias² + Var + σ².
  - Explain why reducing variance is critical.
  - Connect it to *every* method you know (regression, ridge/lasso, trees, NN, splines).
  - Discuss the over-parameterized / **double-descent / benign-overfitting** regime, the instructor's favorite recent topic.

- **Linear regression interpretation.**
  - Read coefficient tables (estimate, SE, t, p).
  - **Interactions** (sex × age etc.): easy to mess up; instructor flagged this as a common error.
  - Count degrees of freedom from the model.
  - Why train error drops with parameters but test error doesn't.
  - Collinearity: the *concept*, not derivations.

- **Logistic regression / classification.**
  - Odds and log-odds; convert probability ↔ odds with calculator.
  - Coefficient interpretation in odds terms.
  - Confusion matrices, sensitivity, specificity, ROC, AUC.
  - Class imbalance effects.
  - LDA / QDA / KNN comparison; **decision boundary derivation** flagged as "another good exam question." Where does the quadratic in QDA come from?

- **Regularization (Ridge & Lasso).**
  - Effect of λ on flexibility / bias / variance.
  - L1 (sparsity) vs L2 (shrinkage).
  - Why regularization can *increase* train error but reduce test error.

- **Cross-validation.**
  - K-fold mechanics (be able to write pseudocode).
  - Why we trust CV over AIC/BIC/Cp (instructor's bias: he doesn't trust theoretical penalties).
  - Reading a CV-error-vs-λ plot to pick a hyperparameter.

### Tier 2: solid / moderate-heavy

- **Trees, bagging, random forests, boosting.** Explicit: *"You should know what tree boosting is."*
  - Hyperparameters: terminal node size, depth, n_trees, learning rate (η).
  - Why weak learners + many trees works (gradual residual fitting).
  - Concept of fitting trees to residuals.
  - You don't need to memorize the boosting pseudocode line-by-line, but you need the concept.

- **Splines / GAMs / non-linear.**
  - Degrees of freedom = knots + degree.
  - Smoothing parameter λ: bigger λ → smoother, *less* wiggly (be careful with the direction, easy T/F trap).
  - B-spline basis idea.

- **Feed-forward neural networks.**
  - Architecture (input/hidden/output), parameter count (weights + biases).
  - Activation functions (ReLU now standard, sigmoid classical, GELU emerging). Why nonlinearity matters: without it, you collapse to linear regression.
  - Backprop = chain rule applied efficiently.
  - Mini-batch SGD: noise has implicit regularization. Batch sizes are powers of 2 (32/128/256/512).
  - **Regularization is mandatory:** L1, L2, dropout, early stopping, data augmentation, label smoothing. Never train without one.
  - Universal approximation: stated, not proved (measure theory is out of scope).

- **PCA.** Loadings, explained variance, scree plot, "how many components for 90%" calculations.

### Tier 3: know the concepts, light on math

- **K-means** (initialization sensitivity, choosing k).
- **Hierarchical clustering**: single/complete/average linkage; **build a dendrogram by hand from a small distance matrix.**
- **Module 2 framework**: supervised vs unsupervised, prediction vs inference, parametric vs nonparametric.

### Tier 4: light coverage; don't over-invest

- **CNNs, RNNs**: context only. Know the high-level idea (weight sharing in CNNs, hidden-state propagation in RNNs, sequential data motivation). Architecture details are NOT examined.
- **SVM**: minimal lecture time. Know it exists, vague concept.
- **Naive Bayes**: barely covered. Variant of generative classification.

---

## 4b. Direction-of-effect cheat sheet (T/F traps)

These are the "increase X → Y goes which way?" rules the instructor stated explicitly. Each is a plausible true-false question. Memorize the *direction*, then explain *why* in one sentence.

| Lever | Direction of effect | Source |
|---|---|---|
| Smoothing-spline λ ↑ | model becomes **smoother / less wiggly** (λ=0 → wiggly fit; λ=∞ → straight line) | Mar 9 |
| Ridge λ ↑ | coefficients shrink toward zero, but **never exactly zero** (λ=0 → OLS; λ=∞ → all zero) | Feb 23, Feb 24 |
| Lasso λ ↑ | coefficients shrink and **hit exactly zero** at corners of L1 ball (sparsity) | Feb 24 |
| KNN k ↑ | from k=1 (jagged, overfits) toward k=N (oversmoothed, underfits) | Jan 27 |
| Boosting learning rate η ↓ | need **more trees**; usually generalizes better | Mar 17, Apr 13 |
| Random forest mtry ↓ | trees become **more decorrelated**, variance reduction kicks in | Mar 17 |
| Polynomial / spline degree ↑ | training error drops; test error U-shapes (then double-descents at very high p) | Jan 19 |
| Adding predictors to OLS | training fit always improves; test fit may worsen (overfit) | Mar 2 |
| PCA without standardization | result dominated by largest-scale variable | Mar 2, Mar 3 |
| Mini-batch SGD batch size ↑ | less noise → less implicit L2 regularization | Apr 21 |
| GAM with no interaction terms | f(x₁) + f(x₂) is additive, **cannot capture x₁·x₂ interactions** | Mar 9 |
| Logistic regression coef β | "increase x by 1" → **odds × e^β** (NOT probability × e^β) | Jan 27 |
| Tree pruning α ↑ | tree gets **smaller** (cost-complexity loss penalizes large \|T\|) | Mar 16 |
| Adding more predictors to OLS | training R² always **goes up**; adjusted R² may go down | Jan 26 |
| LOOCV vs k-fold | LOOCV: **low bias, high variance**; k=5/10: balanced (lower variance, slight bias) | Feb 9, Feb 10 |
| Sigmoid → ReLU substitution | "trivial change" with huge effect, solved vanishing gradients | Apr 20 |
| LDA covariance assumption | shared Σ → **linear** boundary; class-specific Σ_k → **quadratic** | Feb 3 |
| KNN distance with mixed units | Euclidean becomes meaningless without scaling | Jan 27 |
| Hierarchical linkage choice | single → chains; complete → balanced; average → middle ground | Apr 14 |
| NN learning rate η | η = 2 → "horrible, bouncing everywhere"; η ≈ 0.1 → fine | Apr 20 |
| Dropout rate | **20% common, 50% never** (too aggressive) | Apr 21 |
| Heteroscedasticity (residual cone) | SE estimates become biased; CIs unreliable | Feb 2 |

**Easy traps in this list:**
- Ridge does *not* zero coefficients. Lasso does. Geometric reason: L2 ball has no corners; L1 has corners on axes.
- Smoothing-spline λ goes the *opposite* direction from polynomial degree: more λ = *less* flexible.
- Logistic coefficients act on **odds**, not probability. A coefficient of 0.05 with x increasing by 100 means odds × e^5 ≈ 148, which is huge.

---

## 4c. Formulas and conventions worth memorizing

Brief, exam-bait quantities the lecturer wrote on the board.

- **Cubic spline parameter count:** K knots + degree d + intercept ⇒ **K + d + 1** parameters (cubic d=3).
- **Effective DOF for smoothing spline:** **trace(S)** where S is the smoother matrix, which is *not* an integer in general.
- **Subset selection complexity:** best-subset = **2^P** models; forward/backward stepwise = **1 + P(P+1)/2**.
- **Random forest mtry defaults:** **√p** for classification, **p/3** for regression.
- **OOB coverage:** roughly **1/3** of observations are out-of-bag for each bootstrap tree. Used as a free hold-out set.
- **PCA constraint:** loadings normalized so **Σ φ²ᵢ = 1**.
- **PCA variance share:** variance of the i-th principal component = **eigenvalue λᵢ** of the centered data matrix's covariance.
- **PCA variance cutoff conventions:** retain enough PCs to hit **90% / 95% / 99%** of cumulative variance.
- **Cost-complexity pruning:** minimize **RSS + α|T|** where |T| is number of terminal nodes; choose α by k-fold CV.
- **Gini / entropy for tree training; misclassification error for pruning**: different criteria at different stages.
- **Bagging variance under correlation ρ:** variance of average ≈ **ρσ² + (1−ρ)σ²/B**, which does *not* shrink to 0 when ρ > 0. This is *the* reason RF decorrelates trees via mtry.
- **One-standard-error rule:** pick the simplest model whose CV error is within 1 SE of the minimum CV error.
- **OLS closed form:** β̂ = (X'X)⁻¹X'y. Hat matrix **H = X(X'X)⁻¹X'**; leverage h_ii is its diagonal.
- **R²** = 1 − RSS/TSS. **Adjusted R²** = 1 − (RSS/(n−d−1)) / (TSS/(n−1)), which penalizes adding parameters.
- **t-statistic for coefficient:** t = β̂ⱼ / SE(β̂ⱼ), df = n − p − 1.
- **CI for β:** β̂ ± t_{α/2,n−p−1} · SE(β̂). **Prediction interval is wider than confidence interval** (one captures E[Y|X], the other captures a future Y).
- **Var(β̂) = σ²(X'X)⁻¹**: diagonal gives SE² of each coefficient. Collinearity → off-diagonal blows up → inflated SE.
- **Sigmoid:** σ(z) = 1/(1+e⁻ᶻ). **Softmax:** exp(zₖ)/Σⱼ exp(zⱼ).
- **Logistic log-odds:** log(p/(1−p)) = β₀ + β₁x. "Increase x by 1 → odds multiplied by **e^β**."
- **Bayes theorem (classification):** P(Y=k|X) = π_k · f_k(X) / Σ_l π_l · f_l(X).
- **Prior estimate:** π̂_k = n_k / n.
- **LDA discriminant:** δ_k(x) = log π_k + x'Σ⁻¹μ_k − ½ μ_k'Σ⁻¹μ_k. **Linear** in x because the x'Σ⁻¹x term cancels across classes.
- **QDA discriminant:** add ½x'Σ_k⁻¹x − ½ log|Σ_k|. **Quadratic** in x because Σ_k⁻¹ is class-specific so x'Σ_k⁻¹x no longer cancels. *That's where the quadratic comes from.*
- **Pooled covariance (LDA):** Σ̂ = Σ_k [(n_k−1)/(n−K)] · S_k.
- **Bias-variance decomposition:** E[(y − f̂(x))²] = **Bias²[f̂(x)] + Var[f̂(x)] + σ²** (irreducible).
- **Bootstrap conventions:** B = **1,000–10,000**. Sample with replacement, same size n.
- **OOB exact:** P(observation excluded from a bootstrap sample) → (1−1/n)ⁿ → **1/e ≈ 0.368** as n→∞.
- **k-fold CV conventions:** k = **5 or 10** standard. LOOCV = k=n.
- **LOOCV shortcut for linear regression:** CV = (1/n) Σ ((yᵢ − ŷᵢ)/(1−h_ii))², no refit needed.
- **Misclassification metrics:** Sensitivity = TP/(TP+FN); Specificity = TN/(TN+FP); Accuracy = (TP+TN)/n; Error = 1 − Accuracy. AUC = 0.5 chance, 1 perfect.
- **AdaBoost weight update:** α_m = ½ log((1−err_m)/err_m); w_i ← w_i · exp(−α_m · y_i · G_m(x_i)).
- **Gradient boosting:** F̂ ← F̂ + η · tree(residuals). **η ≈ 0.01–0.1** typical.
- **NN parameter count (one hidden layer, p inputs, M hidden units, C outputs):** **M(p + 1) + C(M + 1)**.
- **NN dropout rate:** **20% common, never 50%**. Batch sizes powers of 2 (32 / 128 / 256 / 512). Learning rate η ≤ 0.1.
- **K-means objective:** minimize Σ_k Σ_{i∈Cₖ} ‖xᵢ − x̄ₖ‖².
- **Standardization (z-score):** zᵢⱼ = (xᵢⱼ − x̄ⱼ) / σⱼ. Required before PCA, Ridge, Lasso, K-means, hierarchical clustering, KNN.
- **Categorical encoding:** K levels need **K−1 dummies**. The omitted level is the reference; intercept estimates the reference outcome.
- **Hierarchical principle for interactions:** if you include x₁·x₂, you also include x₁ and x₂ as main effects.

---

## 4d. Worked-example datasets the lecturer used (likely templates)

Past exams reuse the lecturer's running datasets with the data swapped. These are the ones he kept coming back to:

| Dataset | Method context | What you'd be asked to interpret |
|---|---|---|
| **Default / credit-card** | Logistic regression | Coefficient → odds, balance vs income effect |
| **South African heart disease** | Logistic regression, GLM | Odds, multiple-predictor interpretation |
| **Boston Housing** | Random forest, mtry comparison, regression trees | Variable importance, MSE comparison across mtry |
| **Wage data** | GAMs, splines, polynomial | Smoothing parameter effects, additive interpretation |
| **Ozone** | Regression trees | Variable selection (which predictors get split on) |
| **Brain injury** | Bagging | Surprise: bagging didn't improve error (0.18 → 0.19), a nuance question |
| **Body fat / BMI** | Simple linear regression | Practical vs statistical significance |
| **Iris (Fisher)** | LDA / QDA | Discriminant scores, error comparison |
| **Cork tree directions (N/E/S/W)** | Multivariate normal, contrasts | Covariance matrix interpretation |
| **Hitters (baseball)** | Subset selection, regression trees | Cp/AIC/BIC paths, tree pruning |
| **Smarket / stock** | Logistic regression | Probability prediction with threshold |
| **MNIST** | Neural networks, CNNs | Architecture, parameter counting |

Format expectation: you get a printout from one of these (or a structurally identical dataset), and you interpret coefficients, residual plots, ROC, or feature importance.

---

## 4e. Instructor's opinionated takes (likely conceptual-question fodder)

When a lecturer has strong opinions, those opinions become exam questions.

- **Distrusts AIC / BIC / Cp.** Feb 23: *"I don't trust them... they're making some assumption that probably won't hold."* Always prefers cross-validation. Be ready to defend CV over information criteria.
- **Critical of "trade-off" framing for bias-variance.** A fair exam question: *"why is the lecturer skeptical of the word trade-off?"* Answer hinges on benign-overfitting / double-descent regimes where you can drive *both* down.
- **Loves benign overfitting / double descent.** Spent unusual time on it Jan 19, returned Apr 21. *"Benign being like, you know, it won't hurt you."* Likely conceptual question on *when* overfitting hurts vs. helps.
- **Prefers permutation-based variable importance over Gini.** Mar 17: *"I think I would generally go with randomization one because it makes more sense."* Both valid; know the difference.
- **Frames regularization as the central trick.** Feb 23: *"regularization... it really goes at the core of what a lot of statistical learning, machine learning is."* Expect at least one question about it conceptually, not just mechanically.
- **Algorithmic vs data-model philosophy (Breiman).** Mar 10: positions trees as algorithmic / prediction-first vs the parametric / inference tradition. Possible essay-style question.
- **Stresses independence violations as a common mistake.** Feb 9: long anecdote about declining authorship on his mother's paper for this reason. *"Violating these is super common and ruins everything."* Plausible exam scenario: "you have time-series data, what breaks?"

---

## 4f. Common-mistake exam traps the lecturer flagged

These got explicit "students always get this wrong" energy:

1. **Interaction terms in logistic regression.** Saying "males weigh more on average" without accounting for `sex × age` is the canonical wrong answer (Apr 28).
2. **Independence assumption with time/spatial data.** LOOCV/CV is a "terrible idea" with autocorrelation (Feb 9). Standard errors become "horseshit" (Jan 20).
3. **Data reuse across model selection and assessment.** Train, *validation*, and test sets are three different things (Feb 9).
4. **Forgetting to standardize before Ridge or PCA.** Both are scale-sensitive (Feb 23, Mar 2).
5. **Misreading high leverage as high influence.** Leverage = position in X-space; influence = leverage × residual. *"Fat kid on the seesaw"* analogy (Feb 2).
6. **Stopping tree growth early instead of growing fully and pruning.** Bad early splits enable good later splits. Grow then prune (Mar 16).
7. **Treating regression coefficients as causal.** "Fancy correlations, not causal" (Jan 20).
8. **Confusing statistical significance with effect size.** Large N inflates p-value-driven significance even for trivial effects (Jan 20).
9. **Training a neural network without regularization.** Apr 21: *"recipe for disaster."* L1/L2/dropout/early stopping: pick at least one.
10. **Wrong-way cross-validation: feature selection then CV on the same data.** Feb 10: *"You're using the same data to pick your variables and then seeing how well they work. So of course it's going to work well."* Selection must happen *inside* each fold (nested CV).
11. **Heart disease coefficient sign error.** If you flip the reference level of a binary predictor, every coefficient sign flips with it. Apr 28 confirmed this is interpretation-trap fodder.
12. **K-means random init.** Running once is wrong, as the algorithm converges to local minima. Run multiple times, pick lowest within-cluster SS. Apr 14: *"random initialization can really screw you."*

---

## 4g. Procedural templates the lecturer explicitly flagged as exam-likely

Three procedures the instructor walked through and said, in some form, *"this is the kind of question I'd ask."* Each is a step-by-step recipe. Practice each on small numbers until automatic.

### G1: Hierarchical clustering by hand from a distance matrix (Apr 28: "you need to be able to do this")

**Setup:** Given an n×n dissimilarity matrix and a linkage method (single / complete / average), build the dendrogram.

**Algorithm:**
1. Each observation starts as its own cluster.
2. Find the smallest off-diagonal entry → merge those two clusters at that height.
3. Recompute distances from the new cluster to all remaining clusters using the linkage rule:
   - **Single:** d(C_i ∪ C_j, C_k) = min(d(C_i, C_k), d(C_j, C_k))
   - **Complete:** d(C_i ∪ C_j, C_k) = max(d(C_i, C_k), d(C_j, C_k))
   - **Average:** weighted mean of all pairwise distances
4. Repeat until one cluster remains.
5. Draw dendrogram with merge heights on the y-axis.

**Apr 28 explicit grading note:** *"−1 point for each mistake."* The instructor walked through a 4×4 example. Practice with at least 4×4 and 5×5 matrices before exam.

### G2: LDA decision boundary derivation (Feb 3: "this would be a typical exam question")

**Setup:** Two classes A, B with means μ_A, μ_B, shared covariance Σ, priors π_A, π_B.

**Procedure:**
1. Write out δ_A(x) = log π_A + x'Σ⁻¹μ_A − ½ μ_A'Σ⁻¹μ_A.
2. Write out δ_B(x) similarly.
3. Set δ_A(x) = δ_B(x) and solve. The x'Σ⁻¹x term cancels (LDA's hallmark) ⇒ result is **linear in x**.
4. Worked example from Feb 3: μ_A = (1,1), μ_B = (3,3), Σ = 2I → boundary equation **x₁ + x₂ = 4**.

**For QDA:** the x'Σ_k⁻¹x term does **not** cancel (Σ_k differs between classes), so the boundary is quadratic. The instructor explicitly flagged: *"where does the quadratic come from in QDA?"* The answer hinges on this cancellation failure.

### G3: Logistic-coefficient odds interpretation

**Setup:** Given a fitted logistic model (Default ~ balance + income, etc.), read β values from output table.

**Procedure:**
1. Identify which class is coded as 1 (the reference is the *other* one). If the reference flips, all β signs flip.
2. For each predictor: increase x_j by Δ → odds multiply by **e^(β_j · Δ)**.
3. Be careful with units. β = 0.005 with Δ = 100 (e.g. balance) → odds × e^0.5 ≈ 1.65. β = 0.0001 with Δ = 10,000 (e.g. income) → odds × e^1 ≈ 2.7.
4. Convert odds → probability if asked: p = odds / (1 + odds).

**Apr 28 explicit example:** student with $2,000 balance, $40,000 income → plug into the linear predictor, apply sigmoid, threshold at 0.5 → predict default.

### G4: Hand-counting NN parameters

**Setup:** Architecture stated as p inputs → M hidden units → C outputs, one hidden layer, fully connected, with biases.

**Procedure:**
1. Input → hidden weights: p × M; hidden biases: M.
2. Hidden → output weights: M × C; output biases: C.
3. **Total = M(p+1) + C(M+1) = pM + M + MC + C.**

Quick sanity check: forgetting biases is the most common error. The instructor specifically reminds you of "the bias term."

### G5: Bootstrap standard error / CI

**Procedure:**
1. Resample with replacement, same size n.
2. Compute statistic θ̂_b on each resample. Repeat B = 1,000–10,000 times.
3. **Bootstrap SE** = sample SD across {θ̂_1, …, θ̂_B}.
4. **Percentile CI:** the 2.5% and 97.5% quantiles of the bootstrap distribution.

### G6: k-fold CV error and the one-SE rule

**Procedure:**
1. Split data into k folds (k=5 or 10).
2. For each fold, train on k−1, test on held-out, get CV_i.
3. Mean CV error = (1/k) Σ CV_i. SE = sample SD / √k.
4. Pick model with **lowest** CV error → call it m*.
5. **One-SE rule:** among all models with CV error ≤ CV(m*) + SE, pick the *simplest* one. This is the lecturer's preferred selection criterion.

---

## 5. What's OUT of scope (don't waste time)

The instructor went out of his way to exclude these. Quotes are direct.

- **R/Python package names, function syntax, executable code.** *"I won't do that."*
- **F-tests.** Jan 26: *"I probably won't ask any questions about an F-test."*
- **Penalty derivations** (AIC, BIC, Mallows' Cp). Feb 23–24: *"Their derivations are beyond the scope... they all kind of suck in different ways."* Know that they exist and what they're for; don't memorize formulas.
- **Heavy proofs / measure theory.** Apr 20: explicitly out of scope, e.g. for the universal-approximation proof.
- **Neural net details:** skip connections, intra-layer connections, advanced optimizers (momentum, Adam internals), full RNN/CNN architectures. Apr 28: *"You just wouldn't need to know that."*
- **History questions** (who invented backprop, year of X). Apr 14: *"I'm not going to ask you a history question."*
- **Anything in the book NOT covered in lectures or exercises.**
- **Long essays.** "Be concise."
- **Bayesian interpretation of Ridge/Lasso** (Gaussian / Laplace priors). Mar 2: *"I really don't think I'd put this on the test just because it kind of assumes a lot of knowledge that maybe you don't have."* Don't memorize the prior derivation.
- **Multi-class logistic regression.** Jan 27: *"we're not going to talk about... mostly because discriminant analysis and K-nearest neighbors can deal with this case."* Binary logistic only.
- **Variance Inflation Factor (VIF).** Feb 2: marked as self-study; not exam focus.
- **Formal hypothesis tests for normality** (Shapiro-Wilk etc). Feb 2: *"we're not going to talk about it."*
- **Spectral / eigen decomposition** of covariance matrices. Jan 19: *"we don't talk about spectral decomposition"*, deferred to Linear Statistical Models.
- **Imbalanced-class detail / asymmetric ROC analysis.** Jan 27: *"I don't think the book talks much about that."* Know basic sensitivity/specificity; don't go deep.
- **PLS history (Wold) and detailed mechanics.** Mar 3: PLS mentioned only briefly; PCR is the workhorse.
- **Natural-spline basis math.** Mar 9: *"the book doesn't [derive] so I won't either."*
- **Bezier curves / CAD heritage of splines.** Mar 9: pedagogical context only.
- **Moore-Penrose pseudoinverse details.** Feb 2: explicitly bracketed off.
- **Detailed boosting pseudocode line-by-line.** Apr 28: concept matters; pseudocode memorization does not.
- **Stochastic gradient boosting / XGBoost / LightGBM internals.** Mar 17: mentioned as Kaggle winners, not derived.
- **Computational complexity of trees.** Mar 10: noted as NP-hard but not exam-relevant.
- **Elastic Net detailed tuning.** Feb 24: combines L1+L2, concept noted, no worked example.
- **SVM: confirmed fully out, not just light.** Apr 14 explicit: *"It's another algorithm that we, yeah, I was going to talk about it, but then we didn't, and it's fine. I don't think it's that interesting."* No exam questions on hyperplanes, margins, kernels, slack variables, hinge loss, support vectors. Demote SVM from "Tier 4: light" in your study plan to **skip entirely**.
- **Survival analysis** (Kaplan-Meier, Cox PH), **multiple-testing corrections** (Bonferroni, FDR), **censored data**, **time series modeling**: none of these (ISLP Ch. 11, 13) appear in lectures.
- **Weighted KNN, K-means++ initialization, Ward linkage formula, gap statistic.** Mentioned as existing alternatives at most; not derived or examined.
- **F-statistic for joint hypothesis tests.** Confirmed Jan 26 quote: *"I'm more interested in things that can generalize to other distributions."* Don't memorize the F formula.
- **Vanishing/exploding gradients, batch normalization, weight initialization schemes (Xavier/He), Adam internals.** Apr 20–21: not discussed in any depth.
- **RNN architecture details** (LSTM/GRU gates, BPTT). Apr 28: deferred and never examined; only the high-level "hidden state propagates" idea is in scope.
- **Probit, complementary log-log, other GLM link functions.** Jan 27: *"That's probely outside the scope of this course... If you took the GLM course..."*
- **Detailed CNN filter math, pooling variants, modern architectures (ResNet, Transformer).** Apr 21: high-level concept only.

---

## 6. Comparison with previous exams (2023, 2024, 2025)

Past papers exist on disk:

- `/context/exams/TMA4268_2023_Exam.Rmd`
- `/context/exams/TMA4268_2024_Exam.Rmd`
- `/context/exams/TMA4268_2025_Exam.Rmd`

The instructor walked through 2024/2025 questions in the Apr 28 lecture and explicitly described how he'd modify them for 2026.

### Past exams are useful, with translation

| Past exam style | 2026 translation |
|---|---|
| "Fit `lm(...)` then compute MSE" | "Here's the regression output table: interpret coefficients / compute MSE from these residuals" |
| "Memorize the `gbm()` arguments" | "Explain conceptually how the boosting hyperparameters affect bias/variance" |
| "Write R code for k-fold CV" | "Write pseudocode (or math) for k-fold CV" |
| 4-hour grind designed to "run out the clock" | Open book, more interpretation-heavy, conceptual MC mixed in |

### What's actually changed for 2026
1. **Open book** (new this year, first time per instructor).
2. **No code writing, no package memorization** (was a real burden in past papers).
3. **More multiple-choice / true-false** added vs older formats.
4. **More interpretation-heavy.** Output tables given, you analyze them.
5. **Neural networks** taught with a different style than previous instructors used (instructor said he found the prior approach "boring", so don't trust 2023/2024 NN questions as a perfect format guide).
6. **Curriculum largely unchanged** otherwise, modules 1–12 are the same skeleton.

### How to use past exams
- Do every conceptual question (interpretation, true/false, derivations) at full effort.
- For coding questions: rewrite them in your head as "given this output, interpret it", that's the 2026 version.
- Pay attention to bias-variance, regularization, classification metrics, PCA, as these recur every year.
- Don't memorize the R syntax in the answer keys.

---

## 7. Concrete study plan (prioritized)

If you only had one week, do them in this order:

1. **Bias-variance section of ISLP** + every example in lectures Jan 19, Apr 13, Apr 21. Be able to state the decomposition, explain double descent, and apply it to any model.
2. **Re-do every compulsory-exercise problem.** Instructor: *"If you've solved all the exercises, you're generally pretty good."*
3. **Practice output interpretation**: pull every regression / GLM / boosting / CV / ROC printout from the slides and write a 2-sentence interpretation. Pay attention to interaction-term traps in logistic regression.
4. **Hand calculations drill**: odds↔probability, degrees of freedom, MSE from confusion matrix, PCA variance from eigenvalues, hierarchical clustering by hand on a 4×4 distance matrix.
5. **Pseudocode the algorithms**: k-fold CV, k-means, hierarchical agglomerative clustering, gradient descent, boosting (one round), forward selection.
6. **Past exams (2023, 2024, 2025)**: work them under the 2026 translation rules above.
7. **Build your A5 cheat sheet** as you study. Must-haves:
   - Bias-variance decomposition formula
   - Ridge / Lasso objectives, and the L2-vs-L1 geometry diagram
   - Sigmoid / softmax / log-odds: P = e^η / (1+e^η), η = log(P/(1−P))
   - Sensitivity, specificity, FPR, TPR definitions; ROC axes
   - Cubic-spline parameter count: K + d + 1
   - Effective DOF = trace(S)
   - Bagging variance under correlation: ρσ² + (1−ρ)σ²/B
   - mtry defaults (√p, p/3); OOB ≈ 1/3
   - PCA: standardize first; eigenvalue = PC variance; cutoff at 90/95%
   - Cost-complexity pruning: RSS + α|T|
   - Direction-of-effect cheat sheet (§4b)
8. **End-of-module review questions** in the slides, where instructor draws inspiration from these for exam questions.
9. **Drill the worked-example datasets** (§4d). Pull the slides for Default, Boston Housing, Wage, Ozone, Brain Injury, and re-do every interpretation prompt.

### Things to skip or skim
- Long proofs in ISLP.
- AIC/BIC/Cp algebra.
- F-test mechanics.
- RNN/CNN architectures past the high-level idea.
- SVM duality.
- R package documentation.

---

## 8. Quick exam-day checklist

- Bring: textbook (or rely on PDF), A5 handwritten sheet, calculator, pens.
- Strategy: open with multiple choice / fill-in to bank easy points, then interpretation, then the mathy derivation last.
- Always show work, partial credit is on offer.
- If a question seems broken: state your interpretation in one line, then answer.
- Watch the spline λ direction (more λ = smoother, less flexible) and logistic interaction interpretations, both are easy to flip.
- Bias-variance is showing up. Don't skip it. Read that ISLP section twice.

---

*Sources: 27 lecture transcripts (Jan 5 – Apr 28, 2026). Three waves of subagent analysis. **Wave 1** (7 thematic agents) covered explicit "exam"-keyword mentions (concentrated in Jan 5 + Apr 28). **Wave 2** (13 agents, one per under-mined lecture) deep-read the silent middle of the semester, Jan 19, 20, 27; Feb 2, 9, 23, 24; Mar 2, 3, 9, 10, 16, 17, for implicit signals. **Wave 3** (12 agents, one per ISLP-canonical topic cluster) searched for standard textbook content the keyword passes might have missed: linear regression internals, logistic / GLM, generative classifiers, resampling mechanics, regularization formulas, PCA / PCR / PLS, splines / GAMs, trees / RF / boosting, neural networks, unsupervised, performance metrics, statistical foundations, and KNN / SVM / misc. Wave 3 produced sections 4c (formula sheet), 4g (procedural templates), and substantial expansions to 4b, 4d, 4f, and §5. All quoted material is verbatim from Whisper transcripts.*
