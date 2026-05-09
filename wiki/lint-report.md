# Wiki lint report

Generated 2026-05-09. Wiki integrity check across `wiki/lectures/` (27 files), `wiki/concepts/` (94 files), `wiki/mocs/` (12 files), `wiki/README.md`, and `docs/concepts-manifest.md`.

## Summary

- **Atoms (94)**: 32 errors (dead wikilinks across 16 files), 1 length warning, 23 info-level isolated-quote findings
- **Lectures (27)**: 152 errors (dead wikilinks across 23 files), 181 warnings (lecture `topics:` slugs not in manifest, across 26 lectures), 2 info-level length notes
- **MOCs (12)**: 0 errors, 0 warnings (clean)
- **README (1)**: 4 errors (dead links to renamed/non-existent MOCs and root file)
- **Manifest (1)**: 0 errors — every manifest entry has a file and vice versa (94/94 1:1 mapping)
- **Frontmatter (133 files total)**: 0 errors — every concept, lecture, and MOC has complete frontmatter
- **Filenames**: 0 errors — all files match required patterns
- **Specials cross-cutting bidirectional links**: 0 issues — all 6 Specials atoms (`bias-variance-tradeoff`, `regularization`, `cross-validation`, `standardization`, `multivariate-normal`, `double-descent`) are linked from every owning module's MOC and have a `## Returns in other modules` section
- **Empty H2s / placeholders / empty bodies**: 0 errors — clean
- **Out-of-scope contradictions**: 0 errors — every MOC's `## Out of scope` topic-header refers to something that intentionally has no atom; in-scope `[[atom]]` references inside OOS prose (e.g. "the concept is captured in [[double-descent]]") are deliberate cross-references, not contradictions

**Total: 188 errors, 182 warnings, 25 info-level findings.**

The bulk of the errors are mechanical: a single bad pattern (`[[scope]]` / `[[../scope]]` instead of `[[../../docs/scope]]`, repeated 25× across 11 atoms) and the wider issue that lecture pages were written with a finer-granularity topic vocabulary than the manifest's consolidated atom slugs (so most "broken" wikilinks in lectures point to merged-into-parent concepts that simply don't exist as standalone files).

---

## Errors (must fix before relying on the wiki)

### E1. Dead wikilinks — README

`wiki/README.md` references three MOC slugs that don't exist (renamed during fan-out) and one root-level file that doesn't exist.

| Line | Bad link | Suggested fix |
|---|---|---|
| L11 | `[[mocs/m05-resampling]]` | `[[mocs/m05-resample]]` (file is `m05-resample.md`) |
| L13 | `[[mocs/m07-beyond-linear]]` | `[[mocs/m07-beyondlinear]]` (file is `m07-beyondlinear.md`) |
| L18 | `[[mocs/m12-summary]]` | `[[mocs/m12-final]]` (file is `m12-final.md`) |
| L29 | `[[../course-information]]` | file does not exist; either create it or remove this bullet |

### E2. Dead wikilinks — concepts (32 total across 16 files)

The dominant pattern (25 of 32) is a broken reference to the scope file written without the `../../docs/` path prefix. From `wiki/concepts/*.md` the correct relative path is `[[../../docs/scope]]` (used correctly in `wiki/mocs/m12-final.md`).

#### E2a. Broken `[[scope]]` / `[[../scope]]` references (25)

These exist in atoms that quote or refer to the scope rule:

| File | Line(s) | Bad link |
|---|---|---|
| `wiki/concepts/backpropagation.md` | 140 | `[[../scope]]` |
| `wiki/concepts/convolutional-neural-network.md` | 67, 83, 84, 85 | `[[../scope]]` |
| `wiki/concepts/elastic-net.md` | 69 | `[[scope]]` |
| `wiki/concepts/gradient-descent-and-sgd.md` | 113, 141, 143 | `[[../scope]]` |
| `wiki/concepts/knn-classification.md` | 112 | `[[scope]]` |
| `wiki/concepts/knn-regression.md` | 75, 94, 103 | `[[scope]]` |
| `wiki/concepts/nn-regularization.md` | 182, 183, 184 | `[[../scope]]` |
| `wiki/concepts/partial-least-squares.md` | 77, 95, 109 | `[[scope]]` |
| `wiki/concepts/prediction-vs-inference.md` | 119 | `[[scope]]` |
| `wiki/concepts/random-vector-and-covariance.md` | 86 | `[[scope]]` |
| `wiki/concepts/recurrent-neural-network.md` | 88, 108, 109 | `[[../scope]]` |
| `wiki/concepts/universal-approximation.md` | 84 | `[[../scope]]` |

**Suggested fix:** global replace inside `wiki/concepts/`:
- `[[scope]]` → `[[../../docs/scope|scope]]`
- `[[../scope]]` → `[[../../docs/scope|scope]]`

#### E2b. Other broken concept-to-concept links (7)

| File | Line | Bad link | Suggested fix |
|---|---|---|---|
| `wiki/concepts/classification-tree.md` | 29 | `[[gini-index]]` | atom doesn't exist; gini is a section inside `classification-tree`. Use plain text or anchor `[[classification-tree#Gini index]]` |
| `wiki/concepts/flexibility-overfitting-underfitting.md` | 34 | `[[overfitting]]` | atom doesn't exist (sub-concept of this file); make plain text |
| `wiki/concepts/flexibility-overfitting-underfitting.md` | 35 | `[[underfitting]]` | atom doesn't exist (sub-concept of this file); make plain text |
| `wiki/concepts/regularization.md` | 51 | `[[dropout]]` | no atom; dropout is a section inside `nn-regularization`. Use `[[nn-regularization#dropout]]` or plain text |
| `wiki/concepts/regularization.md` | 247 | `[[dropout]]` | same as above |
| `wiki/concepts/variable-importance.md` | 37 | `[[gini-index]]` | atom doesn't exist; sub-concept of `classification-tree`. Use plain text |
| `wiki/concepts/variable-importance.md` | 137 | `[[gini-index]]` | same as above |

### E3. Dead wikilinks — lectures (152 total across 23 files)

Lecture pages were authored with the prof's topic-as-said vocabulary (e.g. `[[hat-matrix]]`, `[[bayes-classifier]]`, `[[multicollinearity]]`, `[[pca]]`, `[[covariance-matrix]]`, `[[k-nearest-neighbors]]`). The manifest then consolidated those into broader parent atoms — so the wikilinks resolve to nothing.

This is the **single biggest fix-target** in the lint pass. Two ways to resolve, file-by-file:
1. **Rewrite** each broken `[[<sub-concept>]]` to the parent atom with a piped alias, e.g. `[[design-matrix-and-hat-matrix|hat matrix]]`, `[[classification-setup|Bayes classifier]]`, `[[collinearity|multicollinearity]]`, `[[principal-component-analysis|PCA]]`, `[[random-vector-and-covariance|covariance matrix]]`. This is the recommended fix because lectures double as router pages — readers expect every named concept to wikilink somewhere.
2. **Strip** the `[[ ]]` brackets, leaving the term as plain prose. Faster but loses the navigation.

#### Distinct broken slugs by frequency

| Broken slug | Count | Suggested parent atom |
|---|---|---|
| `pca` | 6 | `principal-component-analysis` |
| `covariance-matrix` | 6 | `random-vector-and-covariance` |
| `gini-index` | 5 | `classification-tree` (Gini section) |
| `multicollinearity` | 4 | `collinearity` |
| `least-squares` | 4 | `least-squares-and-mle` |
| `k-nearest-neighbors` | 4 | `knn-classification` (or `knn-regression` per context) |
| `dropout` | 4 | `nn-regularization` |
| `discriminant-score` | 4 | `discriminant-score-and-decision-boundary` |
| `bayes-classifier` | 4 | `classification-setup` |
| `qq-plot` | 3 | `residual-diagnostics` |
| `pseudo-inverse` | 3 | (no atom; concept lives in `double-descent`) — make prose, anchor to `double-descent` |
| `nested-cv` | 3 | `nested-cv-and-cv-pitfalls` |
| `model-selection` | 3 | (no atom; route to MOC `m06-modelsel` or `subset-selection`) |
| `maximum-likelihood` | 3 | `least-squares-and-mle` (covers MLE/LS equivalence) |
| `hat-matrix` | 3 | `design-matrix-and-hat-matrix` |
| `variance-inflation-factor` | 2 | (no atom — out-of-scope per L08; make prose) |
| `unsupervised-learning` | 2 | `supervised-vs-unsupervised` |
| `underfitting` | 2 | `flexibility-overfitting-underfitting` |
| `svd` | 2 | (no atom — out-of-scope per L21; make prose) |
| `stochastic-gradient-descent` | 2 | `gradient-descent-and-sgd` |
| `softmax` | 2 | `activation-functions` (sigmoid/ReLU/GELU/softmax) |
| `simple-linear-regression` | 2 | `linear-regression` |
| `relu` | 2 | `activation-functions` |
| `random-vector` | 2 | `random-vector-and-covariance` |
| `overfitting` | 2 | `flexibility-overfitting-underfitting` |
| `natural-spline` | 2 | `regression-splines` |
| `misclassification-rate` | 2 | `confusion-matrix` (no dedicated atom) |
| `mallows-cp` | 2 | `aic-bic-conceptual` (Cp/AIC/BIC conceptual atom) |
| `independence-assumption` | 1 | `gaussian-error-assumptions` (independence is part of it) |
| `correlation-matrix` | 2 | `random-vector-and-covariance` |
| `clustering` | 2 | (no parent; route to `m10-unsuper` MOC or alias to `k-means-clustering`) |
| `categorical-encoding` | 2 | `categorical-encoding-and-interactions` |
| `bayes-error-rate` | 1 | `classification-setup` |
| `interpretability` | 2 | (no atom; per manifest §8 of edge cases — distributed across PDP/var-importance) |

The 1-occurrence broken slugs are similar — see per-file detail below.

#### Per-file dead links — lectures

<details>
<summary>Click to expand per-file lecture dead-link list</summary>

**wiki/lectures/L01-intro.md** (2)
- L206: `[[regression]]` — no atom; alias to `linear-regression` or make prose
- L221: `[[classification]]` — no atom; alias to `classification-setup`

**wiki/lectures/L02-statlearn-1.md** (8)
- L33: `[[supervised-learning]]`, `[[unsupervised-learning]]` — alias to `supervised-vs-unsupervised`
- L48: `[[misspecification]]`
- L97: `[[regression-vs-classification]]`
- L149: `[[clustering]]`, `[[pca]]` — alias to `k-means-clustering` and `principal-component-analysis`
- L240: `[[least-squares]]` — alias to `least-squares-and-mle`
- L250: `[[statlearn-2]]` — typo for `[[L03-statlearn-2]]`

**wiki/lectures/L03-statlearn-2.md** (3)
- L29: `[[training-vs-test-mse]]`
- L161: `[[overfitting]]` — alias to `flexibility-overfitting-underfitting`
- L162: `[[underfitting]]` — alias to `flexibility-overfitting-underfitting`

**wiki/lectures/L04-statlearn-3.md** (10)
- L28, L33, L67: `[[pseudo-inverse]]` — out-of-scope; concept lives in `double-descent`. Make prose with anchor `[[double-descent]]`
- L28, L113: `[[random-vector]]` — alias to `random-vector-and-covariance`
- L28, L36, L143, L170: `[[covariance-matrix]]` — alias to `random-vector-and-covariance`
- L28, L36: `[[correlation-matrix]]` — alias to `random-vector-and-covariance`

**wiki/lectures/L05-linreg-1.md** (12)
- L31, L36, L111: `[[least-squares]]` — alias to `least-squares-and-mle`
- L31, L84: `[[simple-linear-regression]]` — alias to `linear-regression`
- L66: `[[parametric-model]]` — alias to `parametric-vs-nonparametric`
- L79: `[[categorical-encoding]]` — alias to `categorical-encoding-and-interactions`
- L83: `[[bmi]]` — no atom; make prose
- L90: `[[statistical-vs-practical-significance]]` — alias to `t-test-and-significance`
- L121: `[[maximum-likelihood]]` — alias to `least-squares-and-mle`
- L153: `[[independence-assumption]]` — alias to `gaussian-error-assumptions`
- L207: `[[residual-sum-of-squares]]` — no atom; make prose
- L271: `[[linreg-2]]` — typo for `[[L06-linreg-2]]`

**wiki/lectures/L06-linreg-2.md** (14)
- L36, L41, L149: `[[hat-matrix]]` — alias to `design-matrix-and-hat-matrix`
- L36: `[[normal-equations]]` — alias to `design-matrix-and-hat-matrix`
- L36, L232: `[[model-selection]]` — no atom; alias to MOC `[[m06-modelsel]]` or `[[subset-selection]]`
- L36, L48, L65: `[[qq-plot]]` — alias to `residual-diagnostics`
- L60, L235: `[[adjusted-r-squared]]` — alias to `r-squared`
- L260: `[[confidence-interval]]` — alias to `confidence-and-prediction-intervals`
- L261: `[[prediction-interval]]` — alias to `confidence-and-prediction-intervals`
- L306: `[[categorical-encoding]]`, `[[reference-category]]` — alias to `categorical-encoding-and-interactions`

**wiki/lectures/L07-classif-1.md** (5)
- L32, L42, L166, L222: `[[bayes-classifier]]` — alias to `classification-setup`
- L97: `[[maximum-likelihood]]` — alias to `least-squares-and-mle`
- L252: `[[bayes-theorem]]` — no atom; make prose

**wiki/lectures/L08-classif-2.md** (14)
- L37, L144: `[[variance-inflation-factor]]` — out-of-scope per L08; make prose
- L41, L256: `[[discriminant-score]]` — alias to `discriminant-score-and-decision-boundary`
- L65: `[[qq-plot]]` — alias to `residual-diagnostics`
- L84: `[[simple-linear-regression]]` — alias to `linear-regression`
- L98: `[[ols]]` — alias to `linear-regression` or `least-squares-and-mle`
- L104: `[[moore-penrose-pseudoinverse]]` — out-of-scope; make prose with anchor `[[double-descent]]`
- L145, L298: `[[pca]]` — alias to `principal-component-analysis`
- L176, L256: `[[knn-classifier]]` — alias to `knn-classification`
- L222: `[[bayes-classifier]]` — alias to `classification-setup`
- L230: `[[bayes-error-rate]]` — alias to `classification-setup`
- L302: `[[classif-3]]` — typo for `[[L09-classif-3]]`

**wiki/lectures/L09-classif-3.md** (11)
- L30, L34, L98: `[[discriminant-score]]` — alias to `discriminant-score-and-decision-boundary`
- L30: `[[decision-boundary]]` — alias to `discriminant-score-and-decision-boundary`
- L44, L256: `[[k-nearest-neighbors]]` — alias to `knn-classification`
- L143, L198, L247: `[[covariance-matrix]]` — alias to `random-vector-and-covariance`
- L228: `[[fisher]]` — no atom; make prose
- L269: `[[resampling-methods]]` — alias to MOC `[[m05-resample]]`

**wiki/lectures/L10-resample-1.md** (9)
- L31, L87: `[[aic-bic]]` — alias to `aic-bic-conceptual`
- L31: `[[independence-assumption]]` — alias to `gaussian-error-assumptions`
- L31, L153: `[[k-fold-cross-validation]]` — alias to `k-fold-cv`
- L31, L152: `[[leave-one-out-cross-validation]]` — alias to `leave-one-out-cv`
- L46, L110: `[[k-nearest-neighbors]]` — alias to `knn-classification`/`knn-regression`

**wiki/lectures/L11-resample-2.md** (11)
- L28, L34, L91: `[[nested-cv]]` — alias to `nested-cv-and-cv-pitfalls`
- L33, L58: `[[misclassification-rate]]` — alias to `confusion-matrix`
- L60: `[[bayes-decision-boundary]]` — alias to `classification-setup` or `discriminant-score-and-decision-boundary`
- L62: `[[linear-predictor]]` — no atom; make prose
- L67: `[[Misclassification-rate]]` — case-mismatch; same alias as above + lowercase fix
- L137: `[[bradley-efron]]` — no atom (history); make prose per "no history questions" scope rule
- L151: `[[empirical-distribution]]` — no atom; make prose or alias to `bootstrap`
- L196: `[[multiple-linear-regression]]` — alias to `linear-regression`

**wiki/lectures/L12-modelsel-1.md** (1)
- L283: `[[lasso-regression]]` — alias to `lasso`

**wiki/lectures/L13-modelsel-2.md** (5)
- L28, L83: `[[mallows-cp]]` — alias to `aic-bic-conceptual`
- L74: `[[best-subset-selection]]` — alias to `subset-selection`
- L75: `[[forward-stepwise]]` — alias to `subset-selection`
- L76: `[[backward-stepwise]]` — alias to `subset-selection`

**wiki/lectures/L14-modelsel-3.md** (2)
- L155, L236: `[[multicollinearity]]` — alias to `collinearity`

**wiki/lectures/L15-modelsel-4.md** (6)
- L38, L261: `[[multicollinearity]]` — alias to `collinearity`
- L46: `[[forward-selection]]`, `[[backward-selection]]` — alias to `subset-selection`
- L118: `[[best-subset-selection]]` — alias to `subset-selection`
- L159: `[[wold]]` — no atom (history); make prose

**wiki/lectures/L17-trees-1.md** (3)
- L50, L59: `[[natural-spline]]` — alias to `regression-splines`
- L88: `[[cubic-spline]]` — alias to `regression-splines`

**wiki/lectures/L18-trees-2.md** (5)
- L28, L120: `[[gini-index]]` — alias to `classification-tree`
- L28, L121: `[[cross-entropy]]` — alias to `classification-tree`
- L56: `[[model-selection]]` — alias to MOC `[[m06-modelsel]]`

**wiki/lectures/L21-unsupervised-1.md** (11)
- L32: `[[clustering]]`, `[[dropout]]`, `[[explained-variance]]`, `[[interpretability]]`, `[[loadings]]`, `[[pca]]`, `[[scree-plot]]`, `[[svd]]`, `[[unsupervised-learning]]` — bulk alias rewrite needed
  - `clustering` → `k-means-clustering` or `hierarchical-clustering`
  - `dropout` → `nn-regularization`
  - `explained-variance`, `scree-plot` → `explained-variance-and-scree-plot`
  - `interpretability` → no atom; make prose or alias to `partial-dependence-plots`
  - `loadings` → `principal-component-analysis`
  - `pca` → `principal-component-analysis`
  - `svd` → no atom (out-of-scope); make prose
  - `unsupervised-learning` → `supervised-vs-unsupervised`
- L40: `[[svd]]` — same
- L96: `[[interpretability]]` — same

**wiki/lectures/L22-unsupervised-2.md** (1)
- L51: `[[pca]]` — alias to `principal-component-analysis`

**wiki/lectures/L23-nnet-1.md** (6)
- L30, L92: `[[softmax]]` — alias to `activation-functions`
- L30: `[[sigmoid]]`, `[[relu]]` — alias to `activation-functions`
- L30: `[[stochastic-gradient-descent]]`, `[[mini-batch]]` — alias to `gradient-descent-and-sgd`

**wiki/lectures/L24-nnet-2.md** (1)
- L33: `[[dropout]]` — alias to `nn-regularization`

**wiki/lectures/L26-nnet-3.md** (5)
- L84: `[[weight-sharing]]` — no atom; make prose or alias to `convolutional-neural-network` / `recurrent-neural-network`
- L113: `[[autocorrelation]]` — no atom; make prose
- L142: `[[stochastic-gradient-descent]]` — alias to `gradient-descent-and-sgd`
- L154: `[[shapley-values]]` — out-of-scope per L26; make prose
- L202: `[[natural-splines]]` — alias to `regression-splines`

**wiki/lectures/L27-summary.md** (7)
- L122: `[[neural-network]]` — alias to `feedforward-network`
- L122: `[[relu]]` — alias to `activation-functions`
- L149: `[[pca]]` — alias to `principal-component-analysis`
- L219: `[[gam]]` — alias to `generalized-additive-models`
- L238: `[[roc-curve]]` — alias to `roc-auc`
- L244: `[[knn]]` — alias to `knn-classification` or `knn-regression`
- L262: `[[maximum-likelihood]]` — alias to `least-squares-and-mle`

</details>

---

## Warnings (should fix)

### W1. Lecture `topics:` slugs not in concepts manifest (181 instances across 26 lectures)

Lecture frontmatter `topics:` lists fine-grained concept slugs (the prof's spoken vocabulary) — most don't match the manifest's consolidated atom slugs. Same root cause as E3: the lectures were authored before the manifest's atom-merging pass.

Fix options per slug:
1. **Rename the topic** to the manifest's parent slug (e.g. `topics: [hat-matrix]` → `topics: [design-matrix-and-hat-matrix]`)
2. **Add a new atom** if the topic is genuinely standalone and just got missed
3. **Drop the topic entry** if it's redundant after the rename

#### Frequency table — broken topic slugs

The most common renamings needed (apply consistently across all 26 lectures):

| Broken topic slug | Suggested rename (or "drop") | Lectures affected |
|---|---|---|
| `maximum-likelihood` | merge into `least-squares-and-mle` | L05, L06, L07, L27 |
| `pca` | rename to `principal-component-analysis` | L21, L27 |
| `multicollinearity` | rename to `collinearity` | L14, L15 |
| `mallows-cp`, `aic`, `bic`, `aic-bic` | merge into `aic-bic-conceptual` | L10, L12, L13 |
| `simple-linear-regression`, `multiple-linear-regression` | merge into `linear-regression` | L05, L08, L11 |
| `least-squares` | merge into `least-squares-and-mle` | L05 |
| `hat-matrix`, `normal-equations`, `design-matrix` | merge into `design-matrix-and-hat-matrix` | L06, L08, L10 |
| `qq-plot`, `leverage`, `studentized-residuals`, `residual-plots`, `residuals` | merge into `residual-diagnostics` | L06, L08, L20 |
| `confidence-interval`, `prediction-interval` | merge into `confidence-and-prediction-intervals` | L05, L06 |
| `categorical-encoding`, `reference-category`, `interactions` | merge into `categorical-encoding-and-interactions` | L05, L06, L27 |
| `forward-stepwise`, `backward-stepwise`, `forward-stepwise-selection`, `backward-stepwise-selection`, `hybrid-stepwise-selection`, `best-subset-selection` | merge into `subset-selection` | L12, L13 |
| `bayes-classifier`, `bayes-decision-boundary`, `bayes-error-rate` | merge into `classification-setup` | L07, L08, L09 |
| `discriminant-score`, `pooled-covariance` | merge into `discriminant-score-and-decision-boundary` (or LDA atom) | L08, L09 |
| `knn-classifier`, `k-nearest-neighbors`, `knn` | rename to `knn-classification` or `knn-regression` per context | L08, L09, L10, L27 |
| `leave-one-out-cross-validation` | rename to `leave-one-out-cv` | L10 |
| `k-fold-cross-validation` | rename to `k-fold-cv` | L10 |
| `nested-cv`, `cv-wrong-way` | merge into `nested-cv-and-cv-pitfalls` | L11 |
| `dropout`, `early-stopping`, `data-augmentation`, `label-smoothing`, `transfer-learning`, `l1-regularization`, `l2-regularization`, `l1-l2-regularization` | merge into `nn-regularization` | L20, L21, L24 |
| `relu`, `sigmoid`, `softmax`, `activation-function` | merge into `activation-functions` | L22, L23, L24, L27 |
| `gradient-descent`, `stochastic-gradient-descent`, `mini-batch` | merge into `gradient-descent-and-sgd` | L23, L24 |
| `cross-entropy-loss` | merge into `nn-loss-functions` | L23 |
| `pooling`, `cnn` | merge into `convolutional-neural-network` | L24 |
| `weight-sharing` | merge into `convolutional-neural-network` or `recurrent-neural-network` | L26 |
| `cubic-spline`, `natural-spline`, `splines` | merge into `regression-splines` | L16, L17, L26 |
| `effective-degrees-of-freedom` | merge into `smoothing-splines` | L16 |
| `recursive-binary-splitting`, `cart`, `tree-pruning` | merge into `regression-tree` / `cost-complexity-pruning` | L17, L18 |
| `regression-trees`, `classification-trees` | rename (singular) `regression-tree` / `classification-tree` | L18 |
| `gini-index`, `cross-entropy` | merge into `classification-tree` | L18 |
| `random-forests` | rename to `random-forest` | L18 |
| `weak-learner`, `learning-rate` | merge into `weak-learner-and-learning-rate` | L19, L20 |
| `quadratic-loss`, `absolute-loss`, `huber-loss`, `binomial-deviance`, `multinomial-deviance` | merge into `boosting-loss-functions` | L20 |
| `steepest-descent`, `decorrelation` | drop (mechanism, not standalone concept) | L19, L20 |
| `loadings`, `principal-components`, `eigenfaces`, `explained-variance`, `scree-plot`, `svd` | merge into `principal-component-analysis` / `explained-variance-and-scree-plot` (drop `svd` & `eigenfaces` per scope) | L21 |
| `clustering`, `dendrogram`, `linkage`, `dissimilarity-measure`, `euclidean-distance`, `correlation-distance` | merge into `hierarchical-clustering` / `distance-metrics` | L22 |
| `unsupervised-learning`, `supervised-learning`, `regression-vs-classification`, `quantitative-vs-qualitative`, `statistical-vs-machine-learning`, `training-test-validation`, `regression`, `classification` | merge into `supervised-vs-unsupervised` / `training-validation-test-split` / drop branding terms | L01, L02, L03 |
| `random-vector`, `covariance-matrix`, `correlation-matrix`, `marginal-distribution` | merge into `random-vector-and-covariance` | L04 |
| `pseudo-inverse`, `benign-overfitting` | drop / merge into `double-descent` | L04, L24 |
| `flexibility`, `overfitting`, `underfitting`, `training-vs-test-mse` | merge into `flexibility-overfitting-underfitting` | L02, L03 |
| `odds`, `odds-ratio`, `logit-link`, `bernoulli-distribution`, `newtons-method` | merge into `logistic-regression` / `odds-and-log-odds` | L07, L27 |
| `t-test`, `p-value`, `standard-error`, `statistical-vs-practical-significance` | merge into `t-test-and-significance` | L05, L11 |
| `gaussian-errors`, `independence-assumption` | merge into `gaussian-error-assumptions` | L05, L10 |
| `parametric-model` | rename to `parametric-vs-nonparametric` | L05 |
| `multicollinearity` | rename to `collinearity` | L14, L15 |
| `principal-components-regression` | rename to `principal-component-regression` | L14 |
| `model-selection`, `shrinkage`, `hyperparameter`, `implicit-regularization`, `bayesian-interpretation-of-regularization` | drop or alias to `regularization` / `subset-selection` | L12, L13, L14 |
| `data-models-vs-algorithmic-models`, `data-reuse`, `greedy-algorithm`, `interpretability`, `deep-learning`, `neural-networks`, `mnist`, `hidden-layer`, `exam-logistics`, `exam-scope` | drop (descriptive labels, not concept atoms) | L17, L21, L22, L23, L26, L27 |
| `gam` | rename to `generalized-additive-models` | L27 |
| `roc-curve` | rename to `roc-auc` | L27 |
| `misclassification-rate`, `misspecification`, `empirical-distribution`, `linkage`, `autocorrelation` | drop or fold into nearest atom | L02, L07, L08, L11, L26 |

#### Per-lecture topic warnings

Full list available in `/tmp/lint_data/topic_warns.txt`; the most-affected lectures are L08 (9), L21 (11), L20 (11), L23 (10), L24 (12), L05 (12), L06 (12). Fix by editing each lecture's frontmatter `topics:` list to use only manifest slugs.

### W2. Length warnings — concepts (1)

| File | Line count | Note |
|---|---|---|
| `wiki/concepts/training-validation-test-split.md` | 77 | Just under the 80-line floor; either bulk it up with prof signal from L10 or accept as a deliberately thin atom |

No Specials atoms exceed 350 lines. No regular atoms exceed 250 lines.

---

## Info (nice to fix)

### I1. Lecture length notes (2)

| File | Line count | Note |
|---|---|---|
| `wiki/lectures/L01-intro.md` | 248 | 2 lines below the 250 floor — fine; was a short opening lecture |
| `wiki/lectures/L25-nnet-3-aborted.md` | 20 | Intentional stub for an aborted recording (10-character Whisper transcript "Thank you."); see file body |

### I2. Quote anchoring — atoms with blockquotes lacking nearby lecture wikilinks (23)

Lint heuristic: any blockquote block (consecutive `> ` lines, excluding Obsidian `[!callout]` blocks) without a `[[L<NN>-...]]` link within ±2 lines. Most of these are slide-deck quotes (acceptable per the template, since slides aren't lectures), or quotes deep inside a long atom where the nearest lecture link is further away than the heuristic catches. Worth a manual scan to see which actually need an anchor added.

| File:line | Quote opening |
|---|---|
| `wiki/concepts/classification-tree.md:25` | "Now allow for K ≥ 2 number of classes for the response..." |
| `wiki/concepts/confidence-and-prediction-intervals.md:67, 71` | PI uncertainty decomposition / 95% CI definition |
| `wiki/concepts/cost-complexity-pruning.md:25` | "Better idea: grow a very large tree T₀..." |
| `wiki/concepts/diagnostic-vs-sampling-paradigm.md:67` | LDA log-odds linearity result |
| `wiki/concepts/flexibility-overfitting-underfitting.md:34, 35` | overfitting / underfitting definitions (also broken wikilinks — see E2b) |
| `wiki/concepts/generalized-additive-models.md:56` | backfitting algorithm description |
| `wiki/concepts/hierarchical-clustering.md:44–46` | algorithm pseudocode (mid-block, anchor likely earlier) |
| `wiki/concepts/high-dimensional-regression.md:66` | "Multicollinearity: any variable in the model..." |
| `wiki/concepts/k-fold-cv.md:43` | "Setting k = n gives LOOCV" — slide deck |
| `wiki/concepts/k-means-clustering.md:64, 65` | algorithm steps |
| `wiki/concepts/knn-classification.md:64` | exam-style question on K and Bayes boundary |
| `wiki/concepts/local-regression.md:87` | "(iv) The K-nearest neighbors regression..." |
| `wiki/concepts/nested-cv-and-cv-pitfalls.md:38, 39, 41, 121` | wrong-way CV walkthrough |
| `wiki/concepts/one-standard-error-rule.md:68` | "Strictly speaking, this estimate is not quite valid..." |
| `wiki/concepts/partial-least-squares.md:43, 47, 83` | PLS coefficient definitions / PLS≈ridge verdict |
| `wiki/concepts/polynomial-regression.md:45` | "Sometimes the world is not linear..." |
| `wiki/concepts/principal-component-regression.md:23, 89` | PCR construction / PCR = discretized ridge |
| `wiki/concepts/random-forest.md:104` | "Different solutions are possible..." |
| `wiki/concepts/regression-splines.md:57, 58` | natural-cubic dof trap (Obsidian callout) |
| `wiki/concepts/ridge-vs-lasso-geometry.md:30` | "The red ellipses are the contours of the RSS..." |
| `wiki/concepts/roc-auc.md:26, 28, 77` | ROC definition / ROC construction / AUC use |
| `wiki/concepts/sensitivity-specificity.md:28, 30, 58` | sens/spec definitions |
| `wiki/concepts/smoothing-splines.md:43–56, 125` | λ-direction trap (Obsidian callout); smoothing penalty derivation |
| `wiki/concepts/subset-selection.md:30, 83` | subset-selection definition / forward-stepwise hi-dim note |

---

## Suggested next actions

In rough priority order:

1. **Mechanical fix — `[[scope]]` / `[[../scope]]` in 11 atoms (25 broken links).** Simple find-and-replace: `[[scope]]` → `[[../../docs/scope|scope]]`, `[[../scope]]` → `[[../../docs/scope|scope]]`. Affected files listed in E2a. **5 minutes.**

2. **Mechanical fix — README typos (4 broken links).** Edit `wiki/README.md` lines 11, 13, 18, 29 to use the actual MOC filenames (`m05-resample`, `m07-beyondlinear`, `m12-final`) and either create or drop the `course-information` reference. **2 minutes.**

3. **Bulk rewrite — lecture wikilinks to consolidated parent atoms (152 broken links across 23 lectures).** This is the largest mechanical pass. Build a substitution table from the parent-atom mapping in E3 and apply per-file. Recommended approach: re-run the **lectures agent** on each affected lecture with the substitution table as context, asking only to fix wikilinks (don't regenerate prose). Alternatively edit by hand — every broken link has a clear target in the table. **45–90 minutes** by hand, **~15 minutes** via re-run.

4. **Bulk rewrite — lecture `topics:` frontmatter (181 warnings across 26 lectures).** Same fix as (3) but in the frontmatter. The substitution table from W1 covers this. Best done in the same lecture-fix pass as (3). **20–30 minutes** by hand.

5. **Concept-to-concept dead links (7 in atoms).** Decisions per E2b: most should become plain text since the referenced terms are sub-concepts inside a parent atom, not standalone atoms. **5 minutes.**

6. **Length boost — `training-validation-test-split.md` (77 lines).** Re-run the **concepts agent** on this slug only with instructions to add 5–10 lines of L10 signal. Optional. **5 minutes.**

7. **Quote anchoring sweep (23 atoms).** Optional info-level cleanup. Add `— [[L<NN>-…]]` after each isolated blockquote. Hand pass. **15–20 minutes** if doing all.

8. **Verify no agent re-runs are needed.** After (1)–(5), re-run this lint pass. The atom and MOC structure is sound (no frontmatter errors, no Specials bidirectional gaps, no out-of-scope contradictions, no empty bodies, no placeholder text, no filename violations); the wiki's content layer is healthy and the cleanup is purely link-pointer hygiene.

**Total fix-effort estimate: ~90–150 minutes** (or ~25 minutes if you re-run agents instead of hand-editing for steps 3–4).
