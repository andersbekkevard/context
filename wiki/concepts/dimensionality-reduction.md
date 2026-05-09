---
concept: dimensionality-reduction
module: 10-unsuper
lectures: [L08, L09, L14, L15, L21]
isl-ref: 6.3
exercises: []
related: [principal-component-analysis, principal-component-regression, partial-least-squares, linear-discriminant-analysis, explained-variance-and-scree-plot, curse-of-dimensionality, collinearity, k-means-clustering, hierarchical-clustering]
tags:
  - concept
  - module/10-unsuper
aliases:
  - dim reduction
  - feature compression
---

# Dimensionality reduction

The umbrella concept underneath PCA / PCR / PLS / LDA-as-projection / NN-feature-extractor: **collapse $p$ predictors into $M < p$ composite ones** for visualization, downstream regression, or as a fix for [[collinearity]] / [[curse-of-dimensionality]]. The prof returns to it across modules — every method that takes a wide $X$ and produces a narrower one is an instance.

## Definition (prof's framing)

> "Dimensionality reduction tries to reduce, collapse the dimensions … both dimensionality reduction and clustering approaches are trying to simplify the data — to give simple summaries. Dimensionality reduction looks for low-dimensional ones … clustering tries to find groups." — [[L21-unsupervised-1]] / [[L22-unsupervised-2]]

Construct $M$ new variables $Z_1, \ldots, Z_M$ as linear (or nonlinear) functions of the original $X_1, \ldots, X_p$, with $M < p$. Then use the $Z$'s in place of the $X$'s downstream — for visualization (project to 2 / 3 dims), for regression (PCR / PLS), or as input to another model.

> "There's a bunch of dimensionality reduction methods. [PCA] is by far the oldest and the simplest and often the most useful." — [[L21-unsupervised-1]]

## Notation & setup

- $X \in \mathbb{R}^{n \times p}$: original data matrix.
- $Z \in \mathbb{R}^{n \times M}$: reduced data matrix, $M < p$.
- For **linear** dim reduction, each $Z_m = \sum_j \phi_{jm} X_j$ — a linear combination with weights $\phi_{jm}$.
- For **nonlinear** dim reduction (NN feature extractor), $Z = g(X)$ for some learned $g$.

## The unifying recipe

Linear dimensionality reduction is one trick repeated with different choices for the $\phi$'s:

$$Z_m = \sum_{j=1}^p \phi_{jm} X_j, \qquad m = 1, \ldots, M.$$

After reducing, you typically fit a model on $Z$:

$$y = \sum_{m=1}^M \theta_m Z_m + \varepsilon.$$

Composing the two linear maps gives back implied coefficients on the original $X$'s:

$$\beta_j = \sum_{m=1}^M \theta_m \phi_{jm}.$$

> "You're taking the X, you're squishing it down to fit your model, and then you go backwards to the original model again." — [[L14-modelsel-3]]

This is the formal equivalence that makes PCR / PLS feel like "regression with a different prior on $\beta$" — they constrain $\beta$ to live in an $M$-dimensional subspace.

## The instances in this course

| Method | $\phi$ chosen to … | Supervised? | Module |
|---|---|---|---|
| **PCA** | maximize $\text{Var}(Z_m)$ s.t. $\|\phi_m\| = 1$ and $\phi_m \perp \phi_{m'}$ for $m' < m$ | No | 10 (and used in 6) |
| **PCR** | same as PCA, then regress $y$ on first $M$ PCs | No (for $\phi$); yes for $\theta$ | 6 |
| **PLS** | maximize $\text{Cov}(Z_m, Y)$ s.t. orthogonality | Yes | 6 |
| **LDA** | project to $K-1$ discriminant directions max-separating class means | Yes | 4 |
| **NN feature extractor** | last hidden layer of a trained network used as features for downstream regression | Yes (via task loss) | 11 |

> "The PCR pattern — compress $X$ down to fewer features with some method, then regress — generalizes far beyond PCA. Example: video frames as $X$. They're huge. Run them through a learned feature extractor (a neural net) to get a small vector per frame, then regress $Y$ on that compressed representation." — [[L15-modelsel-4]]

So *the same idea* connects an unsupervised classical algorithm (PCA), a supervised classical algorithm (PLS), a generative classifier (LDA-as-projection), and modern deep learning (transfer learning / feature extraction).

## Insights & mental models

**Three reasons to do it.**
1. **Visualization** — plot $n$ observations in the first 2 PCs to see structure that's invisible in $p > 3$ dims. "Who can think in eight dimensions?" — [[L21-unsupervised-1]]
2. **Defeat [[collinearity]] / well-pose $p > n$ regression.** Orthogonal $Z$'s remove the redundancy that makes $X^\top X$ singular. PCR / ridge / lasso all attack the same core problem; dim reduction is the "rotate it away" version.
3. **Compression for downstream models.** Smaller, easier regressions; faster fits; sometimes better generalization (regularization-by-projection).

**Supervised vs unsupervised is the key axis.** PCA / PCR pick directions using only $X$ → can miss directions that matter for $Y$ ("a key assumption … is that the response actually lives in the directions of largest $X$-variance" — [[L15-modelsel-4]]). PLS / LDA / NN-feature-extractor use $Y$ to guide selection → won't miss it but adds variance and risks overfitting to $Y$.

**Linear vs nonlinear.** PCA / PCR / PLS / LDA are all linear projections — they can't capture curved structure. Nonlinear extensions (kernel PCA, autoencoders, NN feature extractors) handle that, but at the cost of interpretability.

**Standardize first** for any variance-driven method (PCA, PCR, PLS, k-means inside the pipeline). See [[standardization]].

**$M$ is the central tuning knob.**
- Unsupervised: pick from scree / cumulative PVE — see [[explained-variance-and-scree-plot]].
- Supervised: pick by **cross-validating the downstream model**. Treat $M$ like any other tuning parameter.

**Dim reduction ≠ variable selection.** PCR / PCA give *combinations* of all original variables — every $X_j$ appears in every $Z_m$ in general. If you need to *drop* variables (interpretability, cost of measurement), use [[lasso]] or [[subset-selection]]. The prof: "Use PCR/ridge when you want predictive power and don't care which raw variables drive it. Use lasso or subset selection when you need this or that one." — [[L15-modelsel-4]]

**The cousin discipline: clustering** also "simplifies" high-dim data, but produces a *discrete* summary (group label per point) rather than a continuous low-dim projection. The prof groups them as the two "summarize high-dim data" tools in module 10.

## Returns across the course

- **L08 ([[L08-classif-2]])**: introduced as the fix for collinearity / curse of dimensionality. PCA is the canonical tool; LDA is previewed as another dim-reduction route.
- **L09 ([[L09-classif-3]])**: LDA framed as $K$ classes → $K-1$ discriminant scores. "This is why LDA is often listed as a dimensionality-reduction method."
- **L14 ([[L14-modelsel-3]])**: PCR pipeline introduced — standardize → PCA → regress on first $M$ PCs → back-transform. The unifying recipe (build $Z$ from $X$, fit, back out $\beta$) is laid down here.
- **L15 ([[L15-modelsel-4]])**: PCR finished; PLS as the supervised counterpart; the "PCR pattern generalizes to NN feature extractors" insight.
- **L21 ([[L21-unsupervised-1]])**: PCA covered in full, framed as "the first dim-reduction tool." Pivot to clustering as the discrete-summary alternative.

## Exam signals

> "Dimensionality reduction looks for low-dimensional ones … clustering tries to find groups. So it's more of a discretized thing. But in both cases, you're looking at a short description of high-dimensional data." — [[L21-unsupervised-1]]

> "The PCR pattern — compress $X$ down to fewer features with some method, then regress — generalizes far beyond PCA." — [[L15-modelsel-4]]

> "By far the oldest and the simplest and often the most useful." — [[L21-unsupervised-1]] (on PCA among dim-reduction methods)

## Pitfalls

- **Confusing dim reduction with variable selection.** Different goals, different tools.
- **Using PCA when $Y$ matters.** PCA is unsupervised — large-variance directions of $X$ may not be predictive directions for $Y$.
- **Not standardizing.** Dim reduction is variance-driven; mixed units → meaningless.
- **Linear-only methods on curved data.** PCA can't capture nonlinear manifolds; if the data lies on a curve / circle / arc, PCA's first two PCs may completely miss the structure. Switch to nonlinear method or accept the limitation.
- **Picking $M$ unsupervised when there's a supervised task.** Cross-validate $M$ against the downstream model — much more principled than the elbow on a scree plot.
- **Treating "low-dim" as "interpretable."** PCs are linear combinations of all original variables; loadings *help* interpretation but don't give clean attribution to single $X$'s.

## Scope vs ISLR

- **In scope:** the unified frame (build $Z$ from $X$, fit, back out $\beta$); the supervised/unsupervised axis; the four/five canonical methods (PCA, PCR, PLS, LDA-as-projection, NN feature extractor); when to use which; standardization mandate; visualization use case; collinearity / high-dim motivation.
- **Look up in ISLR:** §6.3 (the dimensionality-reduction frame for regression — PCR + PLS); §12.2 (PCA in the unsupervised-learning chapter); §4.4 (LDA, with the projection viewpoint).
- **Skip in ISLR:**
  - Spectral / eigen decomposition derivations of $\Sigma$ — out per [[L04-statlearn-3]].
  - Kernel PCA, ICA, manifold learning (t-SNE, UMAP, MDS) — none lectured. Out.
  - Detailed PLS history & chemometrics tuning — [[L15-modelsel-4]]: PCR is the workhorse; PLS gets the one-liner.

## Exercise instances

(No exercises uniquely target *the umbrella concept* — every dim-reduction exercise lives under a specific instance atom: PCA in [[principal-component-analysis|Exercise 10.1]] / Exercise 6.7, PCR in Exercises 6.7–6.8, PLS in Exercise 6.9, LDA in Exercise 4.2.)

## How it might appear on the exam

- **Method comparison:** PCA vs PLS vs LDA vs lasso — match each to its goal (unsupervised dim reduction; supervised dim reduction; class-separating projection; sparse variable selection).
- **Conceptual MC:**
  - "PCA picks directions using $Y$." → **False**.
  - "PLS picks directions using $Y$." → **True**.
  - "Dim reduction performs variable selection." → **False** (the $Z$'s are linear combinations of all $X$'s).
- **Output interpretation:** given a biplot or a 2D projection of NYT stories, identify what the projection has captured.
- **"Why dim-reduce?"** answer: visualization, defeat collinearity, well-pose $p > n$, regularize the downstream model.
- **Choosing $M$:** when to use scree (unsupervised) vs CV (supervised).

## Related

- [[principal-component-analysis]] — the canonical instance; unsupervised and linear.
- [[explained-variance-and-scree-plot]] — how to choose $M$ in the unsupervised case.
- [[principal-component-regression]] — PCA as the front end of regression; one of the module-6 dim-reduction routes.
- [[partial-least-squares]] — the supervised counterpart of PCR.
- [[linear-discriminant-analysis]] — supervised linear projection that doubles as $K-1$-dim reduction.
- [[curse-of-dimensionality]] — the high-$p$ pathology that dim reduction defends against.
- [[collinearity]] — the second pathology dim reduction handles by rotating to orthogonal $Z$'s.
- [[k-means-clustering]] / [[hierarchical-clustering]] — the discrete-summary cousins of dim reduction; also "simplify high-dim data."
