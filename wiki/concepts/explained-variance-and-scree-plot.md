---
concept: explained-variance-and-scree-plot
module: 10-unsuper
lectures: [L14, L15, L21]
isl-ref: 12.2.3
exercises:
  - Exercise6.7 — pick how many PCs to use for the Credit dataset, justify
  - Exercise10.1 — produce PVE and cumulative PVE plots for NYT stories, decide where to chop
related: [principal-component-analysis, principal-component-regression, dimensionality-reduction, cross-validation]
tags:
  - concept
  - module/10-unsuper
aliases:
  - PVE
  - scree plot
  - proportion of variance explained
---

# PCA explained variance and scree plot

The companion to [[principal-component-analysis|PCA]]: how do you decide how many components to keep? Two plots, both built from the eigenvalues — the **scree plot** (PVE per PC) and the **cumulative PVE** curve. The recipe: chop where the marginal PVE flattens out, or where the cumulative crosses a threshold (90/95/99%). When PCA is the front end of a supervised pipeline, **CV on the held-out response** beats either heuristic.

## Definition (prof's framing)

> "Each subsequent dimension is the direction of maximal variance … you're going to be explaining less and less variance." — [[L14-modelsel-3]]

The **proportion of variance explained** by PC $m$ is the share of total data variance that PC $m$ accounts for:

$$\text{PVE}_m = \frac{\lambda_m}{\sum_{k=1}^p \lambda_k} = \frac{\text{Var}(Z_m)}{\sum_k \text{Var}(X_k)}.$$

The **scree plot** plots $\text{PVE}_m$ against $m$ (decreasing); the **cumulative PVE** plots $\sum_{k\le m}\text{PVE}_k$ against $m$ (saturating toward 1). Both are read off the eigenvalues, both have $p$ points on the x-axis (one per PC).

## Notation & setup

- $\lambda_m$ — variance of PC $m$, equivalently the $m$-th eigenvalue of the sample covariance $\Sigma = \tfrac{1}{n-1} X^\top X$ (after centering / standardization).
- $\sum_m \lambda_m = \sum_j \text{Var}(X_j)$ — total variance is preserved under the orthonormal rotation. After standardization each $\text{Var}(X_j) = 1$, so total variance = $p$ (handy mental check).
- $\text{PVE}_m \in (0, 1)$, sums to 1 over $m = 1, \ldots, \min(n-1, p)$.
- "Cumulative PVE" = $\sum_{k=1}^m \text{PVE}_k$ — the share of variance kept if you truncate at $M = m$.

## Formula(s) to know cold

PVE for PC $m$ (book formula, ISL 12.10):

$$
\text{PVE}_m = \frac{\sum_{i=1}^n z_{im}^2}{\sum_{j=1}^p \sum_{i=1}^n x_{ij}^2}
\;=\; \frac{\lambda_m}{\sum_{k=1}^p \lambda_k}.
$$

Cumulative PVE through PC $M$:

$$\text{CumPVE}(M) = \sum_{m=1}^M \text{PVE}_m.$$

The PVE-as-$R^2$ identity (ISL 12.11): the variance of the data decomposes as the variance kept by the first $M$ PCs plus the MSE of the $M$-dim approximation, so $\text{CumPVE}(M) = 1 - \text{RSS}/\text{TSS}$ where RSS is the squared reconstruction error.

## Insights & mental models

**The chop-here recipe.** "Here you can see what the first component actually explains … 60-something percent of the variance. And now with two components, you're almost to 90% … that last bit going from the third to the fourth, that was small. So probably you could chop that away." — [[L21-unsupervised-1]] You're looking for the **elbow** where the next PC stops earning its keep.

**Three readings of the same plot.**
1. *Scree plot (PVE per PC)*: look for the "elbow" — the point where the curve goes from steep to flat. Components past the elbow are noise-tier.
2. *Cumulative PVE*: pick a threshold (90 / 95 / 99%), find the $M$ that crosses it. Common in chemometrics and high-dim genomics.
3. *Eyeball*: the prof's own move on USArrests was just to look — "first PC ~62%, two PCs ~87%, third tiny, chop here."

**Two settings, two stopping rules.**
- **Unsupervised PCA** (visualization, exploration): scree elbow / cumulative threshold. Subjective. The book is candid: "this type of visual analysis is inherently *ad hoc* … there is no well-accepted objective way to decide" (ISL §12.2.4).
- **Supervised PCR / NN-feature-extractor**: pick $M$ by **cross-validating the downstream model**. Treat $M$ as a tuning parameter, choose the value that minimizes CV-MSE / CV-misclassification. This is objective and what the prof actually does in [[principal-component-regression|PCR]].

**Total variance = sum of eigenvalues.** Useful sanity check. Under standardization (variance 1 per column), total variance equals $p$, so $\text{PVE}_m = \lambda_m / p$.

**Scree plot magnitudes scale with the data.** On `USArrests` ($p=4$): PC1 ~62%, PC2 ~25%, two PCs cover 87%. On the NYT stories data ($p=4432$, very wide & sparse): the first two PCs explain only a small fraction; you need many more to cover 90%. In wide-data regimes the cumulative-PVE curve rises slowly and the elbow is gentle.

**The prof's saturating-curve heuristic.** On a $p=13$ regression example: 90% threshold lands at ~7 PCs. *"Remember, with model selection, 7 is a lot better than 13 when you're fitting a model."* — [[L14-modelsel-3]] On a $p=300$ example: 95% might give you 75–80 PCs — still a huge reduction.

## Exam signals

> "Probably you could chop that away." — [[L21-unsupervised-1]] (on the third-to-fourth USArrests gap)

> "Number of PCs needed for 90% of total variance → cumulative sum: 0.54 + 0.30 = 0.84 (not enough), +0.10 → past 0.90, so 3 PCs." — [[L27-summary]] (worked threshold-counting calc; verbatim recipe)

> "This kind of question I would also say is fair game because it tests basic knowledge of PCA that we covered in class, with simple calculations. And again, show your work so that you can get partial credit when the inevitable little mistake happens." — [[L27-summary]] (on the PCA explained-variance / scree question)

## Pitfalls

- **PVE is computed *after* standardization.** If you forget to standardize, your eigenvalues are dominated by the largest-unit variable and PC1 will look spuriously dominant. The slide deck demos exactly this on USArrests (`scale = FALSE` vs `scale = TRUE`). See [[principal-component-analysis]] standardize-first warning.
- **Don't confuse the scree plot with the cumulative plot.** Scree = per-PC PVE (decreasing curve, elbow). Cumulative = running sum (increasing, saturating).
- **The "elbow" is subjective.** Two reasonable readers may differ by ±1 PC. Justify your call. For supervised use, defer to CV instead.
- **When all PCs explain similar variance, PCA isn't doing anything for you.** "If the variables were already orthogonal then adding more PCs is just the same as adding another variable." — [[L15-modelsel-4]] Flat scree → no real low-dim structure; rethink.
- **PVE and "$R^2$ of the approximation" are the same number** (ISL 12.11). Don't be surprised if the cumulative PVE at $M=p$ equals 1 — that's just "100% of the variance reconstructed when you keep all components."
- **Cumulative PVE is monotone non-decreasing.** A drop signals a coding bug.

## Scope vs ISLR

- **In scope:** the PVE formula and its scree-plot / cumulative reading; the chop-here heuristic; the contrast with supervised CV-based selection of $M$ in [[principal-component-regression|PCR]].
- **Look up in ISLR:** §12.2.3 (the PVE derivation, including the $R^2 = 1 - \text{RSS}/\text{TSS}$ identity); §12.2.4 ("Deciding how many principal components to use" — the elbow + the supervised-CV alternative).
- **Skip in ISLR:** spectral-decomposition theory of why the eigenvalues are the variances — [[L04-statlearn-3]]: "we don't talk about spectral decomposition" (deferred to Linear Statistical Models). Use the fact, don't derive it.

## Exercise instances

- **Exercise 6.7** — How many PCs for the Credit dataset, justify. Pure scree-plot / cumulative-PVE reading on a small $p = 11$-ish dataset. Two acceptable answers: elbow read, or threshold-based count.
- **Exercise 10.1** — On the NYT stories data, plot PVE and cumulative PVE; decide where to chop. With $p = 4432$ (very wide, sparse word counts), the cumulative curve rises slowly — the headline observation is that even two PCs cover only a small share of the variance, but they still **separate** music from art (the supervised-validation move).

## How it might appear on the exam

- **Threshold counting (Q3e-style):** given five eigenvalue ratios, "how many PCs for 90%?" — sum until you cross.
- **Elbow reading:** given a scree plot, identify the elbow. Justify the cut-point in one sentence.
- **PVE arithmetic:** "the first PC explains 62%, the second 25% — what's the cumulative PVE through PC2?" Trivial sum, but write it out.
- **Output interpretation:** given a printout from `summary(prcomp(...))` (the standard R output is a row of "Proportion of Variance" and a row of "Cumulative Proportion"), read off how many PCs are needed for some threshold.
- **Method-of-selection question:** "would you use a scree plot or cross-validation to choose $M$ here?" Answer keys on whether the downstream task is supervised. PCA-for-visualization → scree. PCR → CV.
- **T-F traps:**
  - "Cumulative PVE is monotone increasing." → **True**.
  - "PVE is invariant to the units of $X$." → **False** (depends on standardization).
  - "If all PCs have similar PVE, PCA has revealed strong low-dim structure." → **False** (the opposite — flat scree means no structure).

## Related

- [[principal-component-analysis]] — defines the eigenvalues this atom plots; this atom is the "now what" companion.
- [[principal-component-regression]] — supervised use of $M$; CV beats scree for this case.
- [[dimensionality-reduction]] — the umbrella; the scree plot is the standard "how much did we lose?" diagnostic.
- [[cross-validation]] — the better way to pick $M$ when there is a downstream model.
