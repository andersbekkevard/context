---
concept: roc-auc
module: 04-classif
lectures: [L09, L10, L27]
isl-ref: 4.4.2
exercises:
  - Exercise4.5b  -  explain how to construct a ROC curve and why
  - Exercise4.5c  -  define AUC, choose between p(x) (AUC=0.6) and q(x) (AUC=0.7)
  - Exercise4.6j  -  plot ROC and compute AUC for glm/lda/qda/knn on Weekly
related: [sensitivity-specificity, confusion-matrix, logistic-regression, linear-discriminant-analysis, classification-setup]
tags:
  - concept
  - module/04-classif
aliases:
  - ROC curve
  - AUC
  - receiver operating characteristic
---

# ROC curve and AUC

The graphical companion to [[sensitivity-specificity]] for binary classifiers: sweep the threshold from 0 to 1, plot $(1-\text{specificity}, \text{sensitivity})$, see the full trade-off. **AUC** = area under the curve, summary number in $[0, 1]$, bigger is better, 0.5 = chance, < 0.5 = invert your classifier. Standard tool for **comparing classifiers** that produce probabilities.

## Definition (prof's framing)

> "The receiver operating characteristics (ROC) curve gives a graphical display of the sensitivity against (1-specificity), as the threshold value (cut-off on probability) is moved from 0 to 1.", slide deck

> "The ROC curve is drawn by calculating the true positive rate (TPR) and false positive rate (FPR) at every possible threshold (in practice, at selected intervals), then plotting TPR (on the $y$ axis) over FPR (on the $x$ axis).", 2025 exam solution / class material

**AUC**: the area under the ROC curve, in $[0, 1]$. "A higher value indicates a better classifier… useful for comparing the performance of different classifiers.", slide deck.

## Notation & setup

- $y$-axis: **sensitivity** = TPR = $\frac{TP}{TP + FN}$ (range $[0, 1]$, bigger = better at catching positives).
- $x$-axis: **1 − specificity** = FPR = $\frac{FP}{TN + FP}$ (range $[0, 1]$, smaller = better at sparing negatives).
- Each point on the curve corresponds to one threshold $t$.
- The classifier produces a continuous score $\hat p(x)$; predict positive iff $\hat p(x) > t$.

## Reference points on the plot

- **Origin (0, 0):** threshold = 1, never predict positive. TPR = 0, FPR = 0.
- **Top-right (1, 1):** threshold = 0, always predict positive. TPR = 1, FPR = 1.
- **Top-left (0, 1):** the *ideal*, perfect classifier. TPR = 1, FPR = 0. AUC = 1.
- **Diagonal:** chance / coin-flip classifier. AUC = 0.5.
- **Below diagonal:** worse than chance, invert the classifier (flip predictions) to get above the diagonal.

## Formula(s) to know cold

**TPR (sensitivity) and FPR (1 − specificity)** at threshold $t$:

$$\text{TPR}(t) = \frac{\#\{i : \hat p(x_i) > t \text{ and } y_i = 1\}}{\#\{i : y_i = 1\}}$$

$$\text{FPR}(t) = \frac{\#\{i : \hat p(x_i) > t \text{ and } y_i = 0\}}{\#\{i : y_i = 0\}}$$

**AUC** (interpretation): probability that the classifier scores a randomly-chosen positive higher than a randomly-chosen negative.

$$\text{AUC} = \Pr(\hat p(X^+) > \hat p(X^-))$$

In practice computed via the trapezoidal rule under the ROC curve (or `pROC::auc()` if you must, but on the exam, math/pseudocode only).

## Insights & mental models

- **AUC compares classifiers across all thresholds at once.** No commitment to a particular operating point. You can pick the cutoff later based on the cost of FP vs FN.
- **Hugging the top-left corner = ideal.** "An ideal classifier will give a ROC curve which hugs the top left corner.", slide deck.
- **Below the diagonal → invert.** A classifier with AUC = 0.3 is genuinely informative, just predicting backwards. Flip predictions to get AUC = 0.7.
- **Independent of class prevalence (mostly).** ROC + AUC don't change with class imbalance (unlike accuracy). That's why they're the doctors' tool: stable across populations.
- **Cutoff = 0.5 sits somewhere on the curve.** It's a single point, the default operating point. The slide deck walks through this for the SAheart logistic regression: at 0.5 cutoff, sensitivity = 0.477, specificity = 0.814, so the cutoff sits at $(0.186, 0.477)$ on the ROC plot.
- **AUC of 0.5 = chance, 0.7 = "OK," 0.8 = "good," 0.9+ = "very good."** The slide deck cites AUC = 0.95 on Default LDA as "close to the maximum of 1.0, so would be considered very good."
- **Logistic regression and LDA give nearly identical ROC curves on the same data** (slide deck: "virtually indistinguishable"), the linear log-odds form is the shared driver.

## Exam signals

> "He'll show the ROC curve and ask what it means / how to interpret it." - [[L27-summary]]

> "The ROC-curve is made up of all possible cut-offs and their associated sensitivity and specificity.", slide deck

> "AUC is useful for comparing the performance of different classifiers.", slide deck

The 2025 exam Q7 explicitly asked: explain what the ROC curve illustrates, in particular what the two axes represent. Standard interpretation question.

## Pitfalls

- **Confusing the axes.** $y$ = sensitivity (TPR), $x$ = 1 − specificity (FPR). Some software plots specificity on $x$ (`legacy.axes=FALSE` in `pROC`), easy source of confusion. **State your axis convention.**
- **Comparing AUCs without checking they were computed on the same test set.** AUC depends on the data; comparisons are only meaningful for the same test set (or properly cross-validated).
- **Reporting AUC on training data.** Same overfitting issue as confusion matrices, in-sample AUC overestimates generalization.
- **Reading AUC = 0.5 as "useless."** It's actually as informative as the prior, just no better than chance. AUC < 0.5 means actively worse, but flippable.
- **Sweeping the threshold past the data range.** ROC is constructed at thresholds spanning the actual $\hat p$ values produced by the classifier. Going beyond just stays at the corners.

## Scope vs ISLR

- **In scope:** Construction of the curve (sweep threshold, plot TPR vs FPR), AUC definition + interpretation, reference points (origin, top-right, top-left, diagonal), use for classifier comparison.
- **Look up in ISLR:** §4.4.2, p. 152, Figure 4.8 (LDA ROC on Default). Table 4.6 for the type-I/II vocabulary if you need it.
- **Skip in ISLR:** Precision-recall curves, F1 scores, lift/gains charts (information-retrieval-style metrics), never covered.
- **Imbalanced-class asymmetric ROC analysis** - [[L07-classif-1]]: "I don't think the book talks much about that." Out of scope.

## Exercise instances

- **Exercise4.5b**: explain how to construct a ROC curve, and why it's useful (justify the threshold sweep).
- **Exercise4.5c**: define AUC; given two methods $p(x)$ and $q(x)$ with AUC 0.6 and 0.7 on independent validation sets, which to prefer? (Answer: $q(x)$, higher AUC, same data.)
- **Exercise4.6j**: plot ROC curves for `glm`/`lda`/`qda`/`knn` on the `Weekly` data, compute AUC for each. Standard four-classifier comparison.

## How it might appear on the exam

- **Output interpretation:** Given a ROC plot, describe what the axes represent, where the cutoff = 0.5 point would lie, what AUC means.
- **Method comparison:** "Two classifiers have AUC 0.65 and 0.78 on the same test set. Which is better?" → the second.
- **Construction T/F:** "ROC plots sensitivity against specificity" → false (it's against 1 − specificity / FPR). "An AUC of 0 means the classifier is useless" → false (AUC = 0.5 means useless; AUC = 0 means perfectly inverted, just flip).
- **Pseudocode:** "Write pseudocode for constructing a ROC curve given $\hat p(x_i)$ and $y_i$ for $i = 1, \ldots, n$." → for each threshold $t \in \{0, 0.01, \ldots, 1\}$, compute sensitivity and specificity, plot point.
- **AUC interpretation:** "What does AUC = 0.7 mean?" → with probability 0.7, a random positive will be scored higher than a random negative.
- **Threshold-trade-off discussion:** For a medical-disease classifier, where on the ROC curve would you operate, and why? (High sensitivity for serious disease, operate near top-right; high specificity for spam filter, operate near origin.)

## Related

- [[sensitivity-specificity]]: the per-threshold metrics that get plotted.
- [[confusion-matrix]]: what each ROC point comes from.
- [[logistic-regression]]: the canonical probabilistic classifier; ROC is its standard performance summary.
- [[linear-discriminant-analysis]]: also probabilistic, also ROC-able; nearly identical curve to logistic on most datasets.
- [[classification-setup]]: threshold = 0.5 is just the Bayes-classifier default; ROC sweeps every alternative.
