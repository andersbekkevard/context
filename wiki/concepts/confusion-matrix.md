---
concept: confusion-matrix
module: 04-classif
lectures: [L09, L10, L18]
isl-ref: 4.4.2
exercises:
  - Exercise4.6c — compute confusion matrix and overall correct fraction
  - Exercise4.6d — confusion matrix on test set
  - CE1 problem 3c — confusion matrix from logistic regression with 0.5 cutoff
  - CE1 problem 3f — confusion matrix from LDA
  - CE1 problem 3g — confusion matrix from QDA
related: [sensitivity-specificity, roc-auc, classification-setup, logistic-regression, linear-discriminant-analysis, quadratic-discriminant-analysis]
tags:
  - concept
  - module/04-classif
aliases:
  - confusion table
---

# Confusion matrix

A $K \times K$ table cross-tabulating **true class** against **predicted class**. Diagonal = correct, off-diagonal = mistakes. The prof's framing: it's a **diagnostic for *how* the model is failing**, not just *whether*. **In-sample matrices flatter the model** — that's overfitting, and the LDA/QDA contrast (next section) is the prof's favorite illustration.

## Definition (prof's framing)

> "What can happen, this happens a lot, is the true class was K but a bunch got misclassified as 1. This can happen because maybe the two classes are very similar, have high overlap. The confusion matrix is a way to visualize **how the model is getting confused**." — [[L09-classif-3]]

A simple $K \times K$ table:

|              | Predicted = 0 | Predicted = 1 | … | Predicted = $K$ |
|--------------|---------------|---------------|---|-----------------|
| True = 0     | TN            | FP            | … |                 |
| True = 1     | FN            | TP            | … |                 |
| …            |               |               |   |                 |

For binary, the entries get the standard names below.

## Notation & setup

Binary case (positive class = 1, often "disease" or "default"):

|              | $\hat y = 0$ | $\hat y = 1$ |
|--------------|--------------|--------------|
| $y = 0$ (N)  | TN (true neg) | FP (false pos) |
| $y = 1$ (P)  | FN (false neg) | TP (true pos) |

- **P** = $y = 1$ count = TP + FN.
- **N** = $y = 0$ count = TN + FP.
- Total = TP + TN + FP + FN.

Convention varies on whether rows are "true" and columns "predicted" or vice versa. **State your convention.** (Slide deck and the prof's lectures use rows = true.)

## Formula(s) to know cold

**Overall accuracy / error rate:**

$$\text{accuracy} = \frac{TP + TN}{TP + TN + FP + FN}, \quad \text{error} = 1 - \text{accuracy}$$

**Per-class metrics** (more in [[sensitivity-specificity]]):

$$\text{sensitivity} = \frac{TP}{TP + FN}, \quad \text{specificity} = \frac{TN}{TN + FP}$$

## Insights & mental models

- **Diagnostic, not just an aggregate.** A 90% accuracy can hide that one class is 95% missed (under class imbalance). Always look at the off-diagonal pattern, not just the trace.
- **In-sample matrices overfit.** The prof: "in-sample (training) confusion matrices flatter the model. They reflect overfit, not true generalization." — [[L09-classif-3]]
- **The LDA-iris example:** training-set confusion matrices for LDA (error 0.19) and QDA (error 0.17) flip on test data (LDA 0.17, QDA 0.32). The training matrices misled — only test-set matrices tell you about generalization.
- **Class imbalance dominates.** Under heavy imbalance (e.g., 95% class-0), a "always predict 0" classifier scores 95% accuracy — the matrix shows zero TP and zero FP, all FN and TN. Worth looking at the per-class breakdown specifically.
- **Threshold-dependent.** For a probabilistic classifier, the matrix changes with the cutoff — sweeping the cutoff produces the [[roc-auc|ROC curve]]. The default cutoff = 0.5 is just one slice.
- **Multi-class generalization:** $K \times K$ table; diagonal is correct. Off-diagonal block structure tells you which pairs of classes get confused. Sometimes useful to symmetrize / normalize per row.

## Exam signals

> "If I gave you the confusion matrix, you could estimate sensitivity and specificity from it." — [[L27-summary]]

> "Models that are really naive and only predict that it's going to be a zero are already going to do pretty well, because the one class is almost all of the data." — [[L27-summary]]

The 2025 exam Q7 had **multiple confusion-matrix-from-output questions** — interpret a given confusion matrix, compute sensitivity / specificity / error rate. The 2026 exam will follow the same pattern.

## Pitfalls

- **Row/column convention slippage.** State it: "rows = true, columns = predicted" or vice versa. Otherwise sensitivity and specificity flip.
- **Computing accuracy on imbalanced data without checking sensitivity/specificity.** Misleading.
- **Reporting in-sample matrix as if it generalizes.** The classic overfitting trap.
- **Forgetting the cutoff.** Confusion matrix at 0.5 cutoff vs at 0.2 cutoff are different; both are "valid" but encode different trade-offs.
- **Confusing sensitivity and specificity.** Sensitivity = TP / P (rows of true positives, fraction caught). Specificity = TN / N (rows of true negatives, fraction not falsely alarmed). Mnemonic: sensitivity = "Sniffs out positives"; specificity = "Spares the negatives."

## Scope vs ISLR

- **In scope:** Definition of the table, overall accuracy/error, sensitivity, specificity, threshold-dependence, in-sample-vs-out-of-sample distinction, class-imbalance gotcha, multi-class generalization.
- **Look up in ISLR:** §4.4.2, pp. 149–151, Tables 4.4 and 4.5 (LDA on Default with 0.5 vs 0.2 cutoff, side-by-side). §4.4.4, Tables 4.7–4.9 (naive Bayes confusion matrices).
- **Skip in ISLR:** Heavy detail on type I / type II error nomenclature (§4.4.2 Table 4.6) — useful but not exam-relevant beyond knowing sensitivity = 1 − Type II = power.

## Exercise instances

- **Exercise4.6c** — `glm()` on full `Weekly` data; confusion matrix; overall correct fraction; what kinds of mistakes is the model making?
- **Exercise4.6d** — Train on 1990–2008, test on 2009–2010 with `Lag2`-only logistic; held-out confusion matrix.
- **CE1 problem 3c** — Logistic regression with 0.5 cutoff on tennis test set; confusion matrix + sensitivity + specificity.
- **CE1 problem 3f** — Same, for LDA.
- **CE1 problem 3g** — Same, for QDA.

(Exercises 4.6e/f for LDA/QDA implicitly produce confusion matrices too — that's the standard companion to any classifier fit.)

## How it might appear on the exam

- **Read-and-interpret:** Given a 2×2 confusion matrix, compute accuracy, sensitivity, specificity, error rate.
- **Multi-class read:** Given a 3×3 matrix, identify which pair of classes gets most confused, and explain why (high feature overlap, prior asymmetry, etc.).
- **Class-imbalance argument:** Given a confusion matrix on imbalanced data, argue that high accuracy is misleading; bring in sensitivity/specificity.
- **In-sample-vs-out-of-sample:** Given training and test confusion matrices for the same classifier, identify which is which by the error pattern (training tighter); explain.
- **Cutoff sweep:** Given two confusion matrices for the same logistic model at cutoffs 0.5 and 0.2, explain how sensitivity and specificity changed.

## Related

- [[sensitivity-specificity]] — the per-class metrics derived from the confusion matrix.
- [[roc-auc]] — sweep the cutoff, plot, summarize the matrix's threshold-dependent face.
- [[classification-setup]] — 0/1 loss feeds into the matrix.
- [[logistic-regression]], [[linear-discriminant-analysis]], [[quadratic-discriminant-analysis]] — every classifier produces one.
