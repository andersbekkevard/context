---
concept: sensitivity-specificity
module: 04-classif
lectures: [L09, L10, L27]
isl-ref: 4.4.2
exercises:
  - Exercise4.5a - define sensitivity and specificity for a disease/non-disease classifier
  - CE1 problem 3c - calculate sensitivity and specificity from a confusion matrix (logistic)
  - CE1 problem 3f - same for LDA
  - CE1 problem 3g - same for QDA
related: [confusion-matrix, roc-auc, classification-setup, logistic-regression]
tags:
  - concept
  - module/04-classif
aliases:
  - true positive rate
  - true negative rate
  - sensitivity
  - specificity
---

# Sensitivity and specificity

The two binary-classification metrics derived from the confusion matrix. Each picks up *one type* of correctness, sensitivity = catching positives, specificity = sparing negatives. The prof's framing emphasizes the **trade-off** between them (justice-system analogy) and the exam-day instruction: **write the formula** even if you don't plug in numbers.

## Definition (prof's framing)

> "Sensitivity is the proportion of correctly classified positive observations: $\text{TP}/\text{P}$." - slide deck

> "Specificity is the proportion of correctly classified negative observations: $\text{TN}/\text{N}$." - slide deck

In words:
- **Sensitivity** = "of all *actual* positives, how many did we catch?": ability to detect disease/default/whatever.
- **Specificity** = "of all *actual* negatives, how many did we correctly leave alone?": ability to avoid false alarms.

## Notation & setup

Standard binary confusion-matrix entries (see [[confusion-matrix]]):

|              | $\hat y = 0$ | $\hat y = 1$ |
|--------------|--------------|--------------|
| $y = 0$ (N)  | TN           | FP           |
| $y = 1$ (P)  | FN           | TP           |

- **P** = TP + FN (total actual positives).
- **N** = TN + FP (total actual negatives).

## Formula(s) to know cold

$$\text{sensitivity} = \frac{TP}{TP + FN} = \frac{TP}{P} = \text{true positive rate (TPR)} = \text{recall}$$

$$\text{specificity} = \frac{TN}{TN + FP} = \frac{TN}{N} = \text{true negative rate (TNR)}$$

$$1 - \text{specificity} = \frac{FP}{TN + FP} = \text{false positive rate (FPR)}$$

These two quantities are what the [[roc-auc|ROC curve]] plots against each other (TPR on $y$, FPR = $1 - $ specificity on $x$, threshold-swept).

> "On the exam, *write the formula* even if you don't plug in numbers." - manifest one-liner

The slide deck shows the canonical 2×2 layout, memorize the formula, then read entries off the matrix. The prof: "It's why you need a calculator."

## Insights & mental models

- **Trade-off:** higher sensitivity usually comes at lower specificity (and vice versa). The cutoff (default 0.5 for binary classifiers) is the knob.
- **Prof's justice-system analogy:**

> "Typically in the justice system, when we're convicting people, we want a bias towards not putting them in jail. So we're okay letting a few people who committed the crime walk free, because then it keeps the innocent people from going to jail. So if you want to put a lot of people in jail, you want a very sensitive way, your jury reacts in a very sensitive manner, very easily convinced. Whereas if you want to bias towards more specificity, you want them to be more likely to just get all the negative ones right." - [[L09-classif-3]]

- **Domain sets the priority.** Medicine: usually want high sensitivity ("don't miss the disease"). Spam filter: usually want high specificity ("don't quarantine the boss's email").
- **Both depend on the cutoff.** For a probabilistic classifier, decreasing the cutoff (e.g., 0.5 → 0.2) → more positive predictions → sensitivity ↑, specificity ↓. The ROC curve traces every $(1-\text{spec}, \text{sens})$ pair.
- **Class imbalance distorts overall accuracy but not sensitivity/specificity.** A "always predict 0" classifier on heavy class-0 data has high accuracy, sensitivity = 0, specificity = 1, sens/spec correctly catch the failure.
- **The sensitivity-specificity pair is what doctors look at.** Slide deck: "in medicine for two-class problems logistic regression is often preferred (for interpretability) and (always) together with ROC and AUC (for model comparison)."

## Exam signals

> "Write the equation rather than computing, the formula counts as the answer." - [[L27-summary]]

> "Define sensitivity and specificity in plain English **for this specific model**." - [[L27-summary]] (re: 2025 exam Q7)

> "Models that are really naive and only predict that it's going to be a zero are already going to do pretty well, because the one class is almost all of the data. … Discuss in terms of sensitivity/specificity vs. error rate." - [[L27-summary]]

The 2025 exam Q7 explicitly asked: define sensitivity/specificity for the default-prediction setting **in plain English**, compute from the confusion matrix at the 0.5 cutoff. Both LDA and KNN versions of this question appeared.

## Pitfalls

- **Confusing the two.** "Sniffs out positives" / "Spares the negatives" mnemonic. Sensitivity has TP in the numerator; specificity has TN.
- **Computing $TP/(TP+FP)$ instead of $TP/(TP+FN)$.** That's **precision** (positive predictive value), not sensitivity. Different beast.
- **Forgetting that 1 - specificity = FPR.** The ROC curve uses 1 - specificity, not specificity, on the $x$-axis.
- **Reporting accuracy when the question asks about sens/spec.** Easy under time pressure; the prof flagged this.
- **Forgetting to define what "positive" means** in domain terms. "Sensitivity = ability to identify defaulters" makes the formula meaningful; "$TP/(TP+FN)$" alone leaves the grader wondering whether you understand.
- **Convention clash.** Some sources call sensitivity "recall." Statisticians prefer sensitivity.

## Scope vs ISLR

- **In scope:** Definitions, formulas, trade-off, threshold-dependence, justice-system intuition, class-imbalance interpretation, role in ROC.
- **Look up in ISLR:** §4.4.2, pp. 149–151, Table 4.6 (the full type-I/type-II + sensitivity/specificity vocabulary cross-reference).
- **Skip in ISLR:** Detailed type-I/type-II error connections (knowing sensitivity = 1 − Type II = power is enough); precision/recall/F1 from the information-retrieval tradition (never covered).
- **Imbalanced-class asymmetric ROC analysis**: [[L07-classif-1]]: "I don't think the book talks much about that." Out of scope.

## Exercise instances

- **Exercise4.5a**: define sensitivity and specificity for a disease/non-disease classifier. Plain-English plus formula.
- **CE1 problem 3c**: compute sensitivity + specificity from the logistic-regression confusion matrix on the tennis test set.
- **CE1 problem 3f**: same for LDA.
- **CE1 problem 3g**: same for QDA.
- **Exercise4.6j**: implicitly used inside the ROC plotting (sensitivity = $y$-axis).

## How it might appear on the exam

- **Calculator-MCQ:** Given a 2×2 confusion matrix with specific TP/TN/FP/FN counts, compute sensitivity and specificity. The 2025 exam did this with `default` data.
- **Plain-English definition:** "What is sensitivity in this context?" → "the fraction of true defaulters that the model correctly identifies as defaulters."
- **Formula-only:** "Write the formula for specificity from a confusion matrix" → $TN/(TN + FP)$. Half-credit even without the data.
- **Trade-off T/F:** "Increasing the classifier threshold from 0.5 to 0.7 always increases specificity" → true (fewer positive predictions → fewer FP → higher TN/N). "And always decreases sensitivity" → true (also fewer TP → lower TP/P).
- **Class-imbalance interpretation:** Given an imbalanced confusion matrix where the model has 95% accuracy but sensitivity = 0.05, explain why this is bad (model is essentially predicting all-zero; misses almost every positive).
- **Method-comparison:** Given a table of sensitivity/specificity for several classifiers on the same test set, pick the best one and justify.

## Related

- [[confusion-matrix]]: the source table.
- [[roc-auc]]: sweep the cutoff, plot $(1 - \text{spec}, \text{sens})$, summarize with AUC.
- [[classification-setup]]: 0/1 loss is the aggregate; sensitivity/specificity is the per-class breakdown.
- [[logistic-regression]]: the canonical companion in the prof's medical examples.
