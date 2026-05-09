---
moc: 12-final
title: Summary and Exam Review
lectures: [L27]
isl-ch: null
slides:
  - modules/12Final/12Final.md
exercises: []
tags:
  - moc
  - module/12-final
  - exam
---

# Module 12: Summary and Exam Review

Meta-router. Module 12 has **no atoms of its own**, it is the exam-review session, and all exam-relevant content lives in atoms owned by earlier modules. This MOC routes to the canonical exam-review lecture, the scope authority, and the cross-cutting Specials any module-12 query will pass through.

## Lectures
- [[L27-summary]]: Apr 28 dedicated exam Q&A: logistics, the scope rule (verbatim), problem-by-problem walkthrough of the 2025 paper showing how each question is reformatted for the 2026 open-book exam, plus the "mathy question" template (MLE = LS under Gaussian errors)

## Scope authority
- [[scope]]: the canonical "is X in scope?" reference: source hierarchy (exercises > lectures > slides; ISLP for fleshing out, not for scoping), explicit exclusions with verbatim anchors, programming policy, 2026 question patterns, past-exam translation table, exam logistics

## Cross-cutting concepts touched (Specials)

These are the six cross-module atoms that any module-12 query will likely route through. Each is owned by no single module, threading through the whole course.

- [[bias-variance-tradeoff]]: promised exam question per [[L27-summary]] ("definitely going to be a question"); the derivation (CE1 problem 1b) is the canonical "mathy" template
- [[regularization]]: the prof's "central trick of statistical learning"; ridge / lasso / weight decay / dropout / smoothing-spline λ / tree pruning α / implicit SGD all unify here
- [[cross-validation]]: the prof's preferred hyperparameter selector everywhere; right-vs-wrong-way is a flagged trap
- [[standardization]]: mandatory pre-processing before ridge, lasso, PCA, PCR, k-means, hierarchical, KNN, NNs; one-line diagnosis for "results look weird"
- [[multivariate-normal]]: foundation for OLS sampling distribution and LDA/QDA class-conditionals; CE1 problem 1g matches contour plots to Σ
- [[double-descent]]: prof's hobbyhorse; explains why "tradeoff" framing is incomplete; recurs in L04, L11, L13, L24, L26

## Highest-leverage atoms for exam prep

Drawn from [[L27-summary]]'s walkthrough plus the signals catalogued in [[scope]]. These are the atoms most likely to anchor an exam question:

- [[least-squares-and-mle]]: the L27-flagged "mathy" template (Gaussian → MLE = LS)
- [[ridge-regression]], [[lasso]]: Q3a, Q6b on the 2025 paper; bias-variance reasoning applied
- [[logistic-regression]], [[odds-and-log-odds]]: Q3c (compute odds ↔ probability) + Q7 (interaction trap on odds-ratio)
- [[linear-discriminant-analysis]], [[discriminant-score-and-decision-boundary]]: "this would be a typical exam question," said twice in L09
- [[categorical-encoding-and-interactions]]: Q2 and Q7 interaction trap (main-effect coefficient is local to interacting variable = 0)
- [[smoothing-splines]]: Q3d direction-of-effect trap (higher λ → smoother, not wigglier)
- [[principal-component-analysis]], [[explained-variance-and-scree-plot]]: Q3e cumulative-PVE hand calculation
- [[hierarchical-clustering]]: Q5 by-hand dendrogram with complete linkage ("−1 point per mistake" warning)
- [[nn-parameter-count]]: Q3b explicit prof-flagged calculation (don't forget biases)
- [[sensitivity-specificity]], [[roc-auc]], [[confusion-matrix]]: Q7 cont; write the formula even when not plugging in numbers
- [[random-forest]], [[boosting]], [[gradient-boosting]]: Q6c, Q7 cont; tune hyperparameters by CV, interpret why they beat linear models

## Out of scope

This module inherits the whole course's exclusion list. See [[scope]] §"Explicit out-of-scope" for the full catalogue with verbatim anchors. Highlights include: SVM (entire ISLP ch. 9), survival analysis, multiple-testing corrections, AIC/BIC/Cp derivations, F-test mechanics, Bayesian-prior interpretation of ridge/lasso, natural-spline basis math, detailed boosting pseudocode, advanced NN internals (skip connections, Adam, BatchNorm, LSTM/GRU gates, universal-approximation proof), and **all R/Python package names and executable code**.

## ISLP pointer

No single chapter, as module 12 is integrative. For deep treatment of any in-scope topic, route to the chapter owned by the relevant earlier module. ISLP is open-book on the exam; the A5 sheet should hold what ISLP's index doesn't get you to fast (interaction-trap reminders, direction-of-effect cheats, parameter-count formulas).
