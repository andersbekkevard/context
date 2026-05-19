---
moc: 05-resample
title: "M05: Resampling"
lectures: [L10, L11]
isl-ch: 5
slides:
  - modules/5Resample/5Resample.md
exercises:
  - exercises/Exercise5/
  - exercises/compulsory-exercise-1.md
tags:
  - moc
  - module/05-resample
---

# Module 05: Resampling

The prof's "fewer-assumptions" alternative to AIC/BIC for model selection and assessment. Two lectures (Feb 9, Feb 10) that establish [[cross-validation]] as the workhorse hyperparameter tuner, the [[bootstrap]] as the empirical sampling-distribution machine, and the right-vs-wrong-way CV trap as a flagged "lying with statistics" exam target. Foundation for everything in modules 6–11.

## Slides
- <a href="/pdfs/m05-resample-slides.pdf" target="_blank" rel="noopener">m05-resample-slides.pdf</a>

## Lectures
- [[L10-resample-1]]: module 4 recap (sens/spec/ROC), train/val/test split, validation-set / LOOCV / k-fold CV, one-SE rule, independence trap
- [[L11-resample-2]]: CV as bias-variance, nested CV, the wrong-way-CV trap, the bootstrap, bagging preview

## Concepts (atoms in this module)
- [[training-validation-test-split]]: three partitions, three jobs (fit / select / assess); reusing test makes you "too optimistic"
- [[validation-set-approach]]: random ~50/50 split; high variance across splits, biased upward; conservative and easy to explain
- [[leave-one-out-cv]]: train on n−1, hold out 1, repeat n times; low bias, high variance; OLS hat-matrix shortcut needs only one fit
- [[k-fold-cv]]: split into k blocks, hold each out once; k=5 or 10 is the bias-variance compromise; the prof's preferred everything-tuner
- [[one-standard-error-rule]]: among models within 1 SE of the CV minimum, pick the simplest; prof's preferred selection criterion
- [[nested-cv-and-cv-pitfalls]]: outer folds assess, inner folds select; variable selection with the labels lives *inside* the CV loop, the canonical "lying with statistics" trap
- [[aic-bic-conceptual]]: training-error penalties for complexity; **derivations explicitly out of scope**; prof distrusts them and prefers CV
- [[bootstrap]]: resample with replacement, histogram the statistic; "your best model for the world is the data itself"; B = 1000–10000
- [[bagging]]: fit B models on B bootstrap samples and average/vote; variance reduction with floor ρσ² + (1−ρ)σ²/B; precursor to random forests
- [[out-of-bag-error]]: ~1/3 of obs (1 − 1/e ≈ 0.368) are excluded from each bootstrap sample → free per-tree validation set

## Cross-cutting concepts touched (Specials)
- [[bias-variance-tradeoff]]: first introduced module 02; this module revisits in [[L11-resample-2]] to frame validation-set vs LOOCV vs k-fold as the three points on the bias-variance curve for CV schemes
- [[cross-validation]]: this module is its canonical home (mechanics, variants, pitfalls); referenced from every later module that tunes a hyperparameter
- [[regularization]]: touched in [[L11-resample-2]] via bagging as an *implicit* variance-reducing regularizer; full treatment in module 06
- [[double-descent]]: first introduced module 02; revisits in [[L11-resample-2]] via the bagging connection ("over-parameterized single model wins because it's bagging implicitly")

## Exercises
- [[../../exercises/Exercise5]]: five-problem drill on the whole module: describe k-fold CV (5.1), compare to validation/LOOCV (5.2), the wrong-way-CV simulation on p=5000 random predictors (5.3), bootstrap probability 1 − 1/e (5.4), bootstrap algorithm + implementation for OLS SE (5.5–5.6)
- [[../../exercises/compulsory-exercise-1]]: problem 4 is the module's compulsory: 4a writes 10-fold CV pseudocode for KNN regression, 4b is true/false on LOOCV vs k-fold bias/variance, 4d is a B=1000 bootstrap for SE and 95% CI of a logistic-regression-derived probability

## Out of scope (this module)
- **Cp / AIC / BIC / adjusted-R² derivations and formulas** - *"I'm not going to ask you to use these. I'm not going to ask you to derive them."* - [[L10-resample-1]]; the [[aic-bic-conceptual]] atom keeps only "they exist, here's why, prof distrusts them"
- **Fancy resampling-shortcut algebra beyond the LOOCV hat-matrix formula**: the prof shows the (yᵢ − ŷᵢ)/(1−hᵢᵢ) shortcut and stops there; no jackknife / .632 / .632+ derivations

## ISLP pointer
Chapter 5: Resampling Methods. The deep treatment of in-scope concepts in this module is in `wiki/book/05-resample.md`. Specific atoms carry section-level `isl-ref:` pointers; for full algebra of any in-scope concept (LOOCV shortcut derivation, k-fold bias-variance argument, bootstrap CI variants), route Anders to `wiki/book/05-resample.md`.
