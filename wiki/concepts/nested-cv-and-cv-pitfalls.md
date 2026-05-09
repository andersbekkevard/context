---
concept: nested-cv-and-cv-pitfalls
module: 05-resample
lectures: [L11]
isl-ref: 5.1.4
exercises:
  - Exercise5.3 — simulate p=5000 random predictors with random labels; show wrong-way CV gives ~0% misclass while right-way gives ~50% (the truth)
related: [k-fold-cv, training-validation-test-split, cross-validation]
tags:
  - concept
  - module/05-resample
aliases:
  - nested cross-validation
  - wrong-way CV
  - selection bias in CV
  - right way to do CV
---

# Nested CV and the wrong-way CV trap

The prof's headline "lying with statistics" trap for module 5: if you preselect predictors using $y$ outside the CV loop, you can get misclassification ≈ 0 on **pure noise**. The fix is **nested CV** — outer folds for assessment, inner folds for selection — and the discipline that **anything that uses $y$ counts as training and must live inside the CV loop.**

## Definition (prof's framing)

**Nested CV** — two layers of cross-validation:

- **Outer split** — partition the data into folds for *assessment*. In each outer iteration, hold one fold out as a true held-out test set.
- **Inner CV on the rest** — within the remaining outer-training data, run another CV (e.g. 5-fold) to do *selection* (pick hyperparameter / variable subset / model class).
- **Score** the chosen model on the held-out outer fold → contributes to the assessment estimate.
- Repeat across outer folds.

> "Using the test set for both model selection and estimation tends to overfit the test data, and the bias will be underestimated." — slide deck, [[L11-resample-2]]

**Wrong-way CV (selection bias)** — performing any step that uses $y$ (correlation filter, variance filter with response, supervised feature selection, even cherry-picking based on a glimpse) **outside** the CV loop, then wrapping CV around the remaining "fitting" step. The held-out fold has already leaked into the selection — CV estimates are uselessly optimistic.

## The right vs wrong way — verbatim slide

> **Q:** "How can we use cross-validation to produce an estimate of our performance? Can we apply cross-validation only to step 2?"
> **A:** *No, you can't.* Step 1 is part of the training procedure (the class labels have already been used) and must be part of the CV to give an honest estimate of the performance of the classifier.
>
> - **Wrong:** Apply cross-validation in step 2.
> - **Right:** Apply cross-validation to **steps 1 and 2.**

— slide deck (Hastie & Tibshirani's example), reframed in [[L11-resample-2]]

## The setup that exposes the trap (Exercise 5.3)

- $n = 50$ (or $100$) observations, $p = 5000$ predictors, all generated as iid standard normals. **No relationship to $y$.**
- $y$ assigned as 50/50 random labels — true Bayes error rate is **50%** (chance).
- Two-step pipeline: (1) compute correlation between each predictor and $y$, keep the top $d = 25$ (or $d = 10$); (2) fit logistic regression on those $d$.

**Wrong-way CV:** filter once on the full data, then 10-fold CV on logistic regression with the chosen 25.

> [!warning] How bad it gets — verbatim
> "In the recommended exercises, one of the exercises is basically create fake data using data where there should be no relationship at all, but by pre-selecting which variables you use, you actually get a misclassification error of zero — suggesting like, this is an excellent model, when in reality we know that it's crap." — [[L11-resample-2]]

(In the slide deck script, "wrong" gets ~20% misclassification — still way too low for the truth, which is 50%. The "almost zero" version is what happens with even more aggressive filtering / smaller $d$.)

**Right-way CV:** the filter step *itself* is redone inside each training fold, on training-fold data only. The held-out fold is genuinely held out.

```
For j = 1, ..., k:
    Define training_j = all folds except j; validation_j = fold j
    On training_j ONLY:
        Compute correlation of each predictor with y_train
        Pick top d predictors → selected_j
        Fit logistic regression on training_j[, selected_j]
    Predict on validation_j[, selected_j]; compute misclassification on fold j
CV estimate = average over folds
```

Result: ~50% misclassification — the truth.

## The general lesson

> "The correlation is already doing a bit of the work for you. It's already a statistical model. You already are selecting parameters based on this for this specific data." — [[L11-resample-2]]

Anything that uses $y$ — including correlation filters, variance filters that look at response stratification, supervised PCA, anything you do "to look at the data" before fitting — is **part of training**. If it lives outside the CV loop, the CV estimate is biased.

> "This is an example of how to lie with — we'll say bad statistics." — [[L11-resample-2]]

The prof's broader point: this mistake is **endemic**:

> "If you look from this year, you'll find at least one article that's made this mistake within genome stuff. And if you look at other fields like neuroscience you'll find it there, you'll find it basically everywhere. People make the same mistakes all the time, especially because they don't take that many statistics classes." — [[L11-resample-2]]

And **subtle**:

> "You maybe make a selection process and then you forget about it. You're like, oh, I did that already, and then you move on and you're looking at performances and you forget that it's no longer valid. I'm sure if I really looked into it, I probably made that mistake at least once. I don't know if I should admit that." — [[L11-resample-2]]

## Why nested CV solves the assessment problem

Once you've used CV to **select** a model, the CV error you used to select can no longer honestly **assess** the chosen model — that estimate is biased downward (you optimized against it).

Nested CV:

- **Inner folds → selection** ("which polynomial degree, which $K$, which $\lambda$").
- **Outer folds → assessment** (an honest generalization estimate).

Each outer iteration: hold out an outer fold, run inner CV on the rest to pick the model, evaluate the *chosen* model on the held-out outer fold. Average outer-fold errors → honest assessment.

> "You actually would be getting a different selection for each fold, but it tends to give you the same selection anyway, so then you could just look at which ones are popping up." — [[L11-resample-2]]

If the inner CV picks the same hyperparameter across most outer folds → fit the final model on all data with that choice; report outer CV's assessment as your honest test error.

## Insights & mental models

- **Nested CV is what you do when you don't want to spend a precious test set.** The classic alternative is: hold out a true test set once, k-fold CV on the rest for selection, evaluate the chosen model on the test set. Nested CV uses every observation for both jobs at the cost of more compute.
- **The two layers map onto the three-partition framing:**
  - Outer fold = "test" (assessment)
  - Inner training folds (within the outer training data) = "training" (fit)
  - Inner validation folds = "validation" (selection)
- **The "anything that uses $y$" rule is sharper than "anything supervised":** even ad hoc steps like "I looked at the data, dropped the obvious outliers based on $y$, then trained" leak information. If you peeked at $y$ to decide, the peek is part of training.
- **The selection-bias version is genomics' poster child** — Hastie and Tibshirani devote a full subsection to it because they've seen genomics studies make this exact mistake repeatedly.

## Exam signals

> "Using the test set for both model selection and estimation tends to overfit the test data, and the bias will be underestimated." — slide deck, [[L11-resample-2]]

> "Step 1 is part of the training procedure (the class labels have already been used) and must be part of the CV to give an honest estimate of the performance of the classifier." — slide deck

> "We will see in the Recommended Exercises that doing the wrong thing can give a misclassification error approximately 0 — even if the 'true' rate is 50%." — slide deck

The prof spent ~1/4 of L11 on this — heavy emphasis. **Highly likely true/false or short-answer territory.**

## Pitfalls

- **The "I already filtered, now I CV" workflow** — exactly the wrong way.
- **Reusing the same CV estimate for both selection and assessment** — even without a separate filter step, this still biases the assessment downward.
- **Forgetting that "supervised" includes informal peeks at $y$.** Removing outliers based on residuals from a preliminary fit, transforming variables based on $y$-stratified plots — all "training."
- **Mistaking nested CV for repeated CV.** Repeated CV reruns the same k-fold many times to reduce noise; nested CV layers two levels for selection + assessment. Different jobs.

## Scope vs ISLR

- **In scope:** the wrong-way trap (Exercise 5.3 + Hastie/Tibshirani slide), nested CV as the conceptual fix, the "anything using $y$ is training" discipline.
- **Look up in ISLR:** §5.1.4, pp. 207 (the bias-variance discussion that motivates this) and the worked example in §5.3. The selection-bias example is also in Elements of Statistical Learning §7.10.
- **Skip in ISLR (book-only, prof excluded):** formal theoretical guarantees for nested CV — the prof gives the conceptual recipe, not the asymptotics.

## Exercise instances

- **Exercise5.3** — the canonical wrong-way demo: $p = 5000$ random predictors, random labels, top-$d$ correlation filter. Show wrong-way CV gives ~0–20% misclass, right-way CV gives ~50%. Includes the schematic-drawing prompt for both pipelines.

## How it might appear on the exam

- **Conceptual / true-false:** "Filtering predictors by correlation with $y$ outside the CV loop gives an unbiased estimate of test error" → false; selection bias → CV estimate biased downward, possibly to ~0% on pure noise.
- **Pipeline interpretation:** given pseudocode of a CV procedure with a filter step outside the loop, identify the bug and rewrite it correctly.
- **"Why use nested CV?"** → because using the same CV for selection and assessment biases assessment downward; nested CV separates the two jobs (inner = selection, outer = assessment).
- **Worked example sketch:** describe the right-way CV pipeline for a feature-selection-then-classify workflow. Pseudocode is fine per the prof's exam policy.
- **The "lying with statistics" framing:** the prof's verbatim emphasis makes this exam-bait. Be ready to explain *why* it lies (selection bias = the held-out data is no longer truly held out).

## Related

- [[k-fold-cv]] — the workhorse used in both layers
- [[training-validation-test-split]] — the three-partition framing nested CV operationalizes when you can't afford a fixed test set
- [[cross-validation]] — global picture of the prof's preferred tuner
