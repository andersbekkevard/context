---
concept: training-validation-test-split
module: 05-resample
lectures: [L10]
isl-ref: 5.1
exercises: []
related: [validation-set-approach, k-fold-cv, cross-validation, nested-cv-and-cv-pitfalls]
tags:
  - concept
  - module/05-resample
aliases:
  - train-validation-test
  - three-way split
---

# Training / validation / test split

The prof's foundational framing for everything in modules 5–11: **three partitions, three jobs.** Reusing one for another job makes you "too optimistic" — the cardinal sin that motivates the entire resampling module.

## Definition (prof's framing)

Three disjoint subsets of the data with three distinct jobs:

- **Training set** — fit the model.
- **Validation set** — select among candidate models / pick hyperparameters (model selection).
- **Test set** — report final performance (model assessment). *"This is the assessment you want to showcase."* — [[L10-resample-1]]

> "Yeah, it makes sense though, that we would actually need three because these are different goals — to be able to select the model and then also say how good it is." — [[L10-resample-1]]

## Notation & setup

- Common ratios: 60/20/20 or 50/25/25 with abundant data; 80/20 train+test then CV inside the 80% otherwise.
- Test set is **set aside once** at the start and not touched until the very end.
- "Validation" = used to choose between candidates. "Test" = used only to report.

## Insights & mental models

- **The data-rich case:** if you have lots of data, blindly partition into thirds and skip resampling tricks — module 5 is for the (more common) case where you can't afford to throw away 1/3 of the data on a fixed validation block. *"Module 5 is about how to do this efficiently, how to create a test and validation set efficiently."* — [[L10-resample-1]]
- **The psychological resistance:** people refuse to toss 1/3 of their data because *"somehow the extra 20% is going to tell us more or something. It generally doesn't."* — [[L10-resample-1]]
- **Why three not two**: in earlier modules we only had train + test because we hadn't yet introduced model *selection*. Once you're tuning hyperparameters or comparing methods, the selection step needs its own data.
- **CV replaces the validation set, not the test set.** The classic pipeline becomes: (i) split off the test set once; (ii) on the rest, run k-fold CV to pick the model; (iii) refit on all of that "rest" with the chosen hyperparameters; (iv) report performance on the held-out test set. The validation/CV step lives entirely inside the non-test data.

## Exam signals

> [!important] The data-reuse principle (verbatim from L10)
> "We will be too optimistic if we report the error on the test set when we have already used it to choose the best model… Don't do that. I have lots of examples of where people have done that. It's very sad. It's very common. It kind of sucks." — [[L10-resample-1]]

The prof followed with the anecdote about a paper that committed exactly this sin, was rejected, and (annoyingly) put his mother's name on it. *"Don't make the dumb mistakes. Because it's embarrassing."* — [[L10-resample-1]]

## Pitfalls

- **Reusing the test set for selection** → biased-low performance estimate. The whole module pivots on this.
- **Tuning on the test set "just a little"** still counts. Even one peek invalidates the assessment.
- If you go to the test set and the result disappoints, you cannot go back, change the model, and re-test. That's the moment of truth for the chosen model — see [[nested-cv-and-cv-pitfalls]].

## Scope vs ISLR

- **In scope:** the three-partition framing, the data-reuse principle, the motivation for k-fold CV as an efficient validation-set substitute when data is limited.
- **Look up in ISLR:** §5.1 (intro paragraphs to chapter 5) gives the same framing more compactly. The whole of ch. 5 is the operationalization via CV.

## Exercise instances

No exercise drills the partition concept directly — it's foundational scaffolding for every CV / regularization / boosting exercise that follows. The closest is CE1 problem 4a (10-fold CV for KNN), which assumes the partition discipline.

## How it might appear on the exam

- **True/false** on data reuse: "It's fine to use the test set both to pick a hyperparameter and to report final performance" → false; reusing test for selection biases the assessment downward.
- **Conceptual** justification: "Why do we need three partitions instead of two?" → because model selection and model assessment are different jobs; reusing the same set for both inflates the reported performance.
- **Setup prompt** for a more involved CV question: "You have 1000 observations and want to compare ridge / lasso / OLS. Sketch the data-handling pipeline." Expected answer: hold out a test set (say 200), run k-fold CV on the other 800 to pick λ for ridge and lasso, refit each method on all 800 with chosen λ, evaluate on the test 200.

## Related

- [[validation-set-approach]] — the simplest one-split realization
- [[k-fold-cv]] — the practical alternative when data is limited
- [[leave-one-out-cv]] — extreme of k-fold
- [[nested-cv-and-cv-pitfalls]] — what to do when you need both selection AND assessment
- [[cross-validation]] — global picture of the prof's preferred tuner
