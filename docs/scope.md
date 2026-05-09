# Scope: what's on the exam, what isn't

Canonical reference for TMA4268 final exam (2026-05-18). The single authority for "is X in scope?"

Source of truth: the prof's verbatim statements in lectures, especially [[../wiki/lectures/L27-summary]] (the Apr 28 dedicated exam-Q&A session), plus what he actually covered or excluded across the semester.

This file supersedes the scope-related sections of `exam_analysis.md` (§1, §2, §3, §5). `exam_analysis.md` remains useful for tier rankings, direction-of-effect traps, dataset templates, procedural templates, and opinionated takes, but for "is X in scope?" load this file.

## The rule

> "If it was covered in the slides or the exercises, it's fair game. If it's only in the book and we didn't talk about it in class or exercises, it won't be on the test." - [[../wiki/lectures/L27-summary]]

Strengthened by the prof's own emphasis:

> "Especially the exercises." - [[../wiki/lectures/L27-summary]]

## Source hierarchy

The four corpora play distinct roles. Get the role right and the wiki and tutoring stay calibrated.

### For determining what's in scope

In order of weight:

1. **Exercises** (`exercises/Exercise2/` through `exercises/Exercise11/` plus `compulsory-exercise-1.md` and `compulsory-exercise-2.md`): *highest* signal. If a concept is drilled here, it is exam-bait. Compulsory exercises especially: the prof picked these as required, you've solved them, they are the closest analog to what an exam problem looks like.
2. **Lectures** (`wiki/lectures/`): what the prof actually said. If he treated a concept as a topic, it's in scope. If he flagged it as exam-relevant, it's load-bearing. His take on ideas, and the concepts he is most interested in or has strong opinions on, are highly relevant.
3. **Slides** (`modules/`): the deck content. Anything on the slides that he kept (didn't skip in lecture) is in scope, even if he didn't dwell on it.
4. **ISLR** (`book/`): does **not** determine scope. Material that is *only* in the book and never reached lecture, slides, or exercises is OUT.

### For fleshing out IN-SCOPE ideas

Once a concept is filtered as in-scope, **ISLR is the most concise authoritative definition.** Anders will read it himself, and atoms point Claude at the relevant section so it can say "for the full derivation, see ISLR §6.2.1."

So ISLR's role is the inverse of scope-determination:
- Lectures + exercises filter *what part of ISLR is relevant.*
- Within that filtered scope, ISLR is the deep-treatment reference.

Atoms therefore carry an `isl-ref:` pointer per concept, and Claude routes Anders to the book for full mechanics.

### For exam-style question patterns

These do NOT determine scope, but DO inform what kind of questions to expect and practice on:

- **ISLR end-of-chapter exercises**: useful drill material for hand-calculation skills (the conceptual ones; skip the R-coding ones).
- **Past exams** (`exams/TMA4268_2023_Exam.Rmd` etc.): useful with translation. The 2026 exam has been redesigned (open-book, no code, more interpretation), so old papers help mostly for question *style* and which topics recur. The translation table is below.

For both of these: Given that a concept is in scope, these are good sources of exercises for said concept.

## Explicit out-of-scope

These are concepts the prof verbally excluded or never covered. Sourced to verbatim signals where possible.

### Whole topics excluded
- **R/Python package names, function syntax, executable code** - [[../wiki/lectures/L27-summary]]: "no language, no memorizing package names, no language-specific coding."
- **SVM (entire ISLR ch. 9)** - [[../wiki/lectures/L22-unsupervised-2]] / [[../wiki/lectures/L27-summary]]: "I was going to talk about it, but then we didn't, and it's fine. I don't think it's that interesting."
- **Survival analysis** (ISLR ch. 11, Kaplan-Meier, Cox PH): never covered.
- **Multiple testing corrections** (ISLR ch. 13, Bonferroni / FDR): never covered.
- **Time series modeling**: never covered.
- **Multi-class logistic regression** - [[../wiki/lectures/L07-classif-1]]: "we're not going to talk about... discriminant analysis and KNN can deal with this case."
- **Probit, complementary log-log, other GLM link functions** - [[../wiki/lectures/L07-classif-1]]: "outside the scope of this course."

### Concept may be in scope; the algebra/derivation is not
- **AIC / BIC / Cp derivations** - [[../wiki/lectures/L12-modelsel-1]] / [[../wiki/lectures/L13-modelsel-2]]: "I really don't think I'm going to ask any questions about this." The conceptual claim "penalize complexity" stays in scope; the algebra does not.
- **F-test mechanics** - [[../wiki/lectures/L06-linreg-2]]: "won't ask any questions about an F-test."
- **Bayesian interpretation of Ridge/Lasso (Gaussian/Laplace priors)** - [[../wiki/lectures/L14-modelsel-3]]: "I really don't think I'd put this on the test."
- **Spectral / eigen decomposition derivations of covariance** - [[../wiki/lectures/L04-statlearn-3]]: "we don't talk about spectral decomposition", deferred to Linear Statistical Models.
- **Moore-Penrose pseudoinverse details** - [[../wiki/lectures/L08-classif-2]]: "explicitly bracketed off."
- **Natural-spline basis math** - [[../wiki/lectures/L16-beyondlinear-1]]: "the book doesn't [derive], so I won't either."
- **Detailed boosting pseudocode (line-by-line)** - [[../wiki/lectures/L27-summary]]: concept matters, pseudocode memorization does not.
- **Stochastic gradient boosting / XGBoost / LightGBM internals** - [[../wiki/lectures/L20-boosting-2]]: mentioned as Kaggle winners, not derived.
- **Computational complexity of trees** - [[../wiki/lectures/L17-trees-1]]: noted as NP-hard but not exam-relevant.
- **Elastic Net detailed tuning**: concept noted, no worked example.
- **PLS history and detailed mechanics** - [[../wiki/lectures/L15-modelsel-4]]: PCR is the workhorse.

### Neural-network specific exclusions
- **Skip connections, intra-layer connections** - [[../wiki/lectures/L27-summary]]: "you just wouldn't need to know that."
- **Advanced optimizers** (momentum, Adam internals) - [[../wiki/lectures/L23-nnet-1]] / [[../wiki/lectures/L27-summary]]: out of scope.
- **Full RNN/CNN architectures** (LSTM/GRU gates, BPTT, ResNet, Transformer details) - [[../wiki/lectures/L26-nnet-3]]: high-level idea only.
- **Vanishing/exploding gradients, batch normalization, weight initialization (Xavier/He)** - [[../wiki/lectures/L23-nnet-1]] / [[../wiki/lectures/L24-nnet-2]]: not discussed in depth.
- **Detailed CNN filter math, pooling variants, modern architectures** - [[../wiki/lectures/L24-nnet-2]]: high-level concept only.
- **Universal approximation proof** - [[../wiki/lectures/L23-nnet-1]]: stated, not proved (measure theory excluded).

### Other excluded
- **History questions** (who invented backprop, year of X) - [[../wiki/lectures/L22-unsupervised-2]] / [[../wiki/lectures/L27-summary]]: "I'm not going to ask you a history question."
- **Long essays**: be concise.
- **Heavy proofs / measure theory** - [[../wiki/lectures/L23-nnet-1]] / [[../wiki/lectures/L27-summary]]: out of scope.
- **VIF (Variance Inflation Factor)** - [[../wiki/lectures/L08-classif-2]]: marked self-study, not exam.
- **Formal hypothesis tests for normality** (Shapiro-Wilk etc.) - [[../wiki/lectures/L08-classif-2]]: "we're not going to talk about it."
- **Imbalanced-class asymmetric ROC analysis** - [[../wiki/lectures/L07-classif-1]]: book doesn't cover, prof skipped.
- **Bezier curves / CAD heritage of splines** - [[../wiki/lectures/L16-beyondlinear-1]]: pedagogical context only.
- **Weighted KNN, K-means++, Ward linkage formula, gap statistic**: mentioned as alternatives, not derived or examined.

### General principle

Anything in ISLR but NOT covered in lectures or exercises is OUT. When in doubt: grep `wiki/lectures/` and `exercises/` for the term, and if absent, it's out.

## Programming policy

The prof was definitive on Apr 28:

> "There's going to be no language. You don't have to memorize the different R packages, the names of different R packages or Python packages. There's no language-specific coding or anything of that sort." - [[../wiki/lectures/L27-summary]]

Concretely:
- Both R and Python are fine for the *course* (compulsory exercises default to R Markdown, Python tolerated).
- Neither is required on the *exam*. No memorizing function names, package names, syntax.
- Code-style questions from past papers are reformulated for 2026 as **interpretation questions**: instead of "fit a model and compute MSE," you'll see the fitted model's output and be asked to interpret it.
- **Pseudocode is acceptable** for "how would you compute X" questions. Math, English, or pseudocode all OK.

## Question patterns expected on the 2026 exam

From [[../wiki/lectures/L27-summary]] where the prof walked through old papers and showed how he'd rephrase them.

1. **Multiple choice / fill-in-the-blank**: concept checks. Single letter answer is fine.
2. **True/False with optional explanation**: write just T/F, OR explain reasoning for partial credit if uncertain.
3. **Output interpretation**: given a regression / GLM / CV / ROC / dendrogram / scree plot output, explain what it says (coefficients, significance, interactions, AUC, optimal λ, etc.).
4. **Hand calculations**: odds ↔ probability, degrees of freedom counting, MSE from a small table, PCA variance explained, hierarchical clustering distances by hand.
5. **Pseudocode / equation-writing**: "write out how you'd compute X." Math notation, plain English, or pseudocode all OK.
6. **At least one mathy theory question**: *"something mathy but not incredible, no weird spaces or fancy proofs."* The bias-variance derivation is the standout candidate.
7. **Method comparison**: given two models' results, which is better and why? Tie answer to bias-variance / overfitting / interpretability.

### Grading mechanics
- **Show your work for partial credit.** Empty is worse than wrong-with-reasoning.
- **No negative marking.** Always answer.
- **If a question feels broken or trick**: write a short note explaining your interpretation; you'll likely get the points.
- Percentile-based letter grades; the prof gives a small upward nudge but never adjusts down.

## Past exams: how to use them

Past papers are at `exams/TMA4268_2023_Exam.Rmd`, `..._2024_Exam.Rmd`, `..._2025_Exam.Rmd`. The prof walked through 2024/2025 in [[../wiki/lectures/L27-summary]] and described how he'd modify them.

| Past exam style | 2026 translation |
|---|---|
| "Fit `lm(...)` then compute MSE" | "Here's the regression output table: interpret coefficients / compute MSE from these residuals" |
| "Memorize the `gbm()` arguments" | "Explain conceptually how the boosting hyperparameters affect bias/variance" |
| "Write R code for k-fold CV" | "Write pseudocode (or math) for k-fold CV" |
| 4-hour grind designed to "run out the clock" | Open book, more interpretation-heavy, conceptual MC mixed in |

### What's actually changed for 2026
1. **Open book**: first time per the prof.
2. **No code writing, no package memorization.**
3. **More multiple-choice / true-false** vs older formats.
4. **More interpretation-heavy**: output tables given, you analyze them.
5. **Curriculum largely unchanged**: modules 1–12 are the same skeleton.

### How to use past exams during prep
- Do every conceptual question (interpretation, true/false, derivations) at full effort.
- For coding questions: rewrite them in your head as "given this output, interpret it", that's the 2026 version.
- Pay attention to bias-variance, regularization, classification metrics, PCA, as these recur every year.
- Don't memorize the R syntax in the answer keys.

## ISLR end-of-chapter exercises: how to use them

**Not in scope** per the prof's rule, as ISLR exercises are not the same as `exercises/`. But useful as *practice material* for hand-calculation fluency:
- The conceptual questions in each chapter end (those not requiring R) drill the same skills you'll need.
- Skip the `## Applied` sections that require coding.

## Logistics

- **Date / time**: 2026-05-18, 09:00
- **Duration**: 4 hours
- **Format**: Open book, digital (Inspera). Answers written on paper, then scanned.
- **Allowed**: ISLR (physical or PDF in Inspera), one A5 sheet of handwritten notes, calculator
- **Forbidden**: computer, R/Python, internet, executable code

## What this file does NOT cover

For these, see `exam_analysis.md`:
- Tier rankings of in-scope topics by importance
- Direction-of-effect cheat sheet (T/F traps)
- Worked-example dataset templates
- Common-mistake exam traps the prof flagged
- Procedural templates (G1–G6: hierarchical clustering by hand, LDA boundary derivation, etc.)
- Instructor's opinionated takes
