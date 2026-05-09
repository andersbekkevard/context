---
lecture: 27
date: 2026-04-28
module: 12-final
title: Summary and Exam Review
slides: modules/12Final/12Final.md
topics:
  - exam-logistics
  - exam-scope
  - bias-variance-tradeoff
  - linear-regression
  - categorical-encoding-and-interactions
  - lasso
  - neural-networks
  - activation-functions
  - logistic-regression
  - odds-and-log-odds
  - smoothing-splines
  - principal-component-analysis
  - hierarchical-clustering
  - cross-validation
  - boosting
  - generalized-additive-models
  - sensitivity-specificity
  - roc-auc
  - knn
  - random-forest
  - least-squares-and-mle
tags:
  - lecture
  - module/12-final
  - exam
aliases:
  - Lecture 27
  - Exam Review
---

# L27 — Summary and Exam Review

The final class before the exam. The prof skipped the wrap-up slides and instead spent the whole session walking through the 2025 exam (and one 2024 question), explaining how each question would be re-formatted for *his* version of the exam — no R/Python coding, results given as tables, calculations and interpretation only. Bookended by exam logistics and the canonical scope rule. This is the single highest-priority lecture for exam prep.

## Key takeaways

> [!important] The scope rule (verbatim)
> "If it was covered either in the slides or in the exercises, then I would say fair game. If it's only in the book and we didn't talk about it in class or in the exercises, then it's not going to be in the test."

- **Exam: May 18, 09:00, 4 hours, open book.** Bring ISLR (your own copy or the provided PDF), one A5 sheet of handwritten notes, a calculator. Post-it tabs in the book are fine; don't write on them.
- **No language-specific code.** No R/Python package names to memorize. He may ask for **pseudocode** or math, and he'll **give you the regression output / table / plot** instead of making you generate it.
- **There will be a question on the bias-variance decomposition.** Said multiple times in the course. "If you haven't learned anything in the course, I recommend reading that part of the book."
- **Format mix:** mostly multiple choice + true/false + short interpretation, plus at least one mathy/derivation question, plus data-analysis style questions where output is provided. Probably uses paper-with-PDF (write "Q1: A" on paper); may try Inspera buttons but probably won't.
- **Show your work.** Negative scoring not used ("I won't give negative points"). Partial credit if calculator slip but equation is right. If you think a question is broken or a trick, write that down — you can earn points by flagging it.
- **No empty answers.** "If you have some idea I'll probably give you something for something." But don't dump "the history of the world."

## Exam logistics

### When and where

May 18 at 09:00, 4-hour exam. He emphasized that his exams are usually shorter than 4 hours, but the open-book format makes it harder to keep them short — historically these exams "were probably designed to run out the clock … such that if you really didn't know something, you didn't have time to really learn it." He'll try not to do that, but it's still going to be substantial.

### Aids

- **The course book (ISLR 2nd ed).** Bring your own physical copy or use the PDF provided on the testing computers ("standard PDF … you can search with it"). Post-it tabs marking chapters/sections are allowed; don't write on the post-its.
- **One A5 sheet, handwritten.** On whether iPad-handwritten then printed counts: "the safest thing is always just write it actually on paper … the ones who decide that are the old people at the testing center."
- **Calculator.** "Pretty much any calculator is fine … you won't need anything fancy." Bring one — "no one wants to do long division by hand."

### Mechanics

The exam runs through Inspera. The expected workflow: click the question button, the PDF opens with the whole exam, take out paper, write "Question 1 — answer", "Question 2 — B", etc. At the end you hand in the stack of papers to be scanned.

> [!note] Multiple choice format
> "I am tempted to try to use Inspera's stupid buttons. I probably won't. In which case you just write on the paper, like 'Question 1, F. Question 2, B.'"

So expect: write all answers on paper, including MC. Easy to grade either way.

### What's on the curriculum

Lectures + recommended/compulsory exercises + relevant ISLR chapters noted in the slides. "In general, as you probably noticed, the slides and the material largely are mirroring the book, so most everything is in the book." But the scope rule is the canonical filter — see callout above. Sources of inspiration for questions: exercises (most important), questions asked in class, book end-of-chapter exercises, old exams.

### Format mix

> "It does test your ability to do data analysis and your theoretical understanding of these different methods and tasks and ideas. And I'll try to have at least one question that's more mathy where you should derive something … and then some that are just multiple choice that test the general knowledge."

Roughly equal weight on all modules. "Probably won't cover everything, but you don't know which one, so you've got to learn it." If you've solved all the exercises, you're generally pretty good.

### Scoring and grading

- No negative scoring for wrong answers. "I won't give negative points."
- Partial credit for showing work, especially when a calculator slip costs you the final number but the setup is right.
- Flag broken/trick questions in writing — credit available even if your final answer is "wrong."
- Grade conversion: percentile-based, NTNU's standard scale. "I never adjust down … typically I will boost a little bit."

## Walkthrough: 2025 exam, problem-by-problem

He used the 2025 paper as the running example, showing for each question how he'd reformulate it for his exam (strip the R-coding, give the output, ask for interpretation).

### Q1 — Fill-in-the-blank conceptual (regression vs. classification, prediction vs. inference)

A long passage with red blanks; you fill in `regression`, `classification`, `prediction`, `inference`, etc. He'd convert to A/B/C/D and have you write "Problem 1A: classification" on paper.

> "I like this kind of question … it's conceptual, but you also don't have to write a whole book — you just have to know which words to fill in correctly."

Designed so you have to read carefully — e.g. "in the latter case we do not care about the actual model parameters" → that's prediction. The answer is buried in subtle wording, not the obvious word at the top of the paragraph.

### Q2 — Linear regression with interaction (true/false)

Fitted a model with `age`, `sex`, and `age × sex` interaction. Several true/false claims about the slopes. The trap: claims like "males on average weigh more than females by 5" look right from the `sex` coefficient, but they're **only true at age = 0** because of the interaction term. Need to read the model carefully.

> [!warning] Format note for true/false
> "If you want to explain why … you can. If you're confident in the answer you just write True/False, that's fine, or T/F, that's also fine. But if you feel the need to, feel free to write in why you think it's either true or false."

He warned about writing too much: "I don't want the history of the world."

### Q3a — Lasso vs. least squares (multiple-select)

> [!important] Promised on the exam
> "There will be a question on the bias-variance decomposition. So if you haven't learned anything in the course, I recommend reading that part of the book."

The lasso correct answer: less flexible than LS, improved accuracy when the increase in bias is less than the decrease in variance. False: "more flexible than LS." This is straight bias-variance reasoning applied to [[lasso]].

### Q3b — Neural network parameter count + forward pass

Given a feed-forward [[feedforward-network|neural network]] with stated input/hidden/output sizes: "How many weights total, including biases?" Then: given specific weight values, inputs, and a [[activation-functions|relu]] activation, compute the output of the neuron.

> [!note] Question is technically ambiguous
> "Actually, this is a good example of where the question is technically wrong … you could technically have skip connections, or connections between neurons within a layer. So here you could say, 'just to be clear, I'm assuming a feed-forward network,' even though I think everyone would."

His advice for the count: **draw the network, then count the parameters**, including bias terms.

For the output: just multiply, sum, add bias, apply ReLU.

### Q3c — Logistic regression odds

> "What are the odds, the log odds? How do you compute them? What do they mean?"

Definition: $\text{odds} = \frac{P}{1-P}$. Two parts:
1. Given odds = 0.37, compute $P$ → solve $\frac{P}{1-P} = 0.37$ → $P = 0.27$.
2. Given $P = 0.16$, compute odds → $0.19$.

> "This is the kind of question I would ask. It's simple. You calculate it. It's why you need a calculator."

Calculator slip with correct equation written → still gets most of the points.

### Q3d — Smoothing splines (multiple-select)

Increasing $\lambda$ on a [[smoothing-splines|smoothing spline]] does **not** make it more flexible/wiggly — it makes it **smoother**. So the "more wiggly with higher lambda" claim is false; the rest are true. Standard splines material from the lectures.

### Q3e — PCA: explained variance + loadings

Five standardized variables, [[principal-component-analysis|PCA]] gives PCs with explained variances and loadings. Three sub-questions:
1. Total variance explained by the first 4 components → just sum the four eigenvalue ratios.
2. Number of PCs needed for 90% of total variance → cumulative sum: 0.54 + 0.30 = 0.84 (not enough), +0.10 → past 0.90, so **3 PCs**.
3. Compute the score for a given observation on a given PC → plug observation into the loading vector.

> "This kind of question I would also say is fair game because it tests basic knowledge of PCA that we covered in class, with simple calculations. And again, show your work so that you can get partial credit when the inevitable little mistake happens."

### Q3f — Why are nonlinear activations necessary?

Only correct answer: they let the network represent complex nonlinear functions. (A linear sum of linear things is linear.) Distractors: "reduces parameters" (no), "makes the network fully connected" (orthogonal — fully-connectedness is a topology choice).

### Q4 — Polynomial regression: training RSS vs. extra terms

Two models for the same data — model B has redundant quadratic terms ($\beta_2 x^2 + \beta_3 x^2$ — collinear). "If you try to fit this model, your optimizer goes, 'hey, no, this sucks.'" Adding L2 makes $\beta_2 = \beta_3$. The exam writers made a mistake here, but the answerable question is: assuming the truth is linear, will training RSS be lower for the bigger model?

> [!important] Train vs. test — the keyword
> "This is the key word here. If that word wasn't 'training,' if it was 'testing,' it would of course be different."

Answer: **training** RSS will be **lower** for model B (more parameters → fits noise → overfit on training). Test RSS would more likely be worse for model B, but you can't be certain. Bonus point: model B is also a terrible model because the redundant terms can trade off (collinearity).

### Q5 — Hierarchical clustering with complete linkage by hand

Given a 4×4 dissimilarity matrix; build the dendogram with [[hierarchical-clustering|complete linkage]]. Recipe:
1. Find smallest off-diagonal entry → first merge.
2. Recompute distances from the new cluster to remaining points using **max** (complete linkage = max inter-cluster distance).
3. Repeat: merge next-smallest, recompute matrix, until one cluster.

He worked through it: first merge two closest at level 2, recompute the 3×3 matrix taking maxes (e.g. distance from new cluster to point 4 = max(7, ...) = 7), then merge again at level 6, etc. Also k-means is fair game with similar hand-computation style.

> [!note] Negative-point convention
> "Negative one point for each mistake … I won't give negative points, don't worry."

He noted that elsewhere ("in my country") you can actually score negative on a question; he doesn't do that.

### Q6a — Linear regression interpretation (Boston housing)

Original asked you to write R code (`lm(...)`) and read the summary. **He won't.** Instead: he'll give you the equation explicitly written out — e.g.

$$y = \beta_0 + \beta_1 x_1 + \cdots + \beta_3 \cdot \text{rm}^2 + \varepsilon$$

— plus the standard regression output table (estimate, SE, t-value, p-value). Then ask things like:
- How many degrees of freedom does the model consume? (count variables)
- On average, how much more/less does property X cost given Y?
- "Is there evidence that variable Z is relevant?" → he'd say *what test* would you use (F-test for ANOVA), without making you compute it numerically.

### Q6b — Train/test MSE, then 10-fold CV with lasso

Original made you partition data and run lasso. **His version:** he gives you the train/test MSE numbers and the cross-validated $\lambda$ plot. You'd write the test MSE formula in pseudocode/math:

$$\text{MSE}_\text{test} = \frac{1}{|\text{test}|} \sum_{t \in \text{test}} (y_t - \hat y_t)^2$$

Where $\hat y_t = \beta_0 + \sum_i \beta_i x_{it}$. Just write it as math.

For the CV plot, he might ask "how would you pick $\lambda$ here?" (smallest CV error). The interesting follow-up: compare the lasso-with-best-$\lambda$ test MSE (50.8) to the unregularized one (50.78). Lasso did **worse** → conclude that all parameters seem to matter.

> [!note] Bias-variance tie-in
> "Normally with regularization … you trade off — by including it, you trade off between an increase in bias by getting a reduction in variance. But in this case, the reduction in variance is not offset by the increase in bias. So you don't want to add a regularizer, meaning you want to keep all the parameters."

### Q6c — Gradient boosting

> "I won't have you memorize the names of the R functions, of course, but you should know what tree boosting is."

Won't ask you to fit. Will ask: "We need to specify the number of trees, the depth, and the shrinkage. How do you determine good values?" → cross-validation. The point of the question is whether you understand the hyperparameters of [[boosting]] and how to tune them.

He'll give you the test MSE that boosting achieved (smaller than linear regression) and ask you to interpret: "It's probably because there's some non-linear interactions that weren't captured in the linear regression model."

[[variable-importance|Influence/importance plots]] are fair game — know what they mean and where they come from, but you won't compute them.

### Q6d — GAM with B-splines

[[generalized-additive-models|Generalized additive models]] = sums of splines on different variables. Fair questions:
- How many degrees of freedom does this B-spline have/consume? (count the knots)
- Compare the GAM test MSE to other models (boosting wins, GAM beats plain regression).

You won't fit the model.

### Q7 — Logistic regression with sex × pay interaction (default data)

[[logistic-regression]] for binary `default`. Output table: estimate, SE, z-value, p-value. The interaction term sex × pay matters.

> [!warning] Interaction trap
> "How does the feature pay-zero influence the odds to default? … We need to be able to do it for the men and the women."

Key gotcha: figure out the encoding (here male=1, female=2; default=male if sex=0 in some encodings — **make this very obvious in your answer**), then compute the multiplicative effect on odds **separately for males and females** because of the interaction. For each one-unit increase in pay-zero (e.g. one extra month delayed), odds multiply by $e^{\beta_\text{pay}}$ for males, $e^{\beta_\text{pay} + \beta_\text{interaction}}$ for females.

### Q7 cont. — Sensitivity, specificity, ROC

- **Define** [[sensitivity-specificity|sensitivity and specificity]] in plain English **for this specific model** (sensitivity = ability to identify defaulters; specificity = ability to identify non-defaulters).
- **Write the equation** rather than computing — the formula counts as the answer.
- He'll show the [[roc-auc|ROC curve]] and ask what it means / how to interpret it.

### Q7 cont. — KNN classifier (limited reformulation)

> "When I looked at this question, I didn't see how I could possibly ask anything about this other than … if I gave you the confusion matrix, you could estimate sensitivity and specificity from it."

Most KNN questions need code, so on his exam you'd likely just get a [[confusion-matrix]] to interpret. Fair game: describe how [[knn]] works, compute sens/spec from a given confusion matrix.

### Q7 cont. — Tree-based method choice

Original let students pick any tree method. He'd ask: "What tree-based method would you use, and justify the parameters." E.g. [[random-forest|random forest]], $m \approx \sqrt{p}$, etc. Justify your choices.

### Q7 cont. — Class imbalance and bias

Given the distribution of `default` (heavily imbalanced — most are zero), a naive "always predict 0" classifier already does well on accuracy. Discuss in terms of sensitivity/specificity vs. error rate.

> "Models that are really naive and only predict that it's going to be a zero are already going to do pretty well, because the one class is almost all of the data."

## The mathy question (2024 exam, problem 3)

> "I do generally like to keep one theory question."

The example he showed: assume an additive Gaussian error model $y = f(x; \theta) + \varepsilon$ with $\varepsilon \sim N(0, \sigma^2)$. **Show that maximum likelihood and least squares are equivalent in $\theta$.**

Sketch: write the [[least-squares-and-mle|log-likelihood]] of $y_i = f(x_i;\theta) + \varepsilon_i$ under Gaussian noise:

$$\log L(\theta) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_i (y_i - f(x_i;\theta))^2$$

The first term doesn't depend on $\theta$; $\sigma^2$ as a constant doesn't either. The only $\theta$-dependent piece is $\sum (y_i - f(x_i;\theta))^2$, which is maximized for the log-likelihood when **minimized** — and that's exactly the least-squares objective.

> [!note] Q&A correction during derivation
> A student pointed out an inconsistency in the prof's signs/min-vs-max — "Yeah, you're right. The text was written wrong, but that's okay." Maximizing the log-likelihood = minimizing the negative = minimizing the SSE. Watch the signs on the exam.

> "Not incredibly profound or difficult, but at least somewhat theoretical or mathy-ish. I'll try to include something along these lines, where it's mathy but not, you know, no weird spaces or fancy proofs."

## Common-mistake themes the prof flagged

- **Interaction terms.** A "main effect" coefficient on `sex` only gives the male-vs-female difference *when the interacting variable is at zero*. Don't quote it as a global average. Always evaluate odds/effects separately for each level of the interacting categorical when an interaction is present.
- **Train vs. test.** When asked about RSS/MSE, the keyword `training` vs. `test` flips the answer. Adding parameters always lowers training error; test error depends.
- **Smoothing parameter direction.** Higher $\lambda$ in a smoothing spline = **smoother**, not wigglier. Same for ridge/lasso: more regularization = simpler model.
- **Default coding.** State your assumption when reading a model: "Assuming male = 1, female = 0…". Saves you when the encoding is ambiguous.
- **Question is broken / ambiguous.** State the assumption you're making and why. Earn credit by flagging it explicitly rather than silently picking one interpretation.
- **Class imbalance.** A high accuracy can be meaningless under heavy imbalance — check sensitivity/specificity.
- **Collinearity / redundant features.** If two features are essentially the same, optimizers blow up; L2 makes them split the coefficient evenly. Models with them overfit on training data.

## Closing

> "If you've solved all the exercises, you're generally pretty good. Those are designed to really cover the material."

Q&A on grading scale: standard NTNU percentile mapping, no downward adjustment, "typically I will boost a little bit." Email him with questions before the exam, "I'm not the best emailer, but we'll see."

> "Good luck for the test."
