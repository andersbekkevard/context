---
concept: elastic-net
module: 06-modelsel
lectures: [L13, L14]
isl-ref: 6.2.2
exercises: []
related: [ridge-regression, lasso, ridge-vs-lasso-geometry, regularization]
tags:
  - concept
  - module/06-modelsel
aliases:
  - elasticnet
  - L1+L2 regularization
---

# Elastic net

The centrist regularizer: combine the L1 ([[lasso]]) and L2 ([[ridge-regression|ridge]]) penalties so you inherit both — sparsity from L1 plus correlated-variable averaging from L2. The prof's verdict: *"This is probably the one that people use the most."* — [[L13-modelsel-2]]. Atom is short because the prof spent ~one minute on it; concept matters, the practical tuning details don't.

## Definition (prof's framing)

> "Elastic net… combine [L1 + L2] penalties together; tune both via CV." — [[L13-modelsel-2]]

The combined objective:

$$\hat\beta_\text{enet} = \arg\min_\beta \left\{ \text{RSS} + \lambda \sum_{j=1}^p \beta_j^2 + \gamma \sum_{j=1}^p |\beta_j| \right\}$$

Two penalty parameters now ($\lambda$ for L2, $\gamma$ for L1). If CV says "all L1," $\lambda \to 0$ and $\gamma > 0$. If CV says "all L2," vice versa. *"Often parameterized slightly differently in libraries, but the idea is the same."* — [[L13-modelsel-2]]

(A common library parameterization uses a single penalty strength $\lambda$ and a mixing parameter $\alpha \in [0, 1]$: penalty = $\lambda[\alpha \sum|\beta_j| + (1-\alpha) \sum \beta_j^2]$. $\alpha=1$ → lasso; $\alpha=0$ → ridge; $\alpha \in (0,1)$ → elastic net. The prof did not write this form; both parameterizations are equivalent.)

## Notation & setup

- Two regularization parameters (the prof's $\lambda, \gamma$), or equivalently one strength and one mixing parameter.
- Standardize predictors first (same as ridge / lasso).
- Choose both parameters by [[cross-validation]] over a 2D grid.

## Insights & mental models

### The hybrid intuition

Elastic net "inherits sparsity from L1 and the correlated-variable averaging of L2 — solves the failure mode where lasso arbitrarily picks one of two correlated features." — paraphrase of [[L13-modelsel-2]].

When two predictors are correlated:
- **Pure lasso** picks one, zeros the other (data-dependent which one wins).
- **Pure ridge** averages over them (no zeros, no selection).
- **Elastic net** can either zero them both *or* keep both at moderate values, depending on the L1/L2 mix and how much each contributes — **the practical compromise**.

### Geometric picture

The constraint region of elastic net is a **rounded diamond**: corners on the axes (gives sparsity, like lasso) plus rounded edges (averages over correlated variables, like ridge). See [[ridge-vs-lasso-geometry]] for the underlying logic.

### Where it lives in the lasso vs ridge spectrum

> "Lasso falls somewhere between ridge regression and best subset regression, and enjoys some of the properties of each." — slide deck (selection_regularization_presentation_lecture2.md) / [[L15-modelsel-4]]

> "If you only [are] concerned with prediction accuracy, either ridge or lasso. If model interpretability is desirable, lasso is preferred." — slide deck

Elastic net is the practical "use this by default" middle option: gives sparsity for interpretability but is less arbitrary than pure lasso under [[collinearity]].

### Why the prof spent only ~1 minute on it

He treats elastic net as the practical sweet spot but the *interesting* conceptual content is in the L1 vs L2 contrast — once you understand both extremes, elastic net is just "put both penalties in the objective." No new ideas, just tuning.

## Exam signals

> "This is probably the one that people use the most." — [[L13-modelsel-2]]

That's it. The prof did not exam-flag elastic net. From [[scope]]: *"Elastic Net detailed tuning — concept noted, no worked example."* Treat as in-scope at the conceptual level (know what it is, why it exists), out-of-scope for any detailed tuning question.

## Pitfalls

- **Treating it as a third method "different from" ridge and lasso.** It's literally the sum of both penalties — same machinery, just with two knobs.
- **Forgetting it still requires standardization.** Same as ridge / lasso.
- **Believing it "always wins."** It often does in practice, but pure lasso or pure ridge can win on CV for problems that are very sparse (lasso wins) or very dense (ridge wins).
- **Memorizing both prof's parameterization ($\lambda, \gamma$) AND the library ($\lambda, \alpha$) parameterization.** They're equivalent; the prof noted *"often parameterized slightly differently in libraries"* — you don't need both.

## Scope vs ISLR

- **In scope:** the combined L1+L2 objective; the conceptual claim that it inherits sparsity from L1 and correlated-variable averaging from L2; the prof's verdict ("the one people use the most"); the centrist-vs-extremes framing.
- **Look up in ISLR:** §6.2.2 pp. 263–264 mentions elastic net briefly in the comparing-lasso-and-ridge subsection. Not a deep treatment in ISL either.
- **Skip in ISLR:** detailed tuning algorithms for the L1/L2 mixing parameter; the original Zou-Hastie 2005 paper machinery. Not on the exam — *"concept noted, no worked example"* per scope notes and [[L13-modelsel-2]].

## Exercise instances

None. The slide deck does not have a recommended-exercise problem for elastic net; the Credit-data exercises stop at lasso (Exercise6.6). CE1 doesn't touch it either.

## How it might appear on the exam

- **Multiple choice / fill-in**: "Which method combines L1 and L2 penalties?" → elastic net.
- **True / false**: "Elastic net can produce coefficients that are exactly zero." → True (because of the L1 component).
- **Choose-method**: "We have many correlated predictors AND want some sparsity for interpretability — which method?" → elastic net (lasso would arbitrarily drop one of each correlated pair; ridge wouldn't sparsify; elastic net does both).
- **Conceptual short answer**: "What does elastic net buy you over lasso alone?" → reduces lasso's instability under correlated predictors by adding the L2 averaging effect, while preserving variable selection.
- Highly unlikely to be asked for the formula or for tuning details.

## Related

- [[ridge-regression]] — the L2 component.
- [[lasso]] — the L1 component.
- [[ridge-vs-lasso-geometry]] — the picture whose corners-plus-rounded-edges is the elastic-net constraint region.
- [[regularization]] — the cross-cutting Special; elastic net is the practical default in the explicit-regularization family.
