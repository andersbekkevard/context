---
module: 01-intro
isl-ch: 1
---

# Module 01 — Book delta

Module 01 is the framing module. ISLP chapter 1 is itself the framing chapter (Wage / Smarket / NCI60 motivation, notation conventions, brief history). The overlap is large and the delta is small: the prof's organizing vocabulary lives in ISLP §2.1, **not §1**, so a handful of named decompositions and dichotomies he introduces in L01 are absent from the mapped chapter. This file reproduces them in full so they're recoverable at the exam table without flipping ahead to ch. 2.

There are no derivations, no named matrices, no distributional results for module 01 — the math layer of the course starts in module 02. What follows is the small set of **named, lookup-able conceptual artifacts** he established in L01 that are not in ISLP §1.

---

## 1. The supervised data-generating model

[[[L01-intro|L01]], [[statistical-learning]]]

The decomposition the rest of the course assumes:

$$
Y = f(X) + \varepsilon
$$

where

- $X = (X_1, X_2, \ldots, X_p)$ is the vector of $p$ predictors (a.k.a. regressors, covariates, features — the prof flagged "independent variables" as a misleading term to avoid, since predictors are rarely independent),
- $f: \mathbb{R}^p \to \mathbb{R}$ is the unknown systematic component the course aims to estimate,
- $\varepsilon$ is the unobserved random error with the standard assumptions
  $$\mathbb{E}[\varepsilon] = 0, \quad \varepsilon \perp X.$$

The entire supervised half of the course (modules 2–9, 11) is "tools to estimate $f$, and sometimes things about $\varepsilon$."

ISLP introduces this decomposition in §2.1 (not in ch. 1). It is the canonical starting point for every supervised method.

---

## 2. Prediction vs. inference — the two reasons to estimate $f$

[[[L01-intro|L01]], [[prediction-vs-inference]]; ISLP §2.1.1–§2.1.2]

Same model, two uses. **The design choices diverge.**

**Prediction.** Goal: $\hat Y = \hat f(X) \approx Y$ on unseen data. $\hat f$ is allowed to be a black box. [[reducible-vs-irreducible-error|Reducible vs irreducible error]] split:

$$
\mathbb{E}\bigl[(Y - \hat Y)^2\bigr]
= \underbrace{\bigl(f(X) - \hat f(X)\bigr)^2}_{\text{reducible}}
+ \underbrace{\mathrm{Var}(\varepsilon)}_{\text{irreducible}}.
$$

Irreducible error sets the floor; no method can do better than $\mathrm{Var}(\varepsilon)$ in expected squared loss. This is the formal statement of the prof's claim that "even a perfect model can't predict perfectly."

**Inference.** Goal: understand the *form* of $f$ — which $X_j$ matter, sign / size of effects, what the relationship looks like. Black-box methods are disqualified because you can't interpret them.

**Design-choice contrast.**

| | Prediction | Inference |
|---|---|---|
| Care about $\hat\beta$? | No (often). | Yes. Interpretability is the whole point. |
| Care about test MSE / AUC? | Yes — primary metric. | Sanity check only. |
| Care about p-values / SEs / CIs? | Mostly no. | Yes. |
| Tolerated method class | NNs, boosting, deep ensembles. | Linear, GAMs, single trees. |
| Tolerated R² level | High desirable. | Low OK if slopes interpretable (Framingham "blob"). |

**Canonical examples (prof's framing).**

- *Prediction-first*: quants modelling Bitcoin. "They don't really care if the model is right. Being right is secondary." [[L01-intro|L01]]
- *Inference-first*: Framingham systolic-blood-pressure regression — R² is tiny ("blob-grade correlation"), and the paper is still seminal because doctors use the *coefficients*, not the predictions. "The animal's dead. The goal is an understanding of what's going on." [[L01-intro|L01]]

**Trap.** Inference does not require causality. The prof's framing: "ideally causal, more often correlational." Don't overclaim cause from a coefficient.

---

## 3. Supervised vs. unsupervised — the first organizing axis

[[[L01-intro|L01]], [[supervised-vs-unsupervised]]; ISLP §2.1.4]

Definitions:

- **Supervised.** Data is $\{(x_1, y_1), \ldots, (x_n, y_n)\}$ with $y_i$ either continuous (regression) or categorical (classification). Goal: estimate $f$ in $Y = f(X) + \varepsilon$. "You know what you want to get out of it." [[L01-intro|L01]]
- **Unsupervised.** Data is $\{x_1, \ldots, x_n\}$ with no $y$. Goal: find structure (groups, low-dim summaries, dependencies) without an external target. "You're fiddling around and seeing what falls out." [[L01-intro|L01]]

**The defining feature is not whether you have data — it's whether the data has a label / response that says what right looks like.**

**Three umbrella problem types** that organize the entire course:

| Problem type | Response $Y$ | Modules |
|---|---|---|
| Regression | continuous | 3, 6, 7, 8 |
| Classification | categorical | 4 |
| Unsupervised | none | 10 |

**"Supervised in disguise" reframing.** [[[L01-intro|L01]] / [[L02-statlearn-1|L02]]] An apparently-unsupervised problem becomes supervised when you can construct $y$ from the data itself. Examples: LLM next-word prediction, autoencoder reconstruction (input = target), masked-token modelling, click-vs-no-click. The next-word loss is *fully supervised* even though there's no human-labeled $y$. Standard exam trap: "an LLM is unsupervised" → **False**.

**Unsupervised as "dangerous statistics."** [[L01-intro|L01]] No $y$ means no objective measure of success. Run enough analyses on the same data and *something* will look significant; p-values from a clustering-then-test pipeline on the same data are not valid because they ignore the search. Honest pattern: **unsupervised exploration → hypothesis-driven supervised validation** on held-out data or a follow-up experiment.

---

## 4. The four-cell organizing matrix

[[[L01-intro|L01]] — derived from the two orthogonal axes]

The two organizing axes (supervised/unsupervised and prediction/inference) are orthogonal. The four conceptual cells:

| | **Prediction** | **Inference** |
|---|---|---|
| **Supervised** | Forecasting (quant finance, weather, Bitcoin) | Econometrics, medical-risk-factor identification (Framingham) |
| **Unsupervised** | Doesn't really exist | Exploratory clustering / PCA / segmentation |

Useful for tag-a-scenario exam questions: identify each axis independently, the cell falls out.

---

## 5. Statistical learning vs. its neighbours

[[[L01-intro|L01]], [[L02-statlearn-1|L02]], [[statistical-learning]]]

The course distinguishes statistical learning from three adjacent fields. The distinctions are part of the Q1-style fill-in-the-blank vocabulary.

**Statistical learning vs. classical statistics.** Classical stats *states the model first* ("I assume Gaussian…") then argues from data. Statistical learning *takes the data first* and figures out what to do; the distributional assumption is relaxed or absent. The misspecified-model regime is the default working setting.

**Statistical learning vs. machine learning.** Same algorithms, different angle. ML is algorithmic and engineering-focused ("get it to work"). Statistical learning is theory-focused — uncertainty, [[bias-variance-tradeoff|bias-variance]], distributional behaviour under misspecification. "Statistical learning was a way for statisticians to get in on machine learning." [[L01-intro|L01]]

**Statistical learning vs. data science — the 6-step pipeline.** Data science spans:

1. Formulating a hypothesis
2. Acquiring / scraping data
3. Going from unstructured to structured data
4. Setting up a model
5. Implementing the analysis
6. Interpreting and communicating results

**Statistical learning owns steps 4–6.** Steps 1–3 are out of scope for this course. This carve-up is the prof's explicit definition of where stat learning sits in the wider workflow.

---

## 6. The variance-stabilising transformation used in the Framingham worked example

[[L01-intro|L01]]

A concrete numerical recipe the prof demonstrated on the Framingham SBP response: the raw response was right-skewed with a heavy upper tail. Apply

$$
y \;\longmapsto\; -\frac{1}{\sqrt{y}}
$$

and the transformed response becomes approximately bell-shaped. The motivation: [[linear-regression]] performs and infers better when residuals are approximately normal, so reshape the response if it isn't.

The negative sign is cosmetic (it makes the transformed variable increase with $y$ rather than decrease) — the variance-stabilising work is done by the $1/\sqrt{y}$ part. The general lesson:

> "These kinds of transformations are a common trick to take the data and put it into a form where you can use these regression-type models." [[L01-intro|L01]]

This is one of a family of monotone transformations (log, square root, Box–Cox) used to bring skewed responses closer to symmetric. The exact form $-1/\sqrt{y}$ is specific to the prof's worked example; the *principle* (transform-the-response-to-stabilise-variance) is the lookup-able artifact.

---

## Notation and naming differences

- **"Independent variables"** — prof avoids this term because the predictors are rarely independent of each other. Prefer **predictors, regressors, covariates, features**.
- **"Inputs / outputs"** (ISLP §1) ≡ **predictors / response** (prof) ≡ $X$ / $Y$ in $Y = f(X) + \varepsilon$. ISLP §1 uses the casual "inputs/outputs" language; the prof switches to the formal $X / Y$ notation immediately.
- **"Cluster"** (ISLP §1, NCI60 example) ≡ what unsupervised methods produce; the prof reserves "cluster" for the output of [[hierarchical-clustering]] and [[k-means-clustering]] specifically (module 10) — both ISLP and the prof use the term in compatible ways here.
