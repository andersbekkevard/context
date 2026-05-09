---
concept: categorical-encoding-and-interactions
module: 03-linreg
lectures: [L05, L06]
isl-ref: 3.3.1, 3.3.2
exercises:
  - Exercise3.1d — work with factor variable origin (3 levels)
  - Exercise3.1g — fit lm with year × origin interaction, interpret
  - CE1 problem 2c — fit lm with continuous (MAGENUMF) + factor (Gattung), write three group equations
  - CE1 problem 2d — test whether MAGENUMF × Gattung interaction matters
related: [linear-regression, design-matrix-and-hat-matrix, f-test, t-test-and-significance, polynomial-regression]
tags:
  - concept
  - module/03-linreg
aliases:
  - dummy variable
  - reference category
  - interaction term
  - main effects rule
---

# Categorical predictors, dummy coding, and interactions

How to plug factors into the linear model — and the **canonical exam trap**: the main-effect coefficient on a categorical variable is *only* the difference at the other variable = 0 when an interaction is present. Plus the inviolable **main-effects rule**: if you include $X \cdot Z$, you must include $X$ and $Z$ separately.

## Definition (prof's framing)

### Binary predictor (2 levels)

Encode 0/1; same OLS machinery as continuous. Two equivalent ways to write it:

- Conditional: $y = \beta_0$ if $X=0$, $y = \beta_0 + \beta_1$ if $X=1$.
- Indicator: $y = \beta_0 + \beta_1 X$ with $X \in \{0, 1\}$.

### K-level factor (K > 2)

> "If they're all no, you know — that doesn't happen, right? Then the world ends. The reality is if you have all of those things, then you have actually too many parameters that become too interrelated, and then actually the model is not identifiable. The reason being that you can determine the value of one of them from the other two." — [[L06-linreg-2]]

So: use **K − 1 dummy variables** + **one reference category** baked into the intercept. R defaults to alphabetical first as the reference. Coding 0/1/2 instead would be wrong — it imposes an arbitrary ordering.

> "For three categories (black/white/blue) → use **two** dummy variables (e.g. 00, 01, 10), **never** 0/1/2 because that imposes an ordering between the categories." — [[L05-linreg-1]]

### Interaction

Add the product term so two predictors don't have to act additively:

$$y = \beta_0 + \beta_1 X + \beta_2 Z + \beta_3 (X \cdot Z) + \varepsilon.$$

For continuous × binary interaction, this lets the slope of $X$ depend on $Z$.

## Notation & setup

- $K$-level factor → $K - 1$ indicator columns in $\mathbf{X}$.
- The reference level has all dummies = 0; its mean is the intercept.
- For interaction terms in R: `Y ~ X * Z` adds main effects + interaction; `Y ~ X:Z` adds interaction only (not what you usually want — see main-effects rule).
- After fitting, **the model coefficients are *differences* from the reference**, not absolute means.

## Formula(s) to know cold

Writing out the model for each level — the canonical exam exercise (CE1 problem 2c). For a continuous $X$ + 3-level factor $Z \in \{A, B, C\}$ with $A$ as reference:

- Group $A$: $\hat y = \hat\beta_0 + \hat\beta_1 X$.
- Group $B$: $\hat y = (\hat\beta_0 + \hat\beta_2) + \hat\beta_1 X$ — same slope, shifted intercept.
- Group $C$: $\hat y = (\hat\beta_0 + \hat\beta_3) + \hat\beta_1 X$ — same slope, shifted intercept.

With an interaction $X \cdot Z$:

- Group $A$: $\hat y = \hat\beta_0 + \hat\beta_1 X$.
- Group $B$: $\hat y = (\hat\beta_0 + \hat\beta_2) + (\hat\beta_1 + \hat\beta_4) X$.
- Group $C$: $\hat y = (\hat\beta_0 + \hat\beta_3) + (\hat\beta_1 + \hat\beta_5) X$.

So interactions allow **different intercepts AND different slopes** per group.

## Insights & mental models

### The main-effects rule (non-negotiable)

> [!important] Main-effects rule — verbatim
> "If you do include interactions… whenever you include an interaction, you want to include what is referred to as the main effects. So if you want to look at A times B, then you also want to include A and B." — [[L06-linreg-2]]

ISLR calls this the **hierarchical principle**. Generalizes: works for two continuous variables (product), one continuous × one categorical, more than two levels, factor × factor, etc.

> "Creativity is up to you. You can go crazy with this. It's good and bad — you have so many ways you can change this model that at the end you want to be careful what you're doing." — [[L06-linreg-2]]

### The interaction-trap exam question

> "How does the feature pay-zero influence the odds to default? … We need to be able to do it for the men and the women." — [[L27-summary]]

The 2025 Q7 logistic-regression interaction problem (and the 2025 Q2 linear-regression one): **when a model has an $X \cdot Z$ interaction, the main-effect coefficient on $Z$ is *only* the $Z$ difference at $X = 0$.** Don't quote it as a global average.

Concretely from L27 (sex × pay interaction in default data):
- For **male** (sex = 0 say), one extra unit of `pay` multiplies odds by $e^{\beta_{\text{pay}}}$.
- For **female** (sex = 1), one extra unit of `pay` multiplies odds by $e^{\beta_{\text{pay}} + \beta_{\text{interaction}}}$.

State your assumed coding (e.g. "assuming male = 0, female = 1"); save yourself when the encoding is ambiguous.

### Worked example: Credit-card data with student × income

From the slides: model `Balance ~ Income + Student + Income:Student`. Without the interaction, students just have a different intercept (parallel lines). With it, students get **both** a different intercept *and* a different slope. In the actual fit:

- The bias jump for "student" alone *was* significant — students owe more.
- The income×student interaction came out **statistically weak** — slope difference suggestive but not significant.

> "It makes sense — they owe more. Even if they make money, you're still going to owe more." — [[L06-linreg-2]]

### Why not K dummies?

Mathematical statement: with $K$ dummies for a $K$-level factor, the columns are linearly dependent (their sum equals the intercept column), so $\mathbf{X}^\top\mathbf{X}$ is **singular** — no unique inverse. See [[design-matrix-and-hat-matrix]] / [[collinearity]] for the algebra. R simply drops one column for you.

You can change the reference category (e.g., `relevel(factor, ref="Caucasian")`) — the model is mathematically identical, just rewritten.

### Testing a factor as a whole

For "is this 3-level factor relevant?" you can't just look at the individual t-tests on $K - 1$ dummies (each one tests against the reference, not against "no effect"). Use the partial **[[f-test]]** on the joint null $\beta_{\text{dummy 1}} = \cdots = \beta_{\text{dummy K-1}} = 0$. R's `anova()` does this — and Exercise 3.1d makes you do it.

## Exam signals

> [!important] Main-effects rule — verbatim
> "If you do include interactions… whenever you include an interaction, you want to include what is referred to as the main effects. So if you want to look at A times B, then you also want to include A and B." — [[L06-linreg-2]]

> "How does the feature pay-zero influence the odds to default? … We need to be able to do it for the men and the women." — [[L27-summary]]

> 2025 Q2 linear-regression interaction trap: *"claims like 'males on average weigh more than females by 5' look right from the `sex` coefficient, but they're **only true at age = 0** because of the interaction term."* — [[L27-summary]]

> "Default coding. State your assumption when reading a model: 'Assuming male = 1, female = 0…'. Saves you when the encoding is ambiguous." — [[L27-summary]]

## Pitfalls

- **Quoting main effects under interaction.** The number-one trap. Main effect of $Z$ describes $Z = 1$ vs $Z = 0$ *only at $X = 0$*. If $X = 0$ is meaningless (e.g. age = 0), the coefficient itself is meaningless in isolation.
- **0/1/2 coding for ordinal-looking factors.** Imposes a numeric distance you didn't intend. Use $K - 1$ dummies.
- **K dummies and singular $\mathbf{X}^\top\mathbf{X}$.** R prevents this by dropping one. If you build $\mathbf{X}$ by hand and include all $K$, the inverse fails.
- **Forgetting the reference level when interpreting output.** All coefficients are *differences from* the reference. R picks alphabetically by default. State it.
- **Including only the interaction.** `Y ~ X:Z` (without `+ X + Z`) violates the main-effects rule. Almost always wrong.
- **Df accounting.** A $K$-level factor consumes $K - 1$ df, not $K$ and not $1$.
- **"Significant interaction" means…?** Slopes for different groups *truly differ*; not "just different point estimates." Look at the p-value on the product term.

## Scope vs ISLR

- **In scope:** dummy coding, reference category, identifiability, main-effects rule, interactions (continuous × categorical, continuous × continuous), three-group equation writing.
- **Look up in ISLR:** §3.3.1 (pp. 84–88, *Qualitative Predictors*); §3.3.2 (pp. 88–92, *Extensions of the Linear Model* — additive vs interaction).
- **Skip in ISLR (book-only / prof excluded):** sum-to-zero / contrast coding alternatives — beyond R's default treatment coding. Three-way interactions and higher — not covered. ANCOVA-specific terminology — not covered.

## Exercise instances

- Exercise3.1d — work with factor variable `origin` (3 levels: American/European/Japanese); use F-test / `anova()` to test whether `origin` is important
- Exercise3.1g — fit `lm(mpg ~ year * origin + ...)`; is the interaction relevant? interpret what it means for slopes by origin
- CE1 problem 2c — fit additive model `lm(GEWICHT ~ MAGENUMF + Gattung)` (with the transformations from 2b); write the **three group equations** explicitly; do we find evidence that `Gattung` impacts weight?
- CE1 problem 2d — fit `lm(GEWICHT ~ MAGENUMF * Gattung)`; test whether interaction is relevant

## How it might appear on the exam

- **Write the three group equations** for a continuous + factor model — direct from CE1 2c. Substitute the appropriate dummy values (0/0, 1/0, 0/1) into the model; collect terms.
- **Interaction interpretation.** "What does $\beta_3$ mean in $y = \beta_0 + \beta_1 X + \beta_2 Z + \beta_3 X \cdot Z$?" → the *additional* slope contribution for $Z = 1$ over the $Z = 0$ baseline. NOT the slope itself.
- **Interaction trap.** Given a regression with a `sex × age` interaction, evaluate the effect of being male: only at $\text{age} = 0$ unless you add the interaction term scaled by the actual age. The 2025 Q2 + Q7 templates.
- **Why K − 1 dummies?** Identifiability / singular $\mathbf{X}^\top\mathbf{X}$.
- **Main-effects rule T/F.** "It's OK to fit `Y ~ X:Z` without `Y ~ X + Z + X:Z`." False — must include main effects.
- **Test the factor.** "How would you test whether a 3-level factor is relevant?" → partial F-test on the joint null that all $K - 1$ dummies are zero (`anova(small, large)`).

## Related

- [[linear-regression]] — the parent model
- [[design-matrix-and-hat-matrix]] — identifiability / why $K-1$ dummies
- [[f-test]] — how to test a multi-level factor
- [[t-test-and-significance]] — per-coefficient inference (be careful under interaction)
- [[polynomial-regression]] — another extension via basis expansion
