---
concept: odds-and-log-odds
module: 04-classif
lectures: [L07, L09, L27]
isl-ref: 4.3.1
exercises:
  - Exercise4.3a  -  given odds 0.37, find P(default)
  - Exercise4.3b  -  given P=0.16, find odds
related: [logistic-regression, sensitivity-specificity]
tags:
  - concept
  - module/04-classif
aliases:
  - odds
  - log-odds
  - odds ratio
---

# Odds, log-odds, and odds-ratio interpretation

The interpretive core of [[logistic-regression]]: the coefficient $\beta_j$ is a **log odds-ratio**, not a slope on probability. The prof flagged the odds ↔ probability conversion as the kind of question he *will* ask, calculator-friendly, format-flexible, and a classic interaction trap.

## Definition (prof's framing)

> "Odds is used in betting or horse races… 5-to-1 chance that something happens. Then there's a 5-over-6 chance that this thing happens because it happens in 5 out of 6 cases." - [[L07-classif-1]]

Given a probability $p$ of an event:

$$\text{odds} = \frac{p}{1-p} = \frac{\Pr(Y = 1 \mid X)}{\Pr(Y = 0 \mid X)}$$

Odds live in $(0, \infty)$; probability lives in $[0, 1]$. The two encode the same information; odds are the natural scale for the multiplicative logistic model.

## Notation & setup

- $p = \Pr(Y = 1 \mid X = x)$: probability of the "success" / class-1 event given covariates.
- $\text{odds} = p/(1-p)$: ratio of the class-1 probability to the class-0 probability.
- $\text{logit}(p) = \log(p/(1-p))$: the log-odds, the linear predictor in logistic regression.

## Formula(s) to know cold

**Odds ↔ probability conversions:**

$$\text{odds} = \frac{p}{1-p} \qquad p = \frac{\text{odds}}{1 + \text{odds}}$$

**The logistic model on the odds scale (multiplicative):**

$$\frac{p_i}{1 - p_i} = e^{\beta_0} \cdot e^{\beta_1 x_{i1}} \cdots e^{\beta_p x_{ip}}$$

**Odds-ratio interpretation of $\beta_j$**, increase $x_j$ by 1 unit, all other covariates fixed:

$$\frac{\text{odds}(X_j = x_{ij} + 1)}{\text{odds}(X_j = x_{ij})} = e^{\beta_j}$$

Equivalently: **$\beta_j = \log\text{(odds ratio)}$** for a one-unit increase in $x_j$.

## Insights & mental models

- **The unit of $x_j$ matters.** For `income` measured in dollars, $\beta_{\text{income}}$ might look tiny but $e^{10000 \cdot \beta_{\text{income}}}$ gives a meaningful per-\$10k change. The prof walked through this on the `Default` dataset, `balance` and `income` had very different per-unit interpretations because of scale. - [[L07-classif-1]]
- **Reference-class flip flips the sign.** If you encode 1 = default vs 1 = non-default, $\beta_{\text{balance}}$ flips sign. Always state your encoding when interpreting output.
- **Direction of effect by sign of $\beta$.** $\beta_j > 0$ → odds (and hence probability) of class 1 increase with $x_j$; $\beta_j < 0$ → decrease. Magnitude tells the multiplicative factor.
- **Logit on probability has no straight-line interpretation.** A one-unit change in $x_j$ produces a constant change in the *log-odds*, but a non-constant change in $p$, the $p$-change depends on where on the sigmoid you are.

## Interaction trap (prof flagged for the exam)

When the model includes $\beta_j x_j + \beta_k x_k + \beta_{jk} x_j x_k$, the odds ratio for a one-unit increase in $x_j$ is **not** $e^{\beta_j}$, it's $e^{\beta_j + \beta_{jk} x_k}$, which depends on $x_k$. From [[L27-summary]] (Q7 walkthrough on `default` data with `sex × pay_0` interaction):

> "How does the feature pay-zero influence the odds to default? … We need to be able to do it for the men and the women." - [[L27-summary]]

For each group (each level of the interacting categorical), compute the multiplicative effect separately. The 2025 exam question multiplied odds by $e^{\beta_{\text{pay}}} = 2.22$ for males and $e^{\beta_{\text{pay}} + \beta_{\text{interaction}}} = 2.57$ for females. Answer per group, not as a single number.

## Exam signals

> "What are the odds, the log odds? How do you compute them? What do they mean?" - [[L27-summary]]

> "This is the kind of question I would ask. It's simple. You calculate it. It's why you need a calculator." - [[L27-summary]]

> "By increasing the covariate by one unit, we change the odds for y to be in class 1 by a factor of the exponent of beta." - [[L07-classif-1]]

The 2025 exam Q3c was *exactly* the Exercise 4.3 conversion: given odds 0.37 → $p$, given $p = 0.16$ → odds.

## Pitfalls

- **Coefficient $\neq$ probability change.** Reporting "$\beta_{\text{age}} = 0.05$ means age increases default probability by 0.05" is wrong, it's the change in log-odds.
- **Forgetting the reference class.** Default-class encoding flip → sign flip on every coefficient.
- **Reporting odds-ratio for an interaction without specifying the level of the interacting variable.** See the trap above.
- **Computing odds-ratio for a non-unit change.** For a 100-unit change in $x_j$, the odds multiply by $e^{100 \beta_j}$, not $100 \cdot e^{\beta_j}$.

## Scope vs ISLP

- **In scope:** Odds definition, log-odds = logit, odds-ratio interpretation of $\beta_j$, the multiplicative form of the logistic model, the interaction-trap caveat.
- **Look up in ISLP:** §4.3.1 (logistic model + odds), pp. 134–135. Equation (4.4) is the canonical odds-ratio statement.
- **Skip in ISLP:** Nothing relevant excluded, odds is a small, self-contained piece.

## Exercise instances

- **Exercise4.3a**: odds = 0.37 of defaulting → fraction of people who default = $0.37/(1 + 0.37) \approx 0.27$.
- **Exercise4.3b**: $p = 0.16$ → odds = $0.16/0.84 \approx 0.19$.
- **CE1 problem 3b**: interpret $\beta_1$ in the tennis logistic regression: "one extra ace for player 1 multiplies the odds of player 1 winning by $e^{\beta_1}$."

The 2025 exam (Q3c) and 2024 exam both asked the same odds-conversion calculation, this is the most reliably-recurring exam question in module 4.

## How it might appear on the exam

- **Calculator MCQ:** "Given odds = 0.5, what is $p$?" (answer 1/3) or "Given $p = 0.8$, what are the odds?" (answer 4).
- **Coefficient interpretation:** Given a logistic-regression output table, "for a one-unit increase in $x_j$, the odds of $Y = 1$ multiply by ___" → $e^{\beta_j}$.
- **Interaction trap T/F:** "The odds ratio for $x_j$ when an interaction $x_j \cdot x_k$ is in the model is $e^{\beta_j}$" → false, depends on $x_k$.
- **Encoding T/F:** "Flipping the reference class flips the sign of $\beta_j$" → true.

## Related

- [[logistic-regression]]: the parent model where the odds-ratio interpretation lives.
- [[sensitivity-specificity]]: read off a confusion matrix at a fixed cutoff; the cutoff sits on the probability scale, not the odds scale.
