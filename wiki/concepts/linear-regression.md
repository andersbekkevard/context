---
concept: linear-regression
module: 03-linreg
lectures: [L05, L06]
isl-ref: 3.1, 3.2
exercises:
  - Exercise3.1c - fit lm(mpg~.) on Auto, interpret summary (relationship to response, weight coefficient, year coefficient)
  - Exercise3.1d - test importance of factor predictor `origin` with 3 levels
  - CE1 problem 2b - scatterplot weight vs stomach circumference, decide whether linearity makes sense, transform if not
  - CE1 problem 2c - fit additive lm with continuous (MAGENUMF) + factor (Gattung) predictor, write three group equations
related: [least-squares-and-mle, gaussian-error-assumptions, design-matrix-and-hat-matrix, sampling-distribution-of-beta, confidence-and-prediction-intervals, t-test-and-significance, r-squared, residual-diagnostics, collinearity, f-test, categorical-encoding-and-interactions, polynomial-regression, multivariate-normal]
tags:
  - concept
  - module/03-linreg
aliases:
  - simple linear regression
  - multiple linear regression
  - OLS
---

# Linear regression

The prof's frame: regression is **Y given X**, not the joint distribution. The simplest parametric model with the deepest theoretical plumbing, closed-form estimator, exact sampling distribution of β̂, and a workshop for every concept that comes later in the course (bias–variance, regularization, basis expansion).

## Definition (prof's framing)

> "Today we will discuss Y given X. So not the joint distribution of them, but Y given X. So we're trying to essentially make a model of it… we will look at how things co-vary, but in the sense of how Y varies as a function of X." - [[L05-linreg-1]]

Simple linear regression:

$$y_i = \beta_0 + \beta_1 x_i + \varepsilon_i, \qquad \varepsilon_i \sim N(0, \sigma^2)\ \text{i.i.d.}$$

Multiple linear regression (matrix form, the version actually used):

$$\mathbf{y} = \mathbf{X}\boldsymbol\beta + \boldsymbol\varepsilon, \qquad \boldsymbol\varepsilon \sim N_n(\mathbf{0}, \sigma^2 \mathbf{I}).$$

The bias term $\beta_0$ is hidden in the leading column of ones in $\mathbf{X}$, see [[design-matrix-and-hat-matrix]]. Prof prefers "bias" to "intercept" for $\beta_0$.

## Notation & setup

- $n$ samples, $p$ predictors. $\mathbf{X}$ is $n\times(p{+}1)$ (the +1 is the bias column), $\boldsymbol\beta$ is $(p{+}1)\times 1$, $\mathbf{y}$ and $\boldsymbol\varepsilon$ are $n\times 1$.
- "Independent / explanatory / regressor" all refer to $X$; "dependent / outcome / response" all refer to $Y$, prof uses them interchangeably.
- Notation gotcha he flagged: some books include the intercept in $p$, others don't.
- Body-fat / BMI is the running worked example. Slope ≈ 1.8 → one unit of BMI ↔ ~1.8 percentage points of body fat.

## Formula(s) to know cold

Simple least squares (closed-form, easily memorized):

$$\hat\beta_1 = \frac{\sum_i (x_i - \bar x)(y_i - \bar y)}{\sum_i (x_i - \bar x)^2} = \frac{\mathrm{Cov}(\mathbf{x}, \mathbf{y})}{\mathrm{Var}(\mathbf{x})}, \qquad \hat\beta_0 = \bar y - \hat\beta_1 \bar x.$$

Multiple regression closed form (the headline equation):

$$\boxed{\hat{\boldsymbol\beta} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}}$$

Predictions $\hat{\mathbf{y}} = \mathbf{X}\hat{\boldsymbol\beta} = \mathbf{H}\mathbf{y}$ where $\mathbf{H} = \mathbf{X}(\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top$ is the [[design-matrix-and-hat-matrix|hat matrix]].

Residuals $e_i = y_i - \hat y_i$. **Prof's distinction**: residuals are *predictions* of the underlying $\varepsilon_i$, not estimates, error terms are random variables.

## Insights & mental models

### Why study a model this simple

> "Is linear regression too simple? Yeah, for some things… but for other things it can be quite useful. And the nice thing is that, like I said, it's one you can understand." - [[L05-linreg-1]]

Three reasons the prof keeps coming back to it: (i) parametric, you can construct, modify, reason about the parameters; (ii) phenomena from much fancier models (e.g. the second descent past interpolation, see [[double-descent]]) can be derived and understood from the regression lens; (iii) interpretable. He invokes "always aim to minimize the length of my method sections."

He also pointed out that next-word language modeling is structurally not so different, Y is the next token, X is the prior text, just much more complicated.

### "Linear" means linear in the parameters

The most-quoted point in the lecture pair, repeated with emphasis:

> "It's still called linear regression even though you're fitting $y = \beta_0 + \beta_1 x + \beta_2 x^2$. It's a quadratic function but it's still a linear model. And it's linear because it's linear in the coefficients. Linear in the parameters." - [[L06-linreg-2]]

You can throw in $\log x$, $\sqrt x$, $\sin x$, dummies, interactions, still linear regression. See [[polynomial-regression]] and [[categorical-encoding-and-interactions]].

### Closed form is special

Most ML problems can't be solved in one shot, you iterate up the likelihood. Linear regression with full-rank $\mathbf{X}$ has the answer in a single matrix multiplication. The prof's framing:

> "Most of machine learning is finding good tricks to get to that peak… But in the case of linear regression with full-rank X, just right to the top. Very convenient. Which is also why we're studying regression even though we want to study something more complicated." - [[L06-linreg-2]]

### "All models are wrong, some are useful"

He cites George Box (via David Hand) to organize the rest: two purposes, *understand relationships* (interpretive) or *predict* (decisional). "In this course, I think the focus is more about prediction." - [[L06-linreg-2]]

### What you can't ask

> "Is this relationship causal, or better explained by something else?" - [[L05-linreg-1]]

Regression results are "fancy correlations." Don't read causation into the slope.

## Exam signals

> "I might say in the multivariate linear regression case, how would I test if at least one of the predictors is useful in predicting the response, or I might ask: why would I want to know this, what's the point? I like to ask kind of these questions… where you have to just sort of reason through why you're doing something." - [[L06-linreg-2]]

> "It's still called linear regression even though you're fitting $y = \beta_0 + \beta_1 x + \beta_2 x^2$. … It's linear in the parameters." - [[L06-linreg-2]]

The four canonical "important questions" in multiple regression are slide structure he flagged he might exam on, see "How it might appear" below.

## Pitfalls

- **Causation.** Regression is a fancy correlation. Don't claim causation from a slope.
- **"The eye line."** Drawing the best-looking line through a scatterplot tends toward the major axis (joint-distribution view), not the conditional regression line, which minimizes vertical residuals only.
- **Categorical with K > 2 levels coded as 0/1/2.** Imposes ordering; use $K-1$ dummies + reference category. See [[categorical-encoding-and-interactions]].
- **Confusing residual with error.** Errors $\varepsilon_i$ are random and unobservable; residuals $e_i$ are observed predictions of them.
- **n vs p.** OLS needs $n > p$ (full rank $\mathbf{X}^\top \mathbf{X}$). $n \le p$ → no unique solution; need regularization (module 6).

## Scope vs ISLP

- **In scope:** model statement, both estimators (LS / matrix), Gaussian-error assumptions, sampling distribution of $\hat{\boldsymbol\beta}$, CI/PI, t/F tests, $R^2$, categorical encoding, interactions, polynomial / nonlinear transforms, residual diagnostics, collinearity (qualitative).
- **Look up in ISLP:** §3.1, §3.2, §3.3 (pp. 59–104), especially §3.2.1 for the matrix derivation and §3.3 for the extensions.
- **Skip in ISLP (book-only / prof excluded):** F-test mechanics ([[L06-linreg-2]]: "won't ask any questions about an F-test"), VIF formulas ([[L08-classif-2]]: self-study), Moore-Penrose pseudoinverse details, formal Shapiro–Wilk normality tests ([[L08-classif-2]]).

## Exercise instances

- Exercise3.1c, multiple lm of `mpg` on all `Auto` predictors; interpret summary (significant predictors, weight, year)
- Exercise3.1d, F-test / ANOVA logic for the 3-level factor `origin`
- Exercise3.1g, interaction `year × origin`; interpret slope differences
- Exercise3.1h, try $\log X$, $\sqrt X$, $X^2$ to fix residual issues
- CE1 problem 2b, decide on a transformation for the worm data so linearity is reasonable
- CE1 problem 2c, fit lm with `MAGENUMF` (continuous, transformed) + `Gattung` (3-level factor); write three group equations
- CE1 problem 2d, test whether `MAGENUMF × Gattung` interaction matters

## How it might appear on the exam

- **Output interpretation.** Prof said explicitly: he won't make you write `lm(...)` code; he'll *give* you the regression output table (estimate, SE, t-value, p-value) and ask interpretive questions. The 2025 Q6a Boston-housing reformulation is the template - [[L27-summary]].
- **Q1 of the four canonical questions** ("is at least one predictor useful?"), knowing *why* you'd use an F-test (without computing it) is fair game.
- **Three group equations** when a factor predictor enters, just like CE1 2c, write the model out explicitly for each level.
- **Interaction interpretation**: main-effect coefficient on $Z$ is *only* the difference at $X = 0$; this is a flagged exam trap (see Q2 of the 2025 walk-through, [[L27-summary]]). Always state the interaction-aware interpretation.
- **Linear-in-parameters T/F.** "Adding $X^2$ makes the regression nonlinear", false; it's linear in $\boldsymbol\beta$.

## Related

- [[least-squares-and-mle]]: the estimator and its MLE equivalence under Gaussian errors
- [[gaussian-error-assumptions]]: what's assumed and what breaks
- [[design-matrix-and-hat-matrix]]: the algebra of the closed form
- [[sampling-distribution-of-beta]]: distribution of $\hat{\boldsymbol\beta}$ for inference
- [[confidence-and-prediction-intervals]]: uncertainty around mean response vs. future observation
- [[t-test-and-significance]]: per-coefficient inference
- [[r-squared]]: goodness-of-fit measure (prof distrusts)
- [[residual-diagnostics]]: checking the assumptions
- [[collinearity]]: what blows up $(\mathbf{X}^\top \mathbf{X})^{-1}$
- [[f-test]]: Q1 of the four important questions
- [[categorical-encoding-and-interactions]]: extension to factor predictors
- [[polynomial-regression]]: extension via basis expansion
- [[multivariate-normal]]: the foundation for the sampling distribution
- [[bias-variance-tradeoff]]: the lens for model selection on top of OLS
