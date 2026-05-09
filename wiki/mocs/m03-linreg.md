---
moc: 03-linreg
title: Linear Regression
lectures: [L05, L06]
isl-ch: 3
slides:
  - modules/3LinReg/3LinReg.md
exercises:
  - exercises/Exercise3/
  - exercises/compulsory-exercise-1.md
tags:
  - moc
  - module/03-linreg
---

# Module 03: Linear Regression

The course's quantitative bedrock: two lectures (Jan 20, Jan 26) on simple → multiple OLS, the Gaussian-error machinery that gives β̂ a sampling distribution, and the diagnostic / interpretation toolkit. Load-bearing for the exam: the LS↔MLE derivation (the prof's flagged "mathy" question), interaction-coefficient interpretation, and residual-diagnostic reading. CE1 problem 2 is the canonical drill.

## Lectures
- [[L05-linreg-1]]: simple linear regression, least squares + MLE equivalence, Gaussian-error assumptions, sampling distribution of β̂, CI / t-test / R², categorical encoding intro
- [[L06-linreg-2]]: multiple regression in matrix form (design matrix, normal equations, hat matrix), collinearity, F-test (concept only), CI vs PI, interactions, polynomial regression, residual diagnostics

## Concepts (atoms in this module)
- [[linear-regression]]: y = β₀ + Σβⱼxⱼ + ε; closed-form β̂ = (XᵀX)⁻¹Xᵀy; the simplest model with the deepest theoretical plumbing
- [[least-squares-and-mle]]: minimizing SSE ⇔ MLE under Gaussian errors; the prof-flagged "mathy" exam-template derivation (Legendre + Gauss)
- [[gaussian-error-assumptions]]: εᵢ ~ N(0, σ²) i.i.d.; independence is the dangerous one, "violations ruin everything"
- [[design-matrix-and-hat-matrix]]: X is n×(p+1); H = X(XᵀX)⁻¹Xᵀ "puts hats on Y"; diagonal = leverage; appears in LOOCV shortcut and collinearity blow-up
- [[sampling-distribution-of-beta]]: β̂ ~ N(β, σ²(XᵀX)⁻¹); SE shrinks with bigger n and wider X-spread
- [[confidence-and-prediction-intervals]]: CI for the *mean response*, PI for a *future observation*; PI always wider (adds σ²)
- [[t-test-and-significance]]: t = β̂ⱼ / SE(β̂ⱼ); large n inflates significance for trivial slopes ("significance is just sample size"); engineering vs biology framing
- [[r-squared]]: R² = 1 − RSS/TSS; never decreases with more parameters; adjusted R² penalizes p; prof distrusts both, prefers test-set error
- [[residual-diagnostics]]: residuals-vs-fitted, QQ plot ("the kind of thing I would put on a test"), leverage ("fat kid on the seesaw"), studentized residuals
- [[collinearity]]: correlated predictors → XᵀX near-singular → SEs explode; fixes via dropping a variable, ridge, or PCA/PCR
- [[f-test]]: tests H₀: all βⱼ = 0; **prof said he won't ask you to compute it**, only why you'd use it
- [[categorical-encoding-and-interactions]]: K levels need K−1 dummies; main-effects rule; interpreting a main effect *at the reference level*, the canonical interaction trap
- [[polynomial-regression]]: still linear regression (linear in β); the standard simulation playground for bias-variance and double descent

## Cross-cutting concepts touched (Specials)
- [[bias-variance-tradeoff]]: first introduced module 02; this module is where the "Var(β̂) blows up under collinearity" half of the story lives, and polynomial regression is the canonical bias-variance simulation playground
- [[multivariate-normal]]: first introduced module 02; this module uses it for the joint sampling distribution of β̂ in [[L05-linreg-1]] / [[L06-linreg-2]]

## Exercises
- [[../../exercises/Exercise3]]: fit `lm(mpg~.)` on Auto, factor-predictor handling (origin, 3 levels), interaction `year × origin`, autoplot diagnostics (residuals, QQ, leverage), X transformations to fix violations; plus the matrix-form derivations of β̂ and Var(β̂ⱼ), CI vs PI simulations, and error-vs-residual definitions
- [[../../exercises/compulsory-exercise-1]]: problem 2 is the canonical end-to-end drill: scatterplot + linearity check + transformation, additive lm with continuous + factor (write the three group equations), test interaction significance, autoplot residual analysis with vs without transformations, T/F on what a p-value means

## Out of scope (this module)
- **F-test mechanics / formulas** - "I probably won't ask any questions about an F-test… too boring for this class" - [[L06-linreg-2]]
- **Variance Inflation Factor (VIF)** - marked self-study, not exam - [[L08-classif-2]]
- **Moore-Penrose pseudoinverse details** - explicitly bracketed off - [[L08-classif-2]]
- **Formal hypothesis tests for normality (Shapiro-Wilk etc.) and heteroscedasticity tests** - "we're not going to talk about it" - [[L08-classif-2]]
- **Spectral / eigen-decomposition theory of XᵀX** - deferred to Linear Statistical Models - [[L04-statlearn-3]]

## ISLP pointer
Chapter 3: Linear Regression. Deep treatment of in-scope concepts (closed-form β̂, sampling distribution, CI/PI, F-test, interactions, polynomial, diagnostics) is in `book/03-linreg.md`. Atoms carry section-level `isl-ref:` pointers.
