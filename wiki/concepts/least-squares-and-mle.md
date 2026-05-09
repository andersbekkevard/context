---
concept: least-squares-and-mle
module: 03-linreg
lectures: [L05, L06, L12, L27]
isl-ref: 3.1.1, 3.2.1
exercises:
  - Exercise6.1a — derive β̂ = (XᵀX)⁻¹Xᵀy by differentiating RSS
  - Exercise6.1b — show MLE equals LS for the multiple linear regression model under Gaussian errors
related: [linear-regression, gaussian-error-assumptions, design-matrix-and-hat-matrix, sampling-distribution-of-beta, multivariate-normal]
tags:
  - concept
  - module/03-linreg
aliases:
  - LS = MLE
  - maximum likelihood for OLS
  - Legendre vs Gauss
---

# Least squares and the MLE equivalence

The prof's mathy theory question template for the exam: **under Gaussian errors, minimizing SSE is equivalent to maximizing the likelihood.** Legendre got the minimization; Gauss got the distributional understanding.

## Definition (prof's framing)

Least squares: minimize the residual sum of squares.

$$\hat{\boldsymbol\beta} = \arg\min_{\boldsymbol\beta} \sum_{i=1}^n (y_i - \mathbf{x}_i^\top \boldsymbol\beta)^2.$$

MLE under $\varepsilon_i \overset{\text{iid}}\sim N(0, \sigma^2)$: maximize the joint likelihood. The two coincide.

> "If you minimize this least squares error, it's equivalent to minimizing this likelihood function… Legendre, he figured that out, and this one was Gauss, and that was convenient because then he could say that I'm assuming that my epsilon are normally distributed with a zero mean and a fixed variance." — [[L05-linreg-1]]

## Notation & setup

Standard linear model $\mathbf{y} = \mathbf{X}\boldsymbol\beta + \boldsymbol\varepsilon$, $\boldsymbol\varepsilon \sim N_n(\mathbf{0}, \sigma^2 \mathbf{I})$. RSS is the scalar objective; the normal equations come out by setting $\partial \mathrm{RSS}/\partial \boldsymbol\beta = \mathbf{0}$.

## Formula(s) to know cold

The derivation has three lines and the prof did it on the board:

$$\mathrm{RSS}(\boldsymbol\beta) = (\mathbf{y} - \mathbf{X}\boldsymbol\beta)^\top (\mathbf{y} - \mathbf{X}\boldsymbol\beta) = \mathbf{y}^\top\mathbf{y} - 2\boldsymbol\beta^\top \mathbf{X}^\top \mathbf{y} + \boldsymbol\beta^\top \mathbf{X}^\top \mathbf{X}\boldsymbol\beta.$$

$$\frac{\partial \mathrm{RSS}}{\partial \boldsymbol\beta} = -2\mathbf{X}^\top\mathbf{y} + 2\mathbf{X}^\top\mathbf{X}\boldsymbol\beta = \mathbf{0} \implies \mathbf{X}^\top\mathbf{X}\boldsymbol\beta = \mathbf{X}^\top\mathbf{y}.$$

$$\boxed{\hat{\boldsymbol\beta} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}.}$$

The LS = MLE proof — the canonical theory question per [[L27-summary]]:

$$\log L(\boldsymbol\beta, \sigma^2) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - \mathbf{x}_i^\top\boldsymbol\beta)^2.$$

The first term doesn't depend on $\boldsymbol\beta$. $\sigma^2$ enters as a constant scaler. The only $\boldsymbol\beta$-dependent piece is $\sum_i (y_i - \mathbf{x}_i^\top\boldsymbol\beta)^2$, which the log-likelihood **maximizes by minimizing** — exactly the LS objective. ∎

## Insights & mental models

### Why LS in particular

> "Legendre got the minimization, Gauss got the distribution." — [[L05-linreg-1]]

Other choices of error distribution lead to other estimators: minimizing $\sum |y_i - \hat y_i|$ is the MLE under a **Laplace** distribution (symmetric, peakier, fat tails) and is **more robust to outliers** because the cost grows linearly, not quadratically. Minimizing the fourth power "would really, really penalize anything far away" — nobody recommends.

### Geometric picture

Two equivalent views of LS:
1. **Vertical distances:** $\varepsilon_i$ is the vertical drop from point to line; minimize $\sum \varepsilon_i^2$.
2. **Squared rectangles:** each residual becomes a square of side $|\varepsilon_i|$; minimize total area. This makes outlier sensitivity geometric — a big residual contributes a quadratically scaled area. With L1, minimize total *line lengths* (linear in residual) — far less pull.

### Closed form is special

> "Most of machine learning is finding good tricks to get to that peak… But in the case of linear regression with full-rank X — just right to the top." — [[L06-linreg-2]]

The MLE for OLS is solvable in one matrix inversion; almost no other model has this property. (The prof framed this as why we study OLS at all even though we want fancier things — it lets us derive every downstream property exactly.)

## Exam signals

> "I do generally like to keep one theory question. … assume an additive Gaussian error model … Show that maximum likelihood and least squares are equivalent in θ. … Not incredibly profound or difficult, but at least somewhat theoretical or mathy-ish. I'll try to include something along these lines, where it's mathy but not, you know, no weird spaces or fancy proofs." — [[L27-summary]]

> "For those who have taken GLMs or who understand what likelihood is — these are the same thing." — [[L06-linreg-2]]

> "I think the two prior-module exercises he believes have already been done: derive β̂ and show MLE = OLS under Gaussian errors. *If not, please do it.*" — [[L12-modelsel-1]]

## Pitfalls

- **Sign of the log-likelihood.** Maximizing $\log L$ = minimizing $-\log L$ = minimizing SSE (up to a constant). A student caught the prof on a sign during the L27 walkthrough — *"the text was written wrong, but that's okay."* On the exam: state the negation explicitly so a sign slip doesn't cost a point.
- **What you're assuming.** The MLE = LS equivalence relies on Gaussian, IID, mean-zero, common-variance errors (see [[gaussian-error-assumptions]]). If errors are Laplace, MLE is L1, not LS. State the assumption.
- **Constants don't matter.** $-(n/2)\log(2\pi\sigma^2)$ doesn't depend on $\boldsymbol\beta$, so it drops out of the optimization. Also true for the $1/(2\sigma^2)$ factor.
- **Uniqueness.** The closed form requires $\mathbf{X}^\top\mathbf{X}$ invertible — needs $n \ge p+1$ and no [[collinearity|collinearity]]. With perfect collinearity, infinitely many minimizers exist.

## Scope vs ISLR

- **In scope:** the derivation of $\hat{\boldsymbol\beta}$ and the LS = MLE argument under Gaussian errors. Do them by hand at least once (Exercise 6.1).
- **Look up in ISLR:** §3.2.1 (pp. 71–75) — multiple regression LS estimator (matrix form is in the appendix-style boxed equation; the textbook is light on the explicit derivation).
- **Skip in ISLR:** Bayesian / shrinkage interpretations of the likelihood are deferred — the L14 prof excluded the Gaussian/Laplace prior interpretation of ridge/lasso ("really don't think I'd put this on the test").

## Exercise instances

- Exercise6.1a — derive $\hat{\boldsymbol\beta} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ from RSS via differentiation
- Exercise6.1b — show the MLE equals the LS estimator under Gaussian errors

## How it might appear on the exam

- **The mathy-ish theory question.** Prof's verbatim template: "Show that maximum likelihood and least squares are equivalent in $\theta$" under additive Gaussian noise. Write the log-likelihood, drop the constants, identify SSE, conclude. About 6–10 lines of work.
- **Derive the normal equations.** Ask you to differentiate $\mathrm{RSS}(\boldsymbol\beta) = (\mathbf{y} - \mathbf{X}\boldsymbol\beta)^\top (\mathbf{y} - \mathbf{X}\boldsymbol\beta)$ w.r.t. $\boldsymbol\beta$ and solve.
- **What assumptions are needed?** Q3 of CE1 problem 2g — true/false on what a p-value does. The "everything in OLS rests on Gaussian iid errors" thread is exam-bait.
- **What if errors are Laplace?** Could ask conceptually: which loss does this lead to? Answer: $\sum|e_i|$ — robust to outliers.

## Related

- [[linear-regression]] — the parent model
- [[gaussian-error-assumptions]] — the assumptions both LS and MLE rely on
- [[design-matrix-and-hat-matrix]] — the algebra of the closed-form
- [[sampling-distribution-of-beta]] — the next thing you derive once you have the estimator
- [[multivariate-normal]] — the distributional foundation for the proof
