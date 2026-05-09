---
moc: 07-beyondlinear
title: Moving Beyond Linearity
lectures: [L16, L17]
isl-ch: 7
slides:
  - modules/7BeyondLinear/7BeyondLinear.md
exercises:
  - exercises/Exercise7/
tags:
  - moc
  - module/07-beyondlinear
---

# Module 07: Moving Beyond Linearity

One-and-a-half lectures (Mar 9 plus the first half of Mar 10) on a single trick: replace `x` with a basis expansion `b_j(x)` and keep doing linear regression. The atoms walk a difficulty ladder, polynomial → step → regression spline → smoothing spline → LOESS → GAM, culminating in [[generalized-additive-models]] and the same stack lifted to the binary case (logistic GAM). Load-bearing for the exam: the smoothing-spline λ direction T/F trap, building a spline design matrix by hand (Exercise 7.3 / 7.4), and reading a GAM plot.

## Lectures
- [[L16-beyondlinear-1]]: basis-functions framing; polynomial / step / regression splines (cubic + natural); smoothing splines + effective df; LOESS; GAM intro
- [[L17-trees-1]]: Module 7 wrap-up: lift the same toolkit to logistic regression (polynomial logistic, cubic-spline logistic, logistic GAM on the wage > $250k data) before pivoting to trees

## Concepts (atoms in this module)
- [[basis-functions]]: replace `x` with `b_j(x)` and run linear regression on the transformed columns; the unifying frame for the whole module
- [[polynomial-regression]]: still linear in β even when curvy in x; the standard simulation playground (also lives in module 03)
- [[step-functions]]: piecewise-constant fit on chosen intervals; "stupid but actually pretty common"; jumps at cutpoints
- [[regression-splines]]: piecewise cubics joined with continuous 0/1/2-derivatives at knots; truncated-power basis (x, x², x³, (x−c_j)³₊); natural splines force linear extrapolation past boundary knots
- [[smoothing-splines]]: minimize Σ(yᵢ − g(xᵢ))² + λ∫g''(t)²dt; **λ ↑ → smoother** (opposite direction from polynomial degree, flagged T/F trap); effective df = trace(S); choose λ by LOOCV
- [[local-regression]]: fit a local linear regression at every x₀ weighted by a Gaussian kernel; "smoothed K-nearest-neighbors"
- [[generalized-additive-models]]: y = β₀ + Σⱼ fⱼ(xⱼ) + ε with each fⱼ chosen freely (poly, spline, LOESS, indicator); additive, no interactions; logistic GAM is the same trick on the log-odds

## Cross-cutting concepts touched (Specials)
- [[bias-variance-tradeoff]]: first introduced module 02; this module's flexibility knob is the basis dimension / λ; spline df and LOESS bandwidth are the U-shape axes here
- [[regularization]]: first introduced module 06; smoothing splines are a *function-space* L2 penalty (λ on ∫g''²), same machinery as ridge, applied to the second derivative
- [[cross-validation]]: first introduced module 05; this module uses CV (often LOOCV) to pick λ for smoothing splines and the span for LOESS

## Exercises
- [[../../exercises/Exercise7]]: Exercise 7.1 (polynomial mpg ~ horsepower, plot test error vs degree); Exercise 7.3 (derive the natural-cubic-spline design matrix for `year` with one internal knot at 2006); Exercise 7.4 (build the cubic-spline design matrix by hand using the truncated-power basis, verify gam() agrees); Exercise 7.5 (fit a five-component GAM: cubic-spline displacement, polynomial horsepower, linear weight, smoothing-spline acceleration, factor origin)

## Out of scope (this module)
- **Natural-spline boundary-knot derivation** - "in other courses they go through the math of what these natural splines are. The book doesn't, so I won't either" - [[L16-beyondlinear-1]]. The linear-extrapolation *concept* is captured in [[regression-splines]].
- **B-spline basis machinery (the `bs` label)** - "I don't know why they call it BS"; cosmetic, not on the exam - [[L16-beyondlinear-1]]
- **Bezier / shipbuilding history of splines**: pedagogical context only - [[L16-beyondlinear-1]]

## ISLR pointer
Chapter 7: Moving Beyond Linearity. Deep treatment of in-scope concepts (polynomial, step, splines, smoothing splines, LOESS, GAM) is in `book/07-beyondlinear.md`. Atoms carry section-level `isl-ref:` pointers (7.2 step, 7.3 basis, 7.4 splines, 7.5 smoothing splines, 7.6 LOESS, 7.7 GAM).
