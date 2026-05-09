---
concept: ridge-vs-lasso-geometry
module: 06-modelsel
lectures: [L13, L14]
isl-ref: 6.2.2
exercises: []
related: [ridge-regression, lasso, elastic-net, regularization]
tags:
  - concept
  - module/06-modelsel
aliases:
  - ridge-lasso-geometry
  - L1-L2-geometry
  - constraint-region-picture
---

# Ridge vs lasso — geometric interpretation

The picture: in 2D, the RSS objective is a family of ellipses centered at the OLS estimate $\hat\beta_\text{OLS}$, and the regularization penalty defines a constraint region centered at the origin — a **circle for ridge** ($\beta_1^2 + \beta_2^2 \le s$) and a **diamond for lasso** ($|\beta_1| + |\beta_2| \le s$). The penalized solution is where the smallest RSS ellipse first *touches* the constraint region. The diamond's sharp corners on the axes are why lasso produces sparsity; the circle's smoothness is why ridge doesn't. ISL Fig 6.7. The prof flagged this picture explicitly as worth learning to read: *"It's worth learning how to interpret this geometric interpretation of ridge and lasso"* — [[L14-modelsel-3]].

## Definition (prof's framing)

The constraint formulation makes the geometry explicit:

**Lasso:** $\min_\beta \text{RSS}$ subject to $\sum_j |\beta_j| \le s$
**Ridge:** $\min_\beta \text{RSS}$ subject to $\sum_j \beta_j^2 \le s$

(These are the constrained-form duals of the penalized objectives; $s$ is monotonically related to $\lambda$ — large $\lambda$ ↔ small $s$.)

> "The red ellipses are the contours of the RSS. The solid blue areas are the constraint regions, $|\beta_1| + |\beta_2| \le s$ and $\beta_1^2 + \beta_2^2 \le s$. The explanation holds for $p > 2$, just harder to visualize." — slide deck (selection_regularization_presentation_lecture1.md)

## Notation & setup

- $\beta_1, \beta_2$ on the axes (2D for visualization; the same logic generalizes).
- **RSS contours**: ellipses centered at the unrestricted minimum $\hat\beta_\text{OLS}$. Each ellipse is a level set of constant RSS — the further from $\hat\beta_\text{OLS}$, the larger the RSS.
- **Constraint region**:
  - **L1 ball (lasso)**: $\{\beta : |\beta_1| + |\beta_2| \le s\}$ — a diamond (rotated square) with corners on the axes at $(\pm s, 0)$ and $(0, \pm s)$.
  - **L2 ball (ridge)**: $\{\beta : \beta_1^2 + \beta_2^2 \le s\}$ — a circle of radius $\sqrt{s}$.
- **Penalized solution**: the point where the smallest RSS ellipse that touches the constraint region meets it.

## The picture (verbal)

Draw $\hat\beta_\text{OLS}$ somewhere off-axis (typically in the first quadrant for the standard textbook figure). Centered there, draw concentric ellipses of equal RSS, expanding outward. Centered at the origin, draw the constraint region — diamond for lasso, circle for ridge.

Now imagine inflating the smallest RSS ellipse outward from $\hat\beta_\text{OLS}$ until it just kisses the constraint region. That contact point is the penalized estimate.

### Why lasso produces sparsity

The diamond has **sharp corners on the axes**. From a generic direction, the expanding ellipses tend to first hit the diamond at one of those corners — and a corner has $\beta_1 = 0$ or $\beta_2 = 0$.

> "It's the same idea that both this [ellipse] and this [diamond] have constant value of the objective — and where they intersect is one of these corners. It doesn't have to be, but they tend to." — [[L13-modelsel-2]]

Higher-dimensional analogue: the L1 ball has corners along every axis and edges along every coordinate hyperplane; ellipsoids tend to intersect those low-dimensional faces, zeroing some coordinates.

### Why ridge does not

The circle is **smooth everywhere** — no corners, no edges. The first contact between the expanding RSS ellipse and the L2 ball lands at a generic interior-direction point — neither $\beta_1$ nor $\beta_2$ is zero.

This is the geometric form of the same algebraic observation in [[ridge-regression]]: $\frac{d}{d\beta}\beta^2 = 0$ at $\beta = 0$, so there's no gradient pulling solutions onto the axes.

## Insights & mental models

### Constraint vs penalty — same picture, two formulations

The constrained-form ($\le s$) and the penalty-form ($\lambda \cdot \text{penalty}$) are equivalent by Lagrange duality. Each $\lambda > 0$ corresponds to some $s > 0$, and vice versa. Larger $\lambda$ ↔ smaller $s$ ↔ smaller / tighter constraint region ↔ more shrinkage. The geometric picture is easier to read in the constraint form — that's why the slides and ISL Fig 6.7 use it.

### "Capitalist vs socialist" reread

The prof's political analogy maps onto the geometry:

> "Lasso is like very capitalist. Just shoot everyone, let all the poor people die, let the one rich guy win. And L2 is more socialist." — [[L13-modelsel-2]]

> "[Lasso] encourages winners and losers, right? It's the capitalist regularization method. Whereas the ridge one… encourages ties, encourages that everyone has a vote and no one has zeros." — [[L14-modelsel-3]]

The diamond's corners ARE the "one rich guy" — all weight on a single coordinate. The circle's smoothness IS the "everyone has a vote" averaging.

### Correlated predictors — the asymmetry the picture explains

Imagine $\beta_1$ and $\beta_2$ correspond to two highly correlated predictors. The RSS ellipses become **very elongated** along the direction of correlation (you can trade off $\beta_1$ against $\beta_2$ with little RSS change). The expanding ellipse's first contact with:

- **L1 diamond** → snaps to one corner (one coefficient zero), and that choice is sensitive to small perturbations in the data — the data dataset might flip the corner.
- **L2 circle** → settles at an interior point that *averages* over the two — robust, neither coefficient zero.

This is the geometric form of the prof's verbal account ([[L13-modelsel-2]]):

> "If $\beta_1$ and $\beta_2$ are two things that are relatively correlated… we might end up where actually $\beta_1$ is just sent straight to 0 and it's all driven by $\beta_2$. … If we're in this ridge case, when we use ridge regression, we're actually averaging over the two."

### Why elastic net is the practical compromise

[[elastic-net]] uses $\alpha \cdot |\beta| + (1-\alpha) \cdot \beta^2$ as its penalty. The constraint region is a **rounded diamond** — corners along axes (still gives sparsity, like lasso) plus rounded edges (averages over correlated variables, like ridge). Geometrically: the centrist regularizer.

## Exam signals

> "It's worth learning how to interpret this geometric interpretation of ridge and lasso. You know, because it really shows how the two objectives combined… you can see why lasso is more likely to choose winners and losers, and ridge regression is more likely to give you averages or find compromises, do this socialist thing where everyone counts." — [[L14-modelsel-3]]

> "It's the same idea that both this [ellipse] and this [diamond] have constant value of the objective — and where they intersect is one of these corners. It doesn't have to be, but they tend to." — [[L13-modelsel-2]]

This is one of the **few** module-6 slide figures the prof explicitly told the class to learn to read. Likely exam pattern: a labeled or unlabeled version of ISL Fig 6.7 with a question like "which is ridge, which is lasso, and why does one set $\beta_1 = 0$?"

## Pitfalls

- **Treating the geometric picture as a proof.** It's a geometric *intuition* — useful for explaining sparsity, not a derivation. The actual sparsity argument runs through the L1 penalty's discontinuous gradient at zero (see [[lasso]]).
- **Forgetting the picture is in 2D.** The intuition extends to higher dimensions but the visual stops working. The prof flags: *"the explanation holds for $p > 2$, just harder to visualize"* — slide.
- **Confusing the constraint center with the penalty center.** Both are at the origin (constraint regions in $\beta$-space, both centered at $\beta = 0$). The RSS ellipse is centered at $\hat\beta_\text{OLS}$ — *not* at zero.
- **Reading "the diamond's corner is on the axis" as "lasso always zeros at least one $\beta$."** It typically does, but if $\hat\beta_\text{OLS}$ happens to lie along the diamond's edge perpendicular to one axis, you can land at an interior point with no zeros. Likely on the exam: the *typical* behaviour, not edge cases.
- **Believing the ellipse always grows from $\hat\beta_\text{OLS}$ — which is true for the constraint formulation but the picture mixes both.** Don't get tangled. The picture's job is to show "diamond corners → sparsity" and that's it.

## Scope vs ISLR

- **In scope:** the figure (ISL Fig 6.7); reading it correctly; the verbal account ("lasso corners → sparsity, ridge smooth → no sparsity"); the constraint-form formulation; the correlated-variable interpretation.
- **Look up in ISLR:** §6.2.2 pp. 256–264 ("Another Formulation for Ridge Regression and the Lasso" + "The Variable Selection Property of the Lasso" + "Comparing the Lasso and Ridge Regression"). Fig 6.7 is the canonical version of the picture.
- **Skip in ISLR:** the formal $\ell_p$-ball discussion / convex-analysis derivation. The prof gestured at the geometry and stopped.

## Exercise instances

(None directly — this concept is read off the ridge/lasso exercises, not given its own problem. Exercise6.5 (ridge on Credit) and Exercise6.6 (lasso on Credit) produce the coefficient patterns this picture explains.)

## How it might appear on the exam

- **Identify the figure.** Given an unlabeled version of ISL Fig 6.7, identify which panel is ridge and which is lasso, and state why one panel's solution lands on an axis (lasso, diamond corner) while the other doesn't (ridge, smooth circle).
- **Explain why lasso zeros coefficients but ridge doesn't.** Reference the geometric picture: corners on axes vs smooth boundary.
- **Choose-method given a description.** "We have many correlated predictors and want to keep them all" → ridge (smooth → averages). "We expect a sparse model with only ~5 of 100 predictors mattering" → lasso (corners → sparsity).
- **True / false.** "Lasso's L1 constraint region has corners on the coordinate axes." → True. "Ridge's L2 constraint region has corners on the coordinate axes." → False.
- **Conceptual essay (short).** *"Why does lasso perform variable selection but ridge does not? Refer to the geometric interpretation."* — answer: lasso's L1 ball has sharp corners on the axes, so the RSS ellipse tends to first contact at a corner where one coefficient is zero; ridge's L2 ball is smooth, so contact is at an interior direction with no zeros.

## Related

- [[ridge-regression]] — the L2 method whose smooth-ball constraint this picture explains.
- [[lasso]] — the L1 method whose diamond-corner-sparsity this picture explains.
- [[elastic-net]] — the rounded-diamond compromise visible by mixing the two boundaries.
- [[regularization]] — the cross-cutting Special this geometry illustrates for the L1/L2 cases.
