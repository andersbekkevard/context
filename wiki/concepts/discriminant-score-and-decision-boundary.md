---
concept: discriminant-score-and-decision-boundary
module: 04-classif
lectures: [L09, L27]
isl-ref: 4.4.1-4.4.3
exercises:
  - CE1 problem 3e  -  derive δ_k(x) from Bayes' rule, solve δ_0 = δ_1 for the boundary
related: [linear-discriminant-analysis, quadratic-discriminant-analysis, multivariate-normal, logistic-regression, classification-setup]
tags:
  - concept
  - module/04-classif
aliases:
  - decision boundary derivation
  - delta-k
---

# Discriminant score and decision-boundary derivation

The procedural atom for the prof's most-flagged exam pattern in module 4: **"given $\pi_k$, $\mu_k$, $\Sigma$ (or $\Sigma_k$), solve for the decision boundary."** The discriminant score $\delta_k(x)$ is what you maximize over $k$; setting $\delta_k = \delta_\ell$ and solving gives the boundary. Said *twice* on the exam-relevance flag list.

## Definition (prof's framing)

> "Same exact trick as finding the point that divides the classes in two. You do the same thing, only now in a higher dimension and then you get a line out of it, or a plane, whatever it is." - [[L09-classif-3]]

The discriminant score $\delta_k(x)$ is what's left of $\log\!\left(\pi_k f_k(x)\right)$ after dropping every term that doesn't depend on $k$. It is **not** a probability or likelihood, but it preserves the $\arg\max$ , same classification.

## Notation & setup

- $\pi_k$: class prior.
- $f_k(x)$: class-conditional density (multivariate Gaussian for LDA/QDA).
- $\delta_k(x)$: discriminant score for class $k$. Bigger $\delta_k$ → class $k$ wins.
- Decision boundary between classes $k$ and $\ell$: locus where $\delta_k(x) = \delta_\ell(x)$.

## Formula(s) to know cold

**LDA discriminant (1D):**

$$\delta_k(x) = x \cdot \frac{\mu_k}{\sigma^2} - \frac{\mu_k^2}{2\sigma^2} + \log\pi_k$$

**LDA discriminant (multivariate):**

$$\delta_k(x) = x^\top \Sigma^{-1}\mu_k - \tfrac{1}{2}\mu_k^\top \Sigma^{-1}\mu_k + \log\pi_k$$

**QDA discriminant (multivariate):**

$$\delta_k(x) = -\tfrac{1}{2}x^\top \Sigma_k^{-1} x + x^\top \Sigma_k^{-1}\mu_k - \tfrac{1}{2}\mu_k^\top \Sigma_k^{-1}\mu_k - \tfrac{1}{2}\log|\Sigma_k| + \log\pi_k$$

**Decision boundary** (binary, classes 0 and 1):

$$\delta_0(x) = \delta_1(x)$$

LDA → linear in $x$ (a hyperplane). QDA → quadratic in $x$ (a conic).

## Derivation recipe

The standard play, applied for either LDA or QDA:

1. **Start from Bayes' rule.** $\Pr(Y = k \mid X = x) \propto \pi_k f_k(x)$ (denominator is $k$-independent , drop it).
2. **Take logs.** $\log\Pr(Y = k \mid X = x) = \log\pi_k + \log f_k(x) + \text{const}_x$.
3. **Plug in the Gaussian density.** $\log f_k(x) = -\tfrac{p}{2}\log(2\pi) - \tfrac{1}{2}\log|\Sigma_k| - \tfrac{1}{2}(x - \mu_k)^\top \Sigma_k^{-1}(x - \mu_k)$ (use $\Sigma$ for LDA).
4. **Drop everything not depending on $k$.** The $-\tfrac{p}{2}\log(2\pi)$ goes. In LDA, $-\tfrac{1}{2}\log|\Sigma|$ also goes (no $k$); in QDA it stays. The $-\tfrac{1}{2}x^\top \Sigma^{-1} x$ piece: in LDA it has no $k$, **drop**; in QDA it has $k$, **keep** (this is where the quadratic comes from).
5. **Expand the surviving cross term:** $-\tfrac{1}{2}(x - \mu_k)^\top \Sigma^{-1}(x - \mu_k)$ → $x^\top \Sigma^{-1}\mu_k - \tfrac{1}{2}\mu_k^\top \Sigma^{-1}\mu_k$ (after dropping the $k$-free $x^\top \Sigma^{-1} x$ piece in LDA).
6. **What's left is $\delta_k(x)$.**

For the **boundary**, set $\delta_k(x) = \delta_\ell(x)$ and solve for $x$. For LDA in two classes:

$$x^\top \Sigma^{-1}(\mu_0 - \mu_1) - \tfrac{1}{2}\mu_0^\top \Sigma^{-1}\mu_0 + \tfrac{1}{2}\mu_1^\top \Sigma^{-1}\mu_1 + \log\pi_0 - \log\pi_1 = 0$$

A linear equation in $x$ → a hyperplane.

## Worked example , the prof's recurring template

$\mu_A = (1, 1)$, $\mu_B = (3, 3)$, $\Sigma = 2I$, equal priors.

$\Sigma^{-1} = \tfrac{1}{2}I$. The cross-term coefficient is $\Sigma^{-1}(\mu_A - \mu_B) = \tfrac{1}{2}(-2, -2) = (-1, -1)$.

The intercept terms:
- $-\tfrac{1}{2}\mu_A^\top \Sigma^{-1}\mu_A = -\tfrac{1}{2} \cdot \tfrac{1}{2}(1 + 1) = -\tfrac{1}{2}$.
- $+\tfrac{1}{2}\mu_B^\top \Sigma^{-1}\mu_B = +\tfrac{1}{2} \cdot \tfrac{1}{2}(9 + 9) = +\tfrac{9}{2}$.

Sum: $-\tfrac{1}{2} + \tfrac{9}{2} = 4$. Equal priors → $\log\pi_A - \log\pi_B = 0$.

Boundary: $-x_1 - x_2 + 4 = 0$, i.e. $\boxed{x_2 = 4 - x_1}$.

> "I didn't ask for a line. I just equated the two things and then I solved for them and then it became a line. I didn't tell the math to give me a line." - [[L09-classif-3]]

The prof flubbed this live (claimed two terms cancelled when they didn't), came back from break to fix it. Lesson:

> "When I was doing my PhD, one of my co-authors said he doesn't make a habit of doing algebra in public. I've heard his voice in my head a million times. It's a bit of a fool to do public algebra , your brain shuts off, you look like an idiot." - [[L09-classif-3]]

Show your work step-by-step on the exam.

## Insights & mental models

- **Boundary is a *consequence*, not a parameter.** In logistic regression you fit the boundary's slope. In LDA/QDA you fit class densities and the boundary falls out from $\delta_k = \delta_\ell$. - [[L09-classif-3]]
- **Equal priors, equal $\sigma$, two classes, 1D**: boundary at $(\mu_1 + \mu_2)/2$ , the midpoint. Verifiable shortcut.
- **Priors literally move the boundary.** Increasing $\pi_k$ by some amount slides the boundary **away** from class $k$ (more area is classified as $k$). Compute via $\log\pi_k - \log\pi_\ell$ in the boundary equation.
- **In multivariate LDA, the boundary depends on $\mu_0, \mu_1, \Sigma$ jointly.** Specifically, the boundary's normal direction is $\Sigma^{-1}(\mu_0 - \mu_1)$ , not just $\mu_0 - \mu_1$. Σ rotates and rescales.
- **For QDA, the cleanest representation** is the compact form $\delta_k(x) = -\tfrac{1}{2}(x - \mu_k)^\top \Sigma_k^{-1}(x - \mu_k) - \tfrac{1}{2}\log|\Sigma_k| + \log\pi_k$ , it's just (squared Mahalanobis distance) + (volume penalty) + (log-prior).

## Exam signals

> "And that's often an exam question. Or that would be a typical exam question. So you would, given an LDA setting and here are the values for the parameters , you have the pies, you have the mu's, and you'd have a value for the standard deviation , and then you would solve for where is the decision point." - [[L09-classif-3]]

> "This would be another kind of question that you could ask on an exam , like find the equation for the decision boundary between these two categories. Then you solve for the equation for the line or the plane or whatever it happens to be depending on the dimension of the X's." - [[L09-classif-3]]

> "That would be another question one could ask if I was so inspired , show that this leads to this thing. I don't know if that's a very interesting question to ask, but one could ask it." - [[L09-classif-3]] (re: deriving $\delta_k$ itself)

CE1 problem 3e is **exactly** this exam pattern in compulsory form: derive $\delta_k$, solve $\delta_0 = \delta_1$, get $ax_1 + bx_2 + c = 0$, solve for $x_2$.

## Pitfalls

- **Public-algebra slips.** The prof himself did one in lecture , write each line carefully on the exam, don't try to do steps in your head.
- **Forgetting to drop the $-\tfrac{1}{2}x^\top \Sigma^{-1}x$ in LDA.** It's $k$-independent there; drop it. In QDA, it depends on $k$ via $\Sigma_k$; keep it.
- **Mixing $\Sigma$ with $\Sigma^{-1}$.** The discriminant uses the inverse (precision). Don't drop the inverse.
- **Forgetting the $-\tfrac{1}{2}\log|\Sigma_k|$ term in QDA.** It survives because $|\Sigma_k|$ depends on $k$.
- **Sign error on the prior term.** $+\log\pi_k$ , bigger prior → bigger $\delta_k$ → more area classified as $k$.
- **Treating the boundary as $\mu_0 - \mu_1$ in 2D.** No , it's $\Sigma^{-1}(\mu_0 - \mu_1)$ that gives the normal direction. Only when $\Sigma = I$ (or scalar multiple) do they coincide.

## Scope vs ISLR

- **In scope:** Discriminant-score derivation from $\log(\pi_k f_k(x))$, why LDA is linear and QDA quadratic, decision-boundary derivation, the $\mu_A = (1,1)$ / $\mu_B = (3,3)$ / $\Sigma = 2I$ worked example, prior-shift effect.
- **Look up in ISLR:** §4.4.1 (1D LDA derivation, eq. 4.18), §4.4.2 (multivariate, eq. 4.24), §4.4.3 (QDA, eq. 4.28). pp. 145–155.
- **Skip in ISLR:** Fisher's eigenvalue derivation (slide deck "Optional", prof never lectured); detailed naive-Bayes-as-LDA-with-diagonal-Σ algebra (covered abstractly in [[naive-bayes]]).

## Exercise instances

- **CE1 problem 3e**: full procedural application: derive $\delta_k$ from Bayes' rule, solve $\delta_0 = \delta_1$ for the boundary in $ax_1 + bx_2 + c = 0$ form, plot it.

(All other CE1.3 / Exercise4.2 / Exercise4.6 problems on LDA/QDA implicitly require this derivation as a sub-step , see [[linear-discriminant-analysis]] / [[quadratic-discriminant-analysis]] for the full per-method exercise lists.)

## How it might appear on the exam

- **The flagged pattern (twice):** Given $\pi_k$, $\mu_k$, $\Sigma$ (or $\Sigma_k$), derive the boundary equation. Solve in the form $x_2 = bx_1 + a$ (2D) or just the threshold $x = c$ (1D).
- **Discriminant-score derivation:** "Show that for LDA, $\delta_k(x) = x^\top \Sigma^{-1}\mu_k - \tfrac{1}{2}\mu_k^\top \Sigma^{-1}\mu_k + \log\pi_k$." Step-by-step: log of $\pi_k f_k(x)$, drop $k$-free terms, expand the cross term. Emphasizes *why* the $x^2$ drops (no $k$).
- **"Where does the quadratic come from in QDA?"**: the contrast question. Walk through what cancels in LDA but doesn't in QDA.
- **Prior-shift question:** "If $\pi_k$ increases from 0.5 to 0.7, in which direction does the boundary move?" → away from class $k$.
- **Hand-classification:** Given $\delta_A(x_0)$ and $\delta_B(x_0)$ values for a specific test point, pick the larger.

## Related

- [[linear-discriminant-analysis]]: the model + estimators + scope discussion.
- [[quadratic-discriminant-analysis]]: same exercise, kept the $x^2$ piece.
- [[multivariate-normal]]: the density whose log produces $\delta_k$.
- [[logistic-regression]]: same linear log-odds form, different fitting; not derived this way but shares the linear boundary.
- [[classification-setup]]: Bayes-classifier framing that motivates maximizing $\delta_k$.
