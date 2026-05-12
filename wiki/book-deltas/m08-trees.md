---
title: "M08: Tree-Based Methods — Book delta"
module: 08-trees
isl-ch: 8
lectures: [L17, L18, L19]
---

# Module 08: Tree-Based Methods — Book delta

ISLP ch.8 (the trees / [[bagging]] / RF portion of `wiki/book/08-trees.md`) is the deep-treatment reference for module 8. Anders can flip to it for: the [[regression-tree|regression-tree]] split criterion (8.1)–(8.3), Algorithm 8.1 (build-out-then-prune-via-CV), the [[cost-complexity-pruning|cost-complexity]] penalty $C_\alpha(T) = Q(T) + \alpha|T|$ (8.4), the three classification impurities (8.5)–(8.7), the bagged-predictor formula $\hat f_{\text{bag}}$, the verbal OOB description ("on average each bagged tree uses around two-thirds of the observations"), the impurity-based [[variable-importance|variable-importance]] plot, and the [[random-forest|random-forest]] decorrelation trick with $m \approx \sqrt p$.

This file captures the **concrete, lookup-able artifacts the prof taught** that are not cleanly stated as a formula or derivation in ISLP §8.1–§8.2.2. The biggest items are the correlated-trees variance derivation (the *whole point* of why RF beats bagging — ISLP only states it verbally), the $(1-1/n)^n \to 1/e$ derivation for the [[out-of-bag-error|OOB]] fraction, the explicit definition of permutation-based variable importance (ISLP body text covers only impurity-based), the default $m = p/3$ for regression random forests, the scaled-cross-entropy ("deviance") formula, the weakest-link pruning mechanism, and the impurity-sensitivity worked example that motivates Gini/entropy over misclassification.

---

## 1. The correlated-trees variance derivation

[[L19-boosting-1|L19]], [[bagging]], [[random-forest]]

The single load-bearing math result of the [[bagging]] → [[random-forest|random-forest]] pivot. ISLP §8.2.2 explains the idea verbally ("averaging many highly correlated quantities does not lead to as large of a reduction in variance as averaging many uncorrelated quantities") but does **not** write down the formula. The prof did it on the board in [[L19-boosting-1|L19]], and the slide deck has the derivation in full ("Recall: variance for independent / correlated datasets").

**Setup.** Let $X_1, \ldots, X_B$ be $B$ random variables with common variance
$$\mathrm{Var}(X_i) = \sigma^2$$
and pairwise correlation $\rho$, so
$$\mathrm{Cov}(X_i, X_j) = \rho \sigma^2 \quad \text{for } i \neq j.$$

**Derivation.** Expand the variance of the mean into diagonal + off-diagonal covariance terms:

$$
\begin{aligned}
\mathrm{Var}\!\Big(\tfrac{1}{B}\textstyle\sum_{i=1}^B X_i\Big)
&= \frac{1}{B^2}\,\mathrm{Var}\!\Big(\sum_{i=1}^B X_i\Big) \\
&= \frac{1}{B^2}\Big[\sum_{i=1}^B \mathrm{Var}(X_i) \;+\; 2\sum_{i<j}\mathrm{Cov}(X_i, X_j)\Big] \\
&= \frac{1}{B^2}\Big[\,B\sigma^2 \;+\; 2 \cdot \tfrac{B(B-1)}{2}\,\rho\sigma^2\,\Big] \\
&= \frac{\sigma^2}{B} \;+\; \frac{B-1}{B}\,\rho\sigma^2 \\
&= \rho\sigma^2 \;+\; \frac{1-\rho}{B}\,\sigma^2.
\end{aligned}
$$

The off-diagonal sum has $\binom{B}{2} = \tfrac{1}{2}B(B-1)$ pairs, each contributing $\rho\sigma^2$, hence the factor $2 \cdot \tfrac{B(B-1)}{2} = B(B-1)$ before division by $B^2$.

**The boxed result:**

$$
\boxed{\;\mathrm{Var}\!\Big(\tfrac{1}{B}\textstyle\sum_{i=1}^B X_i\Big) \;=\; \rho\sigma^2 \;+\; \frac{1-\rho}{B}\sigma^2\;}
$$

**Sanity checks** (the slide deck flagged these explicitly, "Check: $\rho = 0$ and $\rho = 1$?"):

- $\rho = 0$ (IID): collapses to $\sigma^2 / B$, the textbook $\mathrm{Var}(\bar X) = \sigma^2/n$ result. Variance shrinks linearly in $B$.
- $\rho = 1$ (perfect correlation): collapses to $\sigma^2$. **No variance reduction at all from averaging** — every tree says the same thing.
- General $0 < \rho < 1$: a fixed floor $\rho\sigma^2$ that does not depend on $B$, plus a decaying term $\sigma^2(1-\rho)/B \to 0$.

**The point of the derivation** (the prof verbatim, [[L19-boosting-1|L19]]):

> "This thing does not shrink with respect to $B$. No matter how many times we make different trees, if they're all correlated the same way, then there's a portion that just never improves."

So bagging $B$ trees has a **hard floor** at $\rho\sigma^2$, the residual variance of any single tree times the pairwise correlation. The only way past the floor is to lower $\rho$. **Random forests do exactly that** by restricting each split to a random subset of $m$ predictors, which prevents the strong predictors from being chosen at the same nodes in every tree, decorrelating the ensemble.

**Why this is delta-worthy.** ISLP §8.2.2 gives the qualitative claim but not the formula. The 2024/2025-style exam question "Why are random forests better than bagging?" expects the explicit $\rho\sigma^2 + (1-\rho)\sigma^2/B$ decomposition and the verbal "the first term is a floor in $B$" interpretation — the prof flagged this as one of the high-likelihood mathy theory questions.

---

## 2. The OOB fraction: $(1 - 1/n)^n \to 1/e$

[[L18-trees-2|L18]], [[L19-boosting-1|L19]], [[out-of-bag-error]], [[bootstrap]]

ISLP §8.2.1 reports the result verbally ("on average, each bagged tree makes use of around two-thirds of the observations") and footnotes Exercise 5.2 for the derivation. The actual derivation is exam-flagged (Exercise 8.1d explicitly says *"the result from RecEx5-Problem 4c can be used"*) so it lives here.

**Setup.** Bootstrap sample of size $n$ drawn with replacement from $n$ original observations. Probability observation $i$ is **not** drawn on any single draw is $(n-1)/n = 1 - 1/n$. Probability observation $i$ is **not drawn in any of the $n$ draws**:

$$
P(\text{obs }i \text{ is OOB}) = \Big(1 - \tfrac{1}{n}\Big)^n.
$$

**Limit.** Using $\lim_{n \to \infty}(1 + x/n)^n = e^x$ with $x = -1$:

$$
\Big(1 - \tfrac{1}{n}\Big)^n \;\xrightarrow{n \to \infty}\; e^{-1} \;=\; \frac{1}{e} \;\approx\; 0.368.
$$

**Consequence.** For large $n$:

- **~36.8% of observations are OOB** for any given bootstrap sample / tree.
- **~63.2% of observations are in-bag** for any given tree.
- For $B$ trees, each observation $i$ is OOB for roughly $B/e \approx B/3$ trees. The OOB prediction $\hat y_i^{\text{OOB}}$ averages (regression) or majority-votes (classification) those $\approx B/3$ trees' predictions for $x_i$.

Convergence is from above and fast:

| $n$ | $(1 - 1/n)^n$ |
|---|---|
| 10 | 0.3487 |
| 100 | 0.3660 |
| 1000 | 0.3677 |
| $\infty$ | 0.3679 |

The prof's verbal short-hand from [[L19-boosting-1|L19]]:

> "It's approximately B divided by 3."

meaning, for any given observation $i$, roughly $B/3$ trees can vote on it because $i$ is OOB for that fraction.

**Why this is delta-worthy.** ISLP reports the conclusion but defers the derivation to an end-of-chapter exercise in ch.5. The prof drilled it as an in-class hand-calc, and Exercise 8.1d ports it directly into module 8.

---

## 3. Permutation-based (randomization) variable importance

[[L18-trees-2|L18]], [[L19-boosting-1|L19]], [[variable-importance]]

ISLP §8.2.1 ("Variable Importance Measures") covers **only** the impurity-based version: total decrease in RSS (regression) / Gini (classification) due to splits on $X_j$, averaged over $B$ trees. The prof covered **both** flavors, and stated a preference for permutation importance, *"because it makes more sense."* The permutation flavor is the delta.

**Permutation importance, exact procedure** (uses the OOB sample):

For each predictor $j = 1, \ldots, p$:

1. Run the OOB observations through the trained ensemble. For each tree $b$, evaluate the per-tree OOB error on the OOB indices of that tree:
   $$
   \mathrm{err}_b^{(0)} \;=\; \frac{1}{|\mathrm{OOB}_b|} \sum_{i \in \mathrm{OOB}_b} \mathcal{L}\!\big(y_i, \hat f^{*b}(x_i)\big),
   $$
   where $\mathcal{L}$ is squared error (regression) or $\mathbb{1}[y_i \neq \hat f^{*b}(x_i)]$ (classification).

2. **Permute** the values of predictor $j$ across the OOB sample (whole column shuffled); call the permuted feature matrix $\tilde X^{(j)}$. Keep all other predictors, the labels, and the trained trees fixed.

3. Re-evaluate the per-tree OOB error on the corrupted features:
   $$
   \mathrm{err}_b^{(j)} \;=\; \frac{1}{|\mathrm{OOB}_b|} \sum_{i \in \mathrm{OOB}_b} \mathcal{L}\!\big(y_i, \hat f^{*b}(\tilde x_i^{(j)})\big).
   $$

4. Compute the per-tree performance drop $\Delta_b^{(j)} = \mathrm{err}_b^{(j)} - \mathrm{err}_b^{(0)}$, average across trees, and (optionally) divide by the standard deviation of the differences:
   $$
   \mathrm{VI}_j^{\text{perm}} \;=\; \frac{\bar \Delta^{(j)}}{\mathrm{sd}(\Delta_1^{(j)}, \ldots, \Delta_B^{(j)})}, \qquad
   \bar \Delta^{(j)} \;=\; \frac{1}{B}\sum_{b=1}^B \Delta_b^{(j)}.
   $$

**The reading.** If $X_j$ carries real signal, scrambling its values destroys per-row predictive ability and the OOB error spikes ($\bar\Delta^{(j)} \gg 0$). If $X_j$ is irrelevant, scrambling does nothing ($\bar\Delta^{(j)} \approx 0$). Reported under the names **MeanDecreaseAccuracy** (classification) or **%IncMSE** (regression) in R's `randomForest` output.

**The prof's verbal definition ([[L19-boosting-1|L19]]):**

> "If $X_j$ is important, permuting the observations will decrease the performance a lot. If it doesn't matter, then it won't matter."

**Why prefer permutation over impurity-based?** (Prof's *"makes more sense"* substance, partly verbatim, partly the standard mechanism he gestured at.)

- **Impurity-based importance is biased toward predictors with many possible split points.** Continuous predictors and high-cardinality categoricals have more chances to be picked as a split and to reduce impurity, so they look artificially important. Permutation importance avoids this bias because it directly tests whether knowing the predictor's value helps prediction in the OOB set.
- **Permutation importance is independent of the split criterion** used at training. Impurity-based importance changes if you switch Gini ↔ entropy at training; permutation importance reads off the same OOB sample either way.

**When the two flavors disagree.** Empirically: top predictors tend to agree across both flavors (slide deck and [[L19-boosting-1|L19]] showed both side-by-side on `mtcars` and the brain-injury data — top 2 always preserved, bottom predictor always preserved). The middle of the ranking is where the methods diverge. Prof's framing ([[L19-boosting-1|L19]]):

> "They do measure slightly different things, so you don't expect them to be the same. … They do capture different aspects of what's being measured."

**Why this is delta-worthy.** ISLP §8.2.1 body text describes only the impurity-based plot (Figure 8.9, Heart data, mean decrease in Gini). The permutation procedure, the OOB-permutation mechanism, and the prof's preference are entirely absent from the body of ch.8 — they live in the slide deck and lectures.

---

## 4. Default $m = p/3$ for regression random forests

[[L19-boosting-1|L19]], [[random-forest]]

ISLP §8.2.2 names the classification default $m \approx \sqrt p$ (Heart data, "4 out of the 13 for the `Heart` data") and notes "using a small value of $m$ in building a random forest will typically be helpful when we have a large number of correlated predictors," but does **not** state the **regression** default explicitly in the main text. The prof was emphatic and the slide deck spelled it out:

$$
m \;=\; \sqrt{p} \quad \text{(classification)}, \qquad m \;=\; \frac{p}{3} \quad \text{(regression)}.
$$

(Bagging is the special case $m = p$.)

**Direction-of-tuning.** Smaller $m$ → more decorrelation between trees → lower $\rho$ in the variance formula of §1 → lower variance floor. Recommended when predictors are highly correlated. Larger $m$ → individual trees are stronger but more correlated → variance floor stays high.

**Why this is delta-worthy.** ISLP's main text fixes $m = \sqrt p$ in its examples; the regression default $p/3$ is the kind of fact that's specifically asked for in past exam keys ("justify your choice of `mtry`"). The 2025 exam key reproduced in [[L27-summary|L27]] gives full marks for "$\sqrt p$ with $p = 22$ → use 4 or 5"; the analogous regression answer needs $p/3$.

---

## 5. Weakest-link pruning: explicit mechanism

[[L17-trees-1|L17]], [[L18-trees-2|L18]], [[cost-complexity-pruning]]

ISLP §8.1.1 (after equation 8.4) says only:

> "It turns out that as we increase $\alpha$ from zero in (8.4), branches get pruned from the tree in a nested and predictable fashion."

That's a black-box statement of the existence of a nested sequence. The prof spelled out the **mechanism** by which the sequence is generated (slide deck "weakest-link pruning" + [[L17-trees-1|L17]]/[[L18-trees-2|L18]] verbal):

**The procedure.** Starting from the maximal tree $T_0$:

1. For every internal node $v$, compute the increase in training loss that would result from collapsing $v$ (turning $v$ into a leaf, dropping the subtree rooted at $v$). For regression, that's the increase in RSS; for classification (at the prune step), the increase in misclassification count.

2. Among all internal nodes, pick the one whose collapse increases the loss the **least** — the "weakest link" — and collapse it. The result is the next subtree $T_1$ with $|T_1| = |T_0| - 1$ terminal nodes (or fewer, if the collapse merges multiple leaves).

3. Repeat. Each step removes one weakest link and produces the next subtree in the sequence:

$$
T_0 \;\supset\; T_1 \;\supset\; T_2 \;\supset\; \cdots \;\supset\; T_{\text{root}}.
$$

**Theorem (stated, not proved in lecture).** For each $\alpha \ge 0$, the subtree $T_\alpha$ that minimizes $C_\alpha(T) = Q(T) + \alpha |T|$ over all subtrees $T \subset T_0$ is one of the trees in this nested sequence. Equivalently, every member of the weakest-link sequence is optimal for some interval $[\alpha_k, \alpha_{k+1})$ of complexity penalties. The map $\alpha \mapsto T_\alpha$ is therefore a step function on the finite set $\{T_0, T_1, \ldots, T_{\text{root}}\}$.

The proof is **explicitly out of scope** (the prof linked Bo Lindqvist's MA8701 note on the slide as the "for-those-who-want-it" reference and skipped it in lecture).

**Practical consequence: plot tree size, not $\alpha$.** Because the map $\alpha \mapsto T_\alpha$ is a step function, the relationship between $\alpha$ and the cardinality $|T_\alpha|$ is one-to-one on the breakpoints but uninformative between them. CV plots therefore display held-out error against **tree size** $|T|$, not $\alpha$ directly (slide deck verbatim: *"we can plot the CV error as a function of tree size (instead of $\alpha$ — why?)"*). The CV minimum is read off the size axis directly.

**Why this is delta-worthy.** ISLP states the existence of the nested sequence; the prof gave the algorithm (collapse the weakest link, repeat) that generates it. Worth an exam-pseudocode answer.

---

## 6. The deviance / scaled-cross-entropy formula

[[L18-trees-2|L18]], [[classification-tree]]

ISLP §8.1.2 gives the **cross-entropy** as

$$
D \;=\; -\sum_{k=1}^K \hat p_{mk} \log \hat p_{mk} \tag{ISLP 8.7}
$$

and the lab (§8.3.1) gives the **deviance** in passing as $-2\sum_m \sum_k n_{mk} \log \hat p_{mk}$. But the relationship between the two — the prof's "the R `tree` package's `split='deviance'` *is* cross-entropy, up to a constant factor and the population-vs-proportion scaling" — is a flagged exam trap not stated cleanly in ISLP's main text.

**The slide-deck definition.** For region $j$ with $N_j$ observations and $n_{jk}$ observations of class $k$ (so $\hat p_{jk} = n_{jk}/N_j$), the **deviance** is

$$
\boxed{\;\mathrm{deviance}_j \;=\; -2 \sum_{k=1}^K n_{jk} \log \hat p_{jk}\;}
$$

which is $-2 \log L$ for the multinomial likelihood of the node, treating $\hat p_{jk}$ as the MLE.

**Connection to cross-entropy.** Substitute $n_{jk} = N_j \hat p_{jk}$:

$$
\mathrm{deviance}_j \;=\; -2 N_j \sum_k \hat p_{jk} \log \hat p_{jk} \;=\; 2 N_j \cdot D_j,
$$

where $D_j$ is the cross-entropy of region $j$. So **deviance is a scaled cross-entropy**: scaled by the per-region $2N_j$, which weights larger regions more heavily (the natural weighting for the global loss).

**Implication for `split="deviance"`.** Calling `tree(..., split="deviance")` minimizes the **total** node-summed deviance across the resulting children, which (after the $2N_j$ scaling collapses across the global sum) gives the same split decisions as splitting on weighted cross-entropy. Slide-deck verbatim: *"thus `split='deviance'` implies that we split according to the entropy criterion."*

**Why this is delta-worthy.** ISLP gives the cross-entropy in the body and the deviance in the lab without a derivation linking them; the prof drilled the relationship as one of the named "this is what `deviance` is" facts for output interpretation.

---

## 7. The impurity-sensitivity example: why misclassification is a poor split criterion

[[L18-trees-2|L18]], [[classification-tree]]

ISLP §8.1.2 states the punchline:

> "However, it turns out that classification error is not sufficiently sensitive for tree-growing, and in practice two other measures are preferable."

and demonstrates the related "both children predict the same class" phenomenon on the Heart data (the `RestECG<1` split). It does **not** give the canonical two-class numerical example that the prof did. The slide deck spells it out and the prof walked through it in [[L18-trees-2|L18]].

**The two-class setup.** Parent node with 800 observations: 400 of class A, 400 of class B — written $(400, 400)$. Consider two candidate splits:

- **Split 1.** Left child: $(100, 300)$, right child: $(300, 100)$.
- **Split 2.** Left child: $(200, 400)$, right child: $(200, 0)$.

**Misclassification error of each split** (using $E_m = 1 - \max_k \hat p_{mk}$, weighted by node sizes $N_m/N_{\text{parent}}$):

- Split 1: left predicts B with $\hat p = 300/400 = 0.75$, $E_L = 0.25$; right predicts A with $\hat p = 300/400 = 0.75$, $E_R = 0.25$. Weighted: $\tfrac{400}{800}(0.25) + \tfrac{400}{800}(0.25) = 0.25$.
- Split 2: left predicts B with $\hat p = 400/600 \approx 0.667$, $E_L = 1/3$; right predicts A with $\hat p = 200/200 = 1$, $E_R = 0$. Weighted: $\tfrac{600}{800}(1/3) + \tfrac{200}{800}(0) = 0.25$.

**Both splits have the same weighted misclassification rate of 25%.** Misclassification cannot distinguish them.

**Gini index of each split.** Using $G_m = \sum_k \hat p_{mk}(1 - \hat p_{mk})$ and weighting by $N_m/N_{\text{parent}}$:

- Split 1: each child has $G = 2(0.25)(0.75) = 0.375$. Weighted Gini = $0.375$.
- Split 2: left $G = 2(2/3)(1/3) = 4/9 \approx 0.444$, right $G = 0$ (pure node). Weighted = $\tfrac{600}{800}(4/9) + \tfrac{200}{800}(0) = \tfrac{3}{4} \cdot \tfrac{4}{9} = \tfrac{1}{3} \approx 0.333$.

**Gini prefers Split 2** ($0.333 < 0.375$), because Split 2 produces a **pure node** (right child has zero class B). Misclassification can't see this.

**Cross-entropy is similar in shape.** Both Gini and entropy are strictly concave in $\hat p$, both have global maximum at $\hat p = 0.5$ and zeroes at $\hat p \in \{0, 1\}$, so both penalize "in-the-middle" probabilities more than misclassification (which has a kink at $\hat p = 0.5$ and is otherwise piecewise linear).

**The slide-deck plot** (`p` on x-axis, three curves: misclassification = $\min(p, 1-p)$, Gini = $2p(1-p)$, scaled entropy):

- Misclassification is a tent function, peak at $p = 0.5$, slope $\pm 1$ on either side.
- Gini is a parabola, peak $0.5$ at $p = 0.5$.
- Entropy (scaled to match peak) is taller in the middle, drops more sharply near the boundaries.

The takeaway: Gini and entropy reward purity gains even when the **prediction** at each child doesn't change, because the **probabilities** at the children get more extreme. Misclassification only sees whether the predicted class changed.

**Why this is delta-worthy.** The (400, 400) → (100, 300)/(300, 100) vs (200, 400)/(200, 0) example is the cleanest numerical demonstration of the impurity-sensitivity argument, and the prof flagged it as the load-bearing reason to use Gini or entropy for splitting. ISLP gives the punchline without this calculation.

---

## 8. The differentiability footnote on impurity criteria

[[L18-trees-2|L18]], [[classification-tree]]

The slide deck flags a second reason to prefer Gini / cross-entropy over misclassification:

> "The Gini index and Entropy are differentiable (preferred for numerical optimization!)"

**The math.** Treating $\hat p_{mk}$ as continuous:

- $\partial G / \partial \hat p_{mk} = 1 - 2 \hat p_{mk}$. Continuous everywhere.
- $\partial D / \partial \hat p_{mk} = -\log \hat p_{mk} - 1$. Continuous on $(0, 1]$.
- $\partial E / \partial \hat p_{mk}$ has a **kink** at $\hat p_{mk} = \max_k \hat p_{mk}$, i.e., where the argmax flips. Not differentiable there.

The prof flagged this as the **weaker** of the two reasons ([[L18-trees-2|L18]]):

> "The reality is now actually [misclassification] would be also pretty easy to make differentiable. Like … we could just do a soft max."

So the load-bearing reason is the impurity-sensitivity argument of §7. The differentiability point is a secondary, "easier-for-gradient-style-optimizers" argument that the prof noted is rendered moot by modern soft-relaxation tricks.

**Why this is delta-worthy.** ISLP doesn't discuss differentiability of the three criteria at all; the slide deck does, and the prof has an opinion on which argument actually matters. Useful for a T/F exam question: "Gini is differentiable everywhere on $[0, 1]$" → True. "Misclassification has a kink at $\hat p = 0.5$" → True (for $K = 2$).

---

## 9. Categorical predictors: the $2^{q-1} - 1$ partition count and the binary-outcome ordering trick

[[L18-trees-2|L18]], [[classification-tree]]

ISLP §8.1.2 mentions that splits on qualitative variables "amount to assigning some of the qualitative values to one branch and the remaining to the other," and the `Thal:a` / `ChestPain:bc` notations in Figure 8.6 show example splits. ISLP does **not** state the combinatorial count or the binary-outcome reduction trick. The slide deck does, and the prof flagged them.

**Partition count.** For a predictor with $q$ unordered levels, the number of nontrivial binary partitions is

$$
\boxed{\;\#\text{splits} \;=\; 2^{q-1} - 1\;}
$$

(Each partition assigns each of $q$ levels to one of two groups, $2^q$ total. Subtract 2 for the empty / full assignments and divide by 2 for the left-right symmetry: $(2^q - 2)/2 = 2^{q-1} - 1$.)

**Implication.** Categoricals with many levels are dangerous in two senses:

1. **Variance.** With many possible splits, the tree is sensitive to small data perturbations — high-cardinality categoricals are a variance source. The slide deck verbatim: *"Try to avoid predictors with very many levels!"*
2. **Spurious importance.** Impurity-based variable importance (see §3 above) is biased toward predictors with many split candidates, exactly the high-cardinality categoricals. Permutation importance is the fix.

**The binary-outcome ordering trick.** For a $K = 2$ classification problem, you can avoid the $2^{q-1} - 1$ exhaustive search:

1. For each level $\ell$ of the categorical, compute the proportion of class 1 observations at that level: $\hat p_\ell = (\text{class-1 count at level } \ell) / (\text{total at level } \ell)$.
2. Order the levels by $\hat p_\ell$.
3. Treat the now-ordered levels as an **ordinal** predictor and search only the $q - 1$ ordered cuts.

**Theorem (slide-deck claim, no proof given).** This procedure recovers the same split as the exhaustive Gini-minimizing search.

For multi-class outcomes ($K > 2$) and continuous outcomes, the trick does not apply and the exhaustive search remains $O(2^{q-1})$.

**Why this is delta-worthy.** The $2^{q-1} - 1$ count is the kind of small combinatorial fact that's exam-bait (T/F: "A 6-level categorical predictor has 31 possible binary splits" → True since $2^5 - 1 = 31$). ISLP doesn't state it.

---

## 10. The "useless-looking split" mechanism

[[L18-trees-2|L18]], [[cost-complexity-pruning]]

ISLP §8.1.2 describes the *"both children predict the same class"* split on the `RestECG<1` example and explains it as a node-purity improvement. ISLP §8.1.1 (Tree Pruning paragraph) gives the *"too short-sighted"* argument for why you don't stop early: *"a seemingly worthless split early on in the tree might be followed by a very good split."* The prof linked these two into one mechanism ([[L18-trees-2|L18]]): **the same split that looks useless by misclassification can be the gateway to a good later split, so any early-stopping rule based on misclassification will systematically fail.**

**Numerical example (the prof's [[L18-trees-2|L18]] walk-through).** Parent node: 70 class A, 30 class B.

- Parent Gini: $2(0.7)(0.3) = 0.42$. Parent misclassification: $0.3$. Predicts A.
- Split candidate: left child (40A, 5B), right child (30A, 25B).
- Left child: $\hat p_A = 40/45 \approx 0.89$. Gini $= 2(0.89)(0.11) \approx 0.196$. Misclass: $0.111$. Predicts A.
- Right child: $\hat p_A = 30/55 \approx 0.545$. Gini $= 2(0.545)(0.455) \approx 0.496$. Misclass: $0.455$. Predicts A.

**Weighted child misclassification** = $\tfrac{45}{100}(0.111) + \tfrac{55}{100}(0.455) = 0.05 + 0.25 = 0.30$. Same as parent. **No improvement by misclassification.**

**Weighted child Gini** = $\tfrac{45}{100}(0.196) + \tfrac{55}{100}(0.496) = 0.088 + 0.273 = 0.361 < 0.42$. **Improvement under Gini.**

Both children still predict A — the leaf labels are unchanged — but the **probability of being A** is much more confident in the left child (89%) than the right (55%). The Gini-improvement comes entirely from the increased certainty on the left, even though the prediction label is the same.

**Why this is delta-worthy / Where this lands.** Combines the "useless-looking split" example into a single hand-calculation that demonstrates both:

1. Why misclassification is a bad split criterion (it's blind to this purity improvement).
2. Why early stopping by RSS / misclassification threshold fails (the left child of this split now sits at 40A/5B — almost pure — and a single later split there can produce a 40A/0B leaf, paying off the parent split's "uselessness").

The prof's verbatim [[L18-trees-2|L18]] conclusion:

> "It put kind of like the shitty part of the parent node in one side and the more confident version on the left side."

ISLP gives the qualitative version of this on `RestECG<1`; the prof's numerical-example version is the cleanest exam-ready demonstration.

---

## 11. Boston worked example: numerical hierarchy single-tree → bagging → RF

[[L19-boosting-1|L19]], [[random-forest]]

A concrete number sequence the prof anchored as the "this is what variance reduction looks like" headline (slide deck + [[L19-boosting-1|L19]] board work). ISLP shows the bagging-vs-RF curve on **gene-expression** data (Figure 8.10) and the Heart data (Figure 8.8) — *not* on Boston housing regression, where the contrast is cleaner. The Boston numbers are the prof's canonical demonstration:

| Method | `mtry` ($m$) | Test MSE |
|---|---|---|
| Single regression tree (pruned to 6 leaves) | — | $\approx 35$ |
| Bagging (`randomForest` with `mtry = 13`) | $p = 13$ | $\approx 23.67$ |
| Random forest | $m = p/3 \approx 4$ | $\approx 18.9$ |

The numbers come from `MASS::Boston`, target `medv` (median home value), $p = 13$ predictors, $n/2$ train / $n/2$ test split with `set.seed(1)`.

The story ([[L19-boosting-1|L19]], prof verbatim):

> "It's improved with respect to simple bagging, it's likely because of the more diversity in our trees. And now we are talking about a forest and no longer a tree, and we have trees, trees live in a forest, and we're adding more diversity to our trees so that it gets better."

**Why this is delta-worthy.** Specific MSE numbers for the standard $35 \to 23.67 \to 18.9$ sequence — useful as anchors for a method-comparison exam question of the form "ranked these three approaches by test MSE on Boston housing, explain why." ISLP's worked Boston example in the lab (§8.3.3) gives different numbers under a different split (28.07 / 14.63 / 20.04, with RF actually losing slightly to bagging in their seed) — the slide-deck numbers are what the prof drilled in lecture.

---

## 12. Hyperparameter taxonomy across the tree-ensemble family

[[L19-boosting-1|L19]], [[L27-summary|L27]], [[random-forest]]

A direct comparison table the prof emphasized verbally and that recurs in past-exam keys. ISLP discusses each hyperparameter in turn but does not lay them out side-by-side with the *"tune vs. don't tune"* labels that the prof flagged as exam-bait.

| Method | Parameter | Role | Tune? | Default / typical |
|---|---|---|---|---|
| Single tree | $\alpha$ (cost-complexity) | Tree size penalty | **Yes** (K-fold CV) | Chosen by `cv.tree` |
| Bagging | $B$ = `ntree` | Number of trees | **No** ("use enough") | 500–1000 |
| Random forest | $B$ | Number of trees | **No** | 500–1000 |
| Random forest | $m$ = `mtry` | Predictors per split | **Yes** (the real one) | $\sqrt p$ (classif), $p/3$ (regr) |
| Random forest | Tree depth / `nodesize` | Individual-tree complexity | No (grow unpruned) | Deep / unpruned |

**The hard rule** (prof verbatim, [[L19-boosting-1|L19]], used in 2023 / 2025 exam keys):

> "It just has to be enough … you don't typically estimate this, you don't typically run cross-validation."

referring specifically to $B$ in bagging / RF. Past exam keys deduct points for "I chose `ntree=500` by CV" — the correct justification is "$B$ is not a tuning parameter; pick large enough that the OOB error plateau is reached."

**Why this is delta-worthy.** ISLP §8.2.1 says *"The number of trees $B$ is not a critical parameter with bagging; using a very large value of $B$ will not lead to overfitting"* — true but understated. The prof's rule is sharper: don't tune $B$, period; tune $m$. The taxonomy form is what the exam key expects to see.

---

## 13. Bias-reduction footnote for bagging

[[L11-resample-2|L11]], [[bagging]]

ISLP §8.2.1 frames bagging entirely as a **variance-reduction** procedure (its opening sentence: *"Bootstrap aggregation, or bagging, is a general-purpose procedure for reducing the variance of a statistical learning method."*) The prof added — and flagged — a less-commonly-stated point: bagging can **also reduce bias**, not just variance ([[L11-resample-2|L11]]):

> "By doing that, even though you're always just resampling the same data, you can actually remove bias from your model."

The mechanism (slide-deck-implicit, not in ISLP): a single tree's greedy splits make locally-optimal but globally-suboptimal choices. The greedy bias of any one tree is some function of which observations entered the bootstrap sample. Averaging across many bootstrap samples partially averages over this **search-path bias**, not just over noise in the data. The result is that the bagged ensemble can have **lower bias** than a single tree fit on the full data, in addition to lower variance.

The prof was clear this is the secondary argument; the variance-reduction story is the headline. But the bias point comes up as a T/F exam trap: "Bagging only reduces variance, not bias" → **False**.

**Why this is delta-worthy.** ISLP frames bagging exclusively as a variance device. The prof flagged the bias-reduction footnote as a real, examinable consequence.

---

## Notation and naming differences

- **Region index.** ISLP uses $m$ (and sometimes $j$ in the regression-tree exposition); the prof uses both $m$ and $j$ interchangeably. The atom set follows the prof: $R_m$ in §8.1 for regression, $R_j$ in §8.1.2 for classification, with $\hat p_{mk}$ or $\hat p_{jk}$ for the per-region class proportion. Both conventions occur in lecture and slides without ceremony.
- **Cost-complexity penalty.** ISLP equation 8.4 has $\alpha |T|$; the slide deck writes $C_\alpha(T) = Q(T) + \alpha|T|$ with $Q(T)$ as a generic cost. **$Q(T)$ is the regression RSS in §8.1.1 and the misclassification count in §8.1.2 (prune step), not Gini or entropy** — the prof flagged this asymmetry verbally as the "$Q$ depends on whether you're growing or pruning a classification tree" trap.
- **$|T|$ notation.** Both ISLP and the prof write the number of terminal nodes as $|T|$, but the prof flagged in [[L17-trees-1|L17]] that this is **cardinality of the leaf set**, not absolute value or an $L^1$-norm: *"that's the zeroth norm."* The "zeroth norm" reading is the prof's framing; ISLP just says "number of terminal nodes."
- **Boosting shrinkage.** Out of scope for m08 (module 9), but the prof uses $\eta$ for the learning rate in [[L19-boosting-1|L19]], whereas ISLP equation 8.10 uses $\lambda$. Same quantity.
- **`tree`'s `dev` field.** Slide deck and lecture refer to the R output column `dev` as "deviance" in regression (= RSS up to a multiplicative constant) and "number of misclassifications" in `cv.tree(..., FUN = prune.misclass)`. The single column has two meanings depending on what you fed in; ISLP doesn't discuss `tree` package R output.
- **Split notation $R_1(j, s)$, $R_2(j, s)$.** Both ISLP (eq. 8.2) and the prof define $R_1(j, s) = \{x : x_j < s\}$ and $R_2(j, s) = \{x : x_j \ge s\}$. Strict-vs-non-strict inequality at the boundary doesn't matter in practice (continuous $X_j$ has measure-zero ties), but the **convention is left-strict / right-inclusive** in both sources. The R `tree` package's display labels splits at the midpoint between adjacent integer values for integer predictors (per ISLP footnote 1), which is cosmetic only.
- **"Deviance" overloaded.** Slide deck uses "deviance" to mean the cross-entropy-scaled $-2\sum n_{jk} \log \hat p_{jk}$ when fitting classification trees, and to mean "RSS up to a constant" for regression trees. ISLP follows the same overloading in §8.3.1 (lab). The prof flagged this in [[L18-trees-2|L18]]: *"split='deviance' is cross-entropy, not Gaussian deviance"* — both are deviances in the GLM sense, scaled to match the respective likelihoods.
- **"Out of bag" vs "OOB".** Used interchangeably across the slide deck, lectures, and ISLP.
