---
title: "M10: Unsupervised Learning — Book delta"
module: 10-unsuper
isl-ch: 12
lectures: [L21, L22]
---

# Module 10: Unsupervised Learning — Book delta

This file reproduces the concrete artifacts (formulas, derivations, templates, definitions) that the prof taught for module 10 and that are **not** clean lookup-able statements in `wiki/book/12-unsupervised.md`. The book has Eq 12.17 (k-means objective), Eq 12.18 (pairwise-to-centroid identity), Eq 12.10 (PVE in score-sum form), Table 12.3 (linkage definitions), Algorithms 12.2 and 12.3, the $2^{n-1}$ ordering fact, and the qualitative Euclidean-vs-correlation discussion (Fig 12.15). Everything in this delta file is something the prof said, derived, or computed that is *additional* to that.

---

## 1. PCA

### 1.1 PCA via eigendecomposition of the covariance matrix

The book footnotes that "the [[principal-component-analysis|principal component]] directions $\phi_1, \phi_2, \phi_3, \ldots$ are given by the ordered sequence of eigenvectors of the matrix $\mathbf{X}^T \mathbf{X}$, and the variances of the components are the eigenvalues" (footnote 2 to §12.2.1) and otherwise says "the details are outside of the scope of this book." The prof teaches this as the **operational definition** he wants the student to use on the exam ("the fact, don't derive it"). [[L21-unsupervised-1|L21]]

Let $X \in \mathbb{R}^{n \times p}$ be the centred-and-standardised data matrix. Let
$$
\Sigma \;=\; \tfrac{1}{n-1}\, X^\top X
$$
denote the sample covariance matrix of the columns of $X$. Because $\Sigma$ is symmetric positive semi-definite, it admits the eigendecomposition
$$
\boxed{\;\Sigma \;=\; C\, \Lambda\, C^{-1} \;=\; C\, \Lambda\, C^\top\;}
$$
with $\Lambda = \operatorname{diag}(\lambda_1, \ldots, \lambda_p)$ a diagonal matrix of eigenvalues **in decreasing order** $\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_p \ge 0$, and $C$ an orthonormal matrix whose columns are the unit-norm eigenvectors of $\Sigma$. (Slide-deck "PCA — General setup", [[L21-unsupervised-1|L21]].)

The first principal component is then
$$
\boxed{\;\phi_1 \;=\; \text{eigenvector of } \Sigma \text{ corresponding to the largest eigenvalue } \lambda_1,\;}
$$
because for any unit-norm $\phi$ the variance of $Z = X\phi$ equals
$$
\operatorname{Var}(Z) \;=\; \phi^\top \Sigma\, \phi,
$$
and the maximum of $\phi^\top \Sigma \phi$ on the unit sphere is $\lambda_1$, achieved at the top eigenvector. The $m$-th PC is the eigenvector of the $m$-th eigenvalue:
$$
\operatorname{Var}(Z_m) \;=\; \phi_m^\top \Sigma\, \phi_m \;=\; \lambda_m.
$$

> "And this is the covariance matrix of your data. … The solution to the problem of finding the directions of maximal variance, the PCA problem, is actually solved by finding the eigenvalues and eigenvectors." — [[L21-unsupervised-1|L21]]

> "Your $\phi_1$ ends up just being the eigenvector corresponding to the largest eigenvalue." — [[L21-unsupervised-1|L21]]

The full spectral-derivation theory is OUT of scope ("we don't talk about spectral decomposition" — `docs/scope.md`). What is IN scope: the identity above as a one-line working tool.

### 1.2 PVE as a ratio of eigenvalues

Book Eq 12.10 writes the proportion of variance explained as
$$
\text{PVE}_m \;=\; \frac{\sum_{i=1}^n z_{im}^2}{\sum_{j=1}^p \sum_{i=1}^n x_{ij}^2}.
$$
The prof's working form, used in the Q3e exam template and the slide-deck "PCA — General setup" ([[L21-unsupervised-1|L21]]), is the **eigenvalue ratio**:
$$
\boxed{\;\text{PVE}_m \;=\; \frac{\lambda_m}{\sum_{k=1}^p \lambda_k},\qquad R^2(M) \;=\; \frac{\sum_{i=1}^M \lambda_i}{\sum_{j=1}^p \lambda_j}\;}
$$
where $R^2(M)$ is the cumulative PVE through $M$ PCs (slide notation: "fraction of original variance kept by the first $M$ principal components"). The two forms are equivalent because $\lambda_m = \operatorname{Var}(Z_m) = \tfrac{1}{n}\sum_i z_{im}^2$ (up to the $1/n$ factor that cancels in the ratio).

**Standardisation simplification.** After standardisation each column has variance 1, so the total variance is $\sum_k \lambda_k = p$. Then
$$
\text{PVE}_m \;=\; \frac{\lambda_m}{p}, \qquad R^2(M) \;=\; \frac{1}{p}\sum_{m=1}^M \lambda_m.
$$
This is the form Anders should expect on a Q3e-style hand calculation: eigenvalues are handed to him, he sums and divides by $p$. [[explained-variance-and-scree-plot]]

### 1.3 Q3e exam template — PCA hand calculation

Templated by the prof for the 2026 exam in [[L27-summary|L27]]. Setup: five standardised variables; the question hands you the eigenvalues $\lambda_1, \ldots, \lambda_5$, the loading matrix $[\phi_1, \ldots, \phi_5]$, and one observation $x = (x_1, \ldots, x_5)$. The three sub-questions:

**(a) Total variance explained by the first $M$ PCs.**
$$
R^2(M) \;=\; \frac{\sum_{m=1}^M \lambda_m}{\sum_{k=1}^p \lambda_k}.
$$
After standardisation $\sum_k \lambda_k = p$. Plug and divide.

**(b) Number of PCs needed for 90% (or 95% or 99%) variance.**
Compute the cumulative sum $R^2(M)$ step-by-step until it crosses the threshold:
$$
\text{find smallest } M \text{ such that } \sum_{m=1}^M \lambda_m / \sum_k \lambda_k \;\ge\; 0.90.
$$
The prof's worked numerical example in [[L27-summary|L27]]: eigenvalue ratios $0.54, 0.30, 0.10, \ldots$. Cumulative: $0.54 \to 0.84$ (not enough) $\to 0.94$ (past 0.90). Answer: **3 PCs**. [[explained-variance-and-scree-plot]]

**(c) Compute the score $z_{im}$ for one observation on one PC.**
$$
\boxed{\;z_{im} \;=\; \sum_{j=1}^p \phi_{jm}\, x_{ij} \;=\; \phi_{1m} x_{i1} + \phi_{2m} x_{i2} + \cdots + \phi_{pm} x_{ip}\;}
$$
i.e. plug the observation into the $m$-th loading vector. The book defines this in Eq 12.2 in passing; the **prof flags this exact plug-and-compute as a graded exam item**, with the instruction:

> "Show your work so that you can get partial credit when the inevitable little mistake happens." — [[L27-summary|L27]]

[[principal-component-analysis]] §"How it might appear on the exam"

### 1.4 The objective in terms of $\phi$ (variance maximisation as a quadratic form)

The book's optimisation (12.3) is written in score-sum form:
$$
\max_{\phi_1}\;\frac{1}{n}\sum_{i=1}^n \Big(\sum_{j=1}^p \phi_{j1} x_{ij}\Big)^2 \quad \text{s.t.} \quad \sum_j \phi_{j1}^2 = 1.
$$
The prof's equivalent statement (slide "PCA — General setup", [[L21-unsupervised-1|L21]]) is the **quadratic-form** version that makes the connection to $\Sigma$ explicit:
$$
\boxed{\;\max_{\phi_1: \|\phi_1\|_2 = 1}\; \phi_1^\top\, \Sigma\, \phi_1\;}
$$
Subsequent PCs add the orthogonality constraint:
$$
\phi_m \;=\; \mathop{\mathrm{argmax}}_{\substack{\|\phi\| = 1 \\ \phi \perp \phi_1, \ldots, \phi_{m-1}}} \phi^\top \Sigma\, \phi, \qquad \operatorname{Var}(Z_m) \;=\; \phi_m^\top \Sigma \phi_m \;=\; \lambda_m.
$$
This is the form that immediately yields the eigendecomposition recipe of §1.1. [[L21-unsupervised-1|L21]] / [[principal-component-analysis]]

### 1.5 Standardisation z-score formula (load-bearing for module 10)

The book §12.2.4 and §12.4.2 prescribes "scaling the variables to have standard deviation one" but never writes the formula. The prof writes it explicitly (slide-deck [[L21-unsupervised-1|L21]], also restated for clustering in [[L22-unsupervised-2|L22]]):
$$
\boxed{\;z_{ij} \;=\; \frac{x_{ij} - \bar{x}_j}{s_j}, \qquad \bar{x}_j = \frac{1}{n}\sum_{i=1}^n x_{ij}, \qquad s_j = \sqrt{\tfrac{1}{n-1}\sum_{i=1}^n (x_{ij} - \bar{x}_j)^2}.\;}
$$
After this transformation each column has mean 0 and sample variance 1; therefore $\sum_k \lambda_k = p$. This is the input to the PCA / [[k-means-clustering|k-means]] / [[hierarchical-clustering|hierarchical]] pipelines as the prof teaches them. [[standardization]] / [[L21-unsupervised-1|L21]]

> "If you don't standardize them so that their mean is zero and their variance is one, then if one had a standard deviation of like a million, then that will be your strongest variable." — [[L21-unsupervised-1|L21]]

---

## 2. K-means clustering

The book gives Eq 12.17 (objective), Eq 12.18 (the pairwise-to-centroid identity), and Algorithm 12.2. The book also states monotonicity in two sentences. The items below are the prof's additions.

### 2.1 Structured monotonicity proof (Exercise 10.2 template)

The book says: "in Step 2(a) the cluster means for each feature are the constants that minimise the sum-of-squared deviations, and in Step 2(b), reallocating the observations can only improve (12.18)." The prof's structured proof, the one graded for Exercise 10.2, decomposes this into three explicit steps. [[L22-unsupervised-2|L22]] / [[k-means-clustering]]

**Setup.** The k-means objective is
$$
\mathcal{J}(C_1, \ldots, C_K) \;=\; \sum_{k=1}^K \frac{1}{|C_k|} \sum_{i, i' \in C_k} \sum_{j=1}^p (x_{ij} - x_{i'j})^2.
$$

**Step 1 — Rewrite via identity 12.18.** Apply
$$
\frac{1}{|C_k|} \sum_{i, i' \in C_k} \sum_{j=1}^p (x_{ij} - x_{i'j})^2 \;=\; 2 \sum_{i \in C_k} \sum_{j=1}^p (x_{ij} - \bar{x}_{kj})^2
$$
to each cluster. The objective becomes (up to a factor of 2)
$$
\mathcal{J} \;=\; 2 \sum_{k=1}^K \sum_{i \in C_k} \|x_i - \bar{x}_k\|^2, \qquad \bar{x}_k = \tfrac{1}{|C_k|}\sum_{i \in C_k} x_i.
$$
This is a *sum of squared deviations from cluster means*.

**Step 2 — Centroid step decreases $\mathcal{J}$.** Fix the partition $\{C_k\}$ and vary the per-cluster reference points $c_k$:
$$
\bar{x}_k \;=\; \mathop{\mathrm{argmin}}_{c \in \mathbb{R}^p} \;\sum_{i \in C_k} \|x_i - c\|^2.
$$
(One-liner derivation: differentiate $\sum_i \|x_i - c\|^2$ in $c$, set to zero, get $c = \tfrac{1}{|C_k|}\sum_i x_i$.) So step 2(a), which sets each $c_k$ to the cluster mean, **weakly decreases** every cluster's contribution and therefore $\mathcal{J}$.

**Step 3 — Assignment step decreases $\mathcal{J}$.** Fix the centroids $\{\bar{x}_k\}$ and reassign each observation $i$ to its nearest centroid:
$$
k(i) \;=\; \mathop{\mathrm{argmin}}_{k} \|x_i - \bar{x}_k\|^2.
$$
Each observation's contribution to $\mathcal{J}$ is now individually minimised over $k$, so $\mathcal{J}$ **weakly decreases**.

**Step 4 — Convergence.** $\mathcal{J} \ge 0$, the sequence of values $\mathcal{J}^{(0)} \ge \mathcal{J}^{(1)} \ge \cdots$ is nonincreasing and bounded below, and there are only finitely many partitions of $n$ objects into $K$ groups. Therefore the algorithm must stabilise in finitely many iterations — and the limiting partition is a **local** (not necessarily global) optimum.

> "It's not clear that this will lead to a fixed point just from looking at it, but it does." — [[L22-unsupervised-2|L22]]

[[k-means-clustering]] §"Why monotone (Exercise 10.2)"

### 2.2 Ensemble fix for local-optima (the prof's "funny way")

A workaround the prof named explicitly in [[L22-unsupervised-2|L22]], not in the book. Run the k-means algorithm $B$ times from independent random initialisations, producing assignments $\sigma^{(1)}, \ldots, \sigma^{(B)} : \{1, \ldots, n\} \to \{1, \ldots, K\}$. Define a new dissimilarity between observations $i$ and $i'$ by the *co-clustering frequency*:
$$
\tilde{d}(i, i') \;=\; 1 \;-\; \frac{1}{B}\sum_{b=1}^B \mathbb{1}\{\sigma^{(b)}(i) = \sigma^{(b)}(i')\}.
$$
Then re-cluster the data using $\tilde{d}$ as the input dissimilarity (e.g. through hierarchical clustering, or another round of k-means with a kernel). The intuition is that pairs that *consistently* land together across initialisations are "really" similar; pairs that land together only occasionally are at the boundary.

> "Actually that works quite well but it's kind of a funny way of doing things." — [[L22-unsupervised-2|L22]]

This is an ensembling idea analogous to [[bagging|bagging]]-the-clusterer, and the prof connects it to the [[boosting|boosting]] / [[random-forest|random-forest]] theme from modules 8–9. [[k-means-clustering]]

### 2.3 Brute-force partition count

The book mentions "almost $K^n$ ways to partition $n$ observations into $K$ clusters." The prof's concrete instantiation, useful for an exam justification of *why* we need the iterative algorithm:
$$
\text{partitions of } n \text{ into } K \text{ groups} \;\approx\; K^n.
$$
At $n = 1000, K = 10$ this is $10^{1000}$. ([[L22-unsupervised-2|L22]]) Use this as a one-line "why we don't brute-force" answer.

### 2.4 Trivial degenerate case $K = n$

Not stated in the book. If $K = n$ then every observation is its own cluster, $\bar{x}_k = x_k$, and the objective evaluates to
$$
\mathcal{J}(C_1, \ldots, C_n) \;=\; 0.
$$
This is the global minimum, but it is trivial: no structure has been learnt. [[L22-unsupervised-2|L22]] / [[k-means-clustering]] §"Insights & mental models"

---

## 3. Hierarchical clustering

The book gives Algorithm 12.3, Table 12.3 (linkage formulas), the $2^{n-1}$ reordering fact, and the qualitative gender-vs-nationality non-nested example. The book does **not** walk through a numerical worked example with a complete-linkage hand-computation; the prof does, and grades it with a $-1$-per-mistake rubric. [[L22-unsupervised-2|L22]] / [[L27-summary|L27]] / [[hierarchical-clustering]]

### 3.1 Linkage formulas (notation cleaned)

The book's Table 12.3 gives definitions in prose. Formally, with $A$ and $B$ two clusters and $d(\cdot, \cdot)$ the point-to-point dissimilarity:

| Linkage  | $D(A, B) =$                                                     |
|----------|-----------------------------------------------------------------|
| Complete | $\displaystyle\max_{i \in A,\, j \in B} d(x_i, x_j)$            |
| Single   | $\displaystyle\min_{i \in A,\, j \in B} d(x_i, x_j)$            |
| Average  | $\displaystyle\frac{1}{|A|\,|B|} \sum_{i \in A}\sum_{j \in B} d(x_i, x_j)$ |
| Centroid | $d(\bar{x}_A,\, \bar{x}_B)$                                     |

The prof also name-checks median linkage (in scope only as "exists"). Ward linkage is **out of scope** per `docs/scope.md`. The average-linkage definition is the **unweighted** mean (each pair contributes equally); not the weighted variant some packages use.

> "Average and complete tend to yield more balanced clusters." — [[L22-unsupervised-2|L22]]

### 3.2 Hand-computation recipe (the $-1$-per-mistake exam question)

Templated by the prof in [[L27-summary|L27]] as a graded exam item. From a given $n \times n$ dissimilarity matrix $D$ and a chosen linkage:

> [!important] Recipe
> 1. **Find the smallest off-diagonal entry** $D_{ij}$. Fuse $\{i\}$ and $\{j\}$ at height $h = D_{ij}$.
> 2. **Update the matrix.** Replace rows/columns $i$ and $j$ with a single row/column for the new cluster $\{i, j\}$. The new entries against any remaining $k$ are computed by the linkage rule:
>     - Complete: $D(\{i, j\}, k) = \max(D_{ik}, D_{jk})$.
>     - Single: $D(\{i, j\}, k) = \min(D_{ik}, D_{jk})$.
>     - Average: $D(\{i, j\}, k) = \tfrac{1}{2}(D_{ik} + D_{jk})$ when fusing singletons; in general, average over all cross-cluster pairs.
> 3. **Repeat** on the smaller matrix: find next-smallest entry, fuse, recompute.
> 4. **Stop** when one cluster remains.
> 5. **Draw the dendrogram** with x-axis = arbitrary leaf order and y-axis = fusion heights. Each fusion is a horizontal bar at the recorded height linking its children.
> 6. **Cut** at the requested height (or to get $K$ clusters) and **report** the cluster membership.

> [!warning] Grading rubric
> "Negative one point for each mistake … I won't give negative points, don't worry." — [[L27-summary|L27]]
>
> Mistakes propagate: a wrong first fusion ruins the whole tree. Always recompute the matrix in writing after every step.

### 3.3 Worked example — complete linkage on the prof's 4×4 matrix

This is the prof's slide-deck worked example for the [[L22-unsupervised-2|L22]] hand-computation drill (also Exercise 2 of ISLP §12.6, whose solution the book does **not** provide). [[L22-unsupervised-2|L22]] slides "Exercise 2 from the book" / [[hierarchical-clustering]] §"Hand-computation procedure"

**Input matrix:**
$$
D \;=\; \begin{pmatrix} 0 & 0.3 & 0.4 & 0.7 \\ 0.3 & 0 & 0.5 & 0.8 \\ 0.4 & 0.5 & 0 & 0.45 \\ 0.7 & 0.8 & 0.45 & 0 \end{pmatrix}.
$$

**Step 1.** Smallest off-diagonal: $D_{12} = 0.3$. **Fuse $\{1, 2\}$ at height $0.3$.**

**Step 2.** Update the matrix under **complete** linkage:
$$
D(\{1, 2\}, 3) \;=\; \max(D_{13}, D_{23}) \;=\; \max(0.4, 0.5) \;=\; 0.5,
$$
$$
D(\{1, 2\}, 4) \;=\; \max(D_{14}, D_{24}) \;=\; \max(0.7, 0.8) \;=\; 0.8,
$$
$$
D(3, 4) \;=\; 0.45.
$$
New $3 \times 3$ matrix (rows: $\{1,2\}, 3, 4$):
$$
\begin{pmatrix} 0 & 0.5 & 0.8 \\ 0.5 & 0 & 0.45 \\ 0.8 & 0.45 & 0 \end{pmatrix}.
$$

**Step 3.** Smallest entry: $D_{34} = 0.45$. **Fuse $\{3, 4\}$ at height $0.45$.**

**Step 4.** Update:
$$
D(\{1, 2\}, \{3, 4\}) \;=\; \max(D_{13}, D_{14}, D_{23}, D_{24}) \;=\; \max(0.4, 0.7, 0.5, 0.8) \;=\; 0.8.
$$

**Step 5.** **Final fusion at height $0.8$**, producing the single cluster $\{1, 2, 3, 4\}$.

**Dendrogram (heights only — the x-ordering is arbitrary):**
- Bottom: leaves $1, 2, 3, 4$.
- Bar at $h = 0.3$ joining 1 and 2.
- Bar at $h = 0.45$ joining 3 and 4.
- Bar at $h = 0.8$ joining $\{1, 2\}$ and $\{3, 4\}$.

**Cuts.** A cut at any $h \in (0.45,\; 0.8)$ yields **two clusters**: $\{1, 2\}$ and $\{3, 4\}$. A cut at $h \in (0.3,\; 0.45)$ yields **three clusters**: $\{1, 2\}$, $\{3\}$, $\{4\}$. A cut at $h < 0.3$ yields four singletons.

### 3.4 Worked example — single linkage on the same matrix

Same input matrix; swap $\max$ for $\min$. [[L22-unsupervised-2|L22]] slides Exercise 2 (b)

**Step 1.** Smallest entry $D_{12} = 0.3$. **Fuse $\{1, 2\}$ at $0.3$.**

**Step 2.** Update under single linkage:
$$
D(\{1, 2\}, 3) \;=\; \min(0.4, 0.5) \;=\; 0.4, \qquad D(\{1, 2\}, 4) \;=\; \min(0.7, 0.8) \;=\; 0.7,
$$
and $D(3, 4) = 0.45$.

**Step 3.** Smallest entry: $D(\{1, 2\}, 3) = 0.4$. **Fuse $\{1, 2, 3\}$ at $0.4$.**

**Step 4.** Update:
$$
D(\{1, 2, 3\}, 4) \;=\; \min(D_{14}, D_{24}, D_{34}) \;=\; \min(0.7, 0.8, 0.45) \;=\; 0.45.
$$

**Step 5.** **Final fusion at height $0.45$.**

**Cut at $K = 2$.** Below $h = 0.45$ the partition is $\{1, 2, 3\}$ and $\{4\}$ — **different from complete linkage**, which gave $\{1, 2\}$ and $\{3, 4\}$. This is the canonical "linkage choice changes the answer" demonstration the prof uses.

### 3.5 Correlation-based distance, explicit formula

The book (§12.4.2, Fig 12.15) describes correlation-based distance qualitatively as "two observations are similar if their features are highly correlated", but does not give the formula or its range. The prof writes it out explicitly in [[L22-unsupervised-2|L22]]:
$$
\boxed{\;d_{\text{corr}}(x_i, x_{i'}) \;=\; 1 - \rho(x_i, x_{i'}),\;}
$$
where $\rho(x_i, x_{i'})$ is the **Pearson correlation between the two observation profiles** (each profile is a length-$p$ vector across features). The range:
$$
\rho \in [-1, 1] \;\Longrightarrow\; d_{\text{corr}} \in [0, 2],
$$
with $d_{\text{corr}} = 0$ at perfect positive correlation and $d_{\text{corr}} = 2$ at perfect anti-correlation. The variant $1 - |\rho|$ is used when the sign of the correlation is not meaningful (range $[0, 1]$).

> "Often people will say all right let's minimize 1 minus the Pearson correlation. All right, that way, if they're perfectly correlated, you have a dissimilarity of zero." — [[L22-unsupervised-2|L22]]

> "Note: Correlation is actually a similarity measure, not a distance measure." — slide-deck [[L21-unsupervised-1|L21]]/[[L22-unsupervised-2|L22]]

This is a similarity-to-distance conversion, *not* a proper metric: $d_{\text{corr}}$ does not satisfy the triangle inequality. For clustering this does not matter — only the ordering of pairs is needed. [[distance-metrics]]

### 3.6 Squared Euclidean vs Euclidean inside k-means (ranking equivalence)

Not explicit in the book — the book uses squared Euclidean throughout §12.4.1 without commenting on why the square root is dropped. The prof's justification, useful as a one-line exam answer:

> "There's no reason to take the square root, it's just an extra step that won't get you anything … it won't change the goal or won't change who wins or how they win." — [[L22-unsupervised-2|L22]]

Formally: $\sqrt{\cdot}$ is strictly monotone on $[0, \infty)$, so for any partition $\{C_k\}$ and any pair of partitions $\{C_k\}, \{C_k'\}$,
$$
\sum_{k}\sum_{i \in C_k} \|x_i - \bar{x}_k\|^2 \;<\; \sum_{k}\sum_{i \in C_k'} \|x_i - \bar{x}_k'\|^2
$$
iff the same inequality holds with $\|\cdot\|^2$ replaced by $\|\cdot\|$ as a function of which partition wins on a per-cluster basis. The squared form is preferred because the centroid-as-minimizer property (§2.1, step 2) holds in closed form for squared distance and *not* for plain Euclidean distance — for plain Euclidean, the cluster-wise minimizer is the geometric median (a much harder object). [[distance-metrics]] / [[k-means-clustering]]

---

## 4. Notation / terminology drift

A few small differences between the prof's notation and the book's that an exam reader should keep straight.

- **Covariance normalisation.** The prof writes $\Sigma = \tfrac{1}{n-1} X^\top X$ (slide [[L21-unsupervised-1|L21]]). The book uses $\tfrac{1}{n} X^\top X$ in the variance form of Eq 12.10 (consistent with population variance). The eigenvectors are identical; only the eigenvalues scale by $n/(n-1)$, and ratios (PVE) are unaffected.
- **Eigendecomposition shorthand.** The slide writes $\Sigma = C \Lambda C^{-1}$. Because $\Sigma$ is symmetric, $C$ is orthonormal and $C^{-1} = C^\top$. Some packages return $C$, others $C^\top$; the eigenvectors are the same up to sign.
- **"Dissimilarity" vs "distance".** The prof uses these interchangeably; the book technically distinguishes (a "dissimilarity" need not satisfy the triangle inequality). For module 10 they are the same object.
- **Average linkage.** Both the prof and the book mean the unweighted mean over pairs (each cross-cluster pair contributes once with weight $1/(|A||B|)$). Some R packages and the exam_analysis cheat-sheet phrase it as a "weighted mean" — that is loose language for the same formula. UPGMA = unweighted average linkage; do not confuse with WPGMA (weighted, not in scope).
- **PCA loading sign.** The prof and the book both note that loadings are unique up to a sign flip. If two software packages disagree on the sign of a loading column, that is not an error.
- **Sample variance denominator inside standardisation.** The prof writes $s_j = \sqrt{\tfrac{1}{n-1}\sum_i (x_{ij} - \bar{x}_j)^2}$ ($n-1$). Many packages use $n$. After dividing by $s_j$, the resulting columns have *sample* variance 1 vs *population* variance 1 — the difference is cosmetic for everything downstream.
- **$z_m$ vs $Z_m$.** The book uses uppercase $Z_m$ for the score variable, lowercase $z_{im}$ for the score of observation $i$. The prof's slide-deck uses both interchangeably; the wiki keeps the book convention.
- **The letter $K$.** In hierarchical clustering, $K$ is the number of clusters obtained by cutting the dendrogram at some height. In k-means, $K$ is the pre-specified target. Same symbol, different roles.
