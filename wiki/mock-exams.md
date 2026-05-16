---
title: Mock Exams
---

Calibrated mock exams for the **May 18, 2026 TMA4268 final**. Each mock is constructed from the prof's [exam-review lecture](lectures/L27-summary.md), the [exam analysis](exam_analysis.md), the [2023--2025 finals](exams.md), and the in-scope concept atoms. Per-problem weights add to $100\,\% = 100$ points and map directly onto NTNU's *prosentvurderingsmetoden* grade boundaries (A: 89--100, B: 77--88, C: 65--76, D: 53--64, E: 41--52, F: 0--40).

> [!note] Workflow
> Tackle the mock cold under exam conditions (4 hours, open book: ISLP + your A5 sheet + a calculator). Then compare against the solution PDF. The solution sketches use the same partial-credit grading conventions Stefanie and Sara used on the official 2024 / 2025 keys.

## Available mocks

| # | Exam | Solution proposal | Weighting (per problem, %) |
|---|---|---|---|
| 1 | <a href="/pdfs/mock-exam-1.pdf" target="_blank" rel="noopener">Mock Exam 1 (PDF)</a> | <a href="/pdfs/mock-exam-1-solution.pdf" target="_blank" rel="noopener">Solution (PDF)</a> | 10 / 28 / 16 / 22 / 24 |
| 2 | <a href="/pdfs/mock-exam-2.pdf" target="_blank" rel="noopener">Mock Exam 2 (PDF)</a> | <a href="/pdfs/mock-exam-2-solution.pdf" target="_blank" rel="noopener">Solution (PDF)</a> | 10 / 28 / 16 / 22 / 24 |
| 3 | <a href="/pdfs/mock-exam-3.pdf" target="_blank" rel="noopener">Mock Exam 3 (PDF)</a> | <a href="/pdfs/mock-exam-3-solution.pdf" target="_blank" rel="noopener">Solution (PDF)</a> | 10 / 28 / 16 / 22 / 24 |
| 4 | <a href="/pdfs/mock-exam-4.pdf" target="_blank" rel="noopener">Mock Exam 4 (PDF)</a> | <a href="/pdfs/mock-exam-4-solution.pdf" target="_blank" rel="noopener">Solution (PDF)</a> | 10 / 28 / 16 / 22 / 24 |
| 5 | <a href="/pdfs/mock-exam-5.pdf" target="_blank" rel="noopener">Mock Exam 5 (PDF)</a> | <a href="/pdfs/mock-exam-5-solution.pdf" target="_blank" rel="noopener">Solution (PDF)</a> | 10 / 28 / 16 / 22 / 24 |
| 6 | <a href="/pdfs/mock-exam-6.pdf" target="_blank" rel="noopener">Mock Exam 6 (PDF)</a> | <a href="/pdfs/mock-exam-6-solution.pdf" target="_blank" rel="noopener">Solution (PDF)</a> | 10 / 28 / 16 / 18 / 28 |
| 7 | <a href="/pdfs/mock-exam-7.pdf" target="_blank" rel="noopener">Mock Exam 7 (PDF)</a> | <a href="/pdfs/mock-exam-7-solution.pdf" target="_blank" rel="noopener">Solution (PDF)</a> | 10 / 28 / 16 / 20 / 26 |

LaTeX sources live at `mock-exams/mock-exam-N.tex` and `mock-exams/mock-exam-N-solution.tex` in the repo. Each mock deliberately rotates topics relative to the others so they cover complementary slices of the prof's scope:

- **Mock 1** --- MLE\,=\,LS mathy derivation; complete-linkage hierarchical clustering on a 4$\times$4 matrix; lasso (with $\lambda$ via 10-fold CV) as the regularizer; the logistic-regression interaction trap (\texttt{balance}\,$\times$\,\texttt{sex}) sits in the classification problem.
- **Mock 2** --- variance of an average of correlated predictors as the mathy derivation (the random-forest decorrelation argument); single-linkage hierarchical clustering on a 5$\times$5 matrix; ridge \emph{and} PCR side by side; subset-selection MC; the logistic interaction trap again.
- **Mock 3** --- LDA decision-boundary derivation as the mathy slot (the prof flagged this twice in lecture as ``a typical exam question''); single-linkage hierarchical clustering on a different 4$\times$4 matrix; ridge in regression alongside a GAM with smoothing-spline + cubic-spline terms and gradient-boosted trees; LDA vs.\ QDA confusion-matrix comparison in the classification problem; the interaction trap (\texttt{bmi}\,$\times$\,\texttt{smoker}) is on the OLS side, not the logistic side; bootstrap, K-means, and OOB-error MC slots that mock 1 didn't have.
- **Mock 4 (calibration pass)** --- the **bias--variance decomposition derivation** as the mathy slot (the prof's *guaranteed* exam topic, re-flagged three times); two pseudocode problems (**nested $k$-fold CV** with the wrong-way trap, **bootstrap SE** for the sample median, **gradient-boosting** inner loop); harder linear regression with explicit **(near-)collinearity** between \texttt{area}, \texttt{rooms}, and \texttt{area\_per\_room} plus an \texttt{age}-quadratic minimum and a standardised \texttt{distance}$\times$\texttt{has\_balcony} interaction; classification problem is the **heaviest yet**, with logistic + a regularized neural net (dropout, early stopping, label smoothing) + AdaBoost; XGBoost / mini-batch SGD / backprop / boosting-flavor T/F all show up in P2; hierarchical clustering and $K$-means kept to a 2-pt T/F so they don't disappear entirely.
- **Mock 5 (calibration pass)** --- bias--variance decomposition as the mathy slot (the prof's *explicitly promised* topic from L27); calibration pass weighting up classification, CV pseudocode + nested CV, bootstrap SE, collinearity, backprop + mini-batch SGD, label smoothing / early stopping / dropout, and the AdaBoost / gradient-boosting / XGBoost family; weighting down dendrograms (kept as a small 2-pt complete-linkage 4$\times$4) and k-means (dropped); hardened linear-regression problem (polynomial + interaction + 3-level categorical + standardized predictors), small backprop chain-rule derivation in the NN classifier; AdaBoost pseudocode plus a single-iteration numerical drill.
- **Mock 6 (calibration pass)** --- bias--variance decomposition as the mathy slot (the prof's most-promised exam topic), with a shrinkage-estimator application; **nested-CV pseudocode** and a small **backpropagation** chain-rule derivation flesh out P3. Concrete-strength regression with deliberate \texttt{water}/\texttt{superplast} collinearity (and a continuous\,$\times$\,categorical interaction) makes the linear-regression problem the hardest of the series; telecom-churn classification ramps up classification share, with **AdaBoost pseudocode + hand-calc of $\alpha_m$ and weight updates** as the boosting question. Heavier weight on cross-validation (incl. the wrong-way / nested-CV pitfall), bootstrap SE, collinearity, mini-batch SGD, dropout / early stopping / label smoothing, and the boosting variants (AdaBoost, gradient boosting, XGBoost); dendrograms, ROC / AUC / sensitivity / specificity, and $K$-means deliberately down-weighted relative to mocks 1--3.
- **Mock 7 (calibration pass)** --- bias\,--\,variance \emph{decomposition derivation} as the mathy slot (the prof's explicit promise in L27); CV and \emph{nested} CV pseudocode + an AdaBoost-by-hand round in Problem 3; collinearity in linear regression diagnosed via two compatible fingerprints (correlation $0.99$, inflated SEs) on a wine-quality dataset; classification weighted up to $26\,\%$ via a churn problem comparing logistic regression, LDA, and gradient boosting; dendrograms / AUC / $K$-means correspondingly downweighted; backprop, mini-batch SGD, dropout, label smoothing, and early stopping promoted into Problem 2.

## How each mock is constructed

The construction recipe is the same every time:

1. **Format mix from the prof's stated signals.** Mostly multiple choice + true/false + short interpretation, plus exactly one mathy / derivation problem, plus two data-analysis problems where the regression / GLM / CV / ROC output is *given* (no code expected).
2. **Topic balance from the slide-and-exercise scope.** Roughly equal weight across the twelve modules, biased toward what the prof flagged: bias--variance (guaranteed), interactions, regularization, classification metrics, PCA, hand-by-hand hierarchical clustering, NN parameter counting, boosting hyperparameters.
3. **Past-exam reformulation rules.** Wherever a 2023--2025 question used "fit this in R," the mock replaces it with "here is the output, interpret." Old paper coding tasks become interpretation tasks; old paper conceptual / theoretical / hand-calculation tasks stay intact.
4. **Difficulty calibration.** Aimed to spread strong, average, and weak students across the percentile bands. There is no "trick" problem; partial credit is generous; the mathy slot follows the prof's bar of "*mathy but not, you know, no weird spaces or fancy proofs.*"
5. **Internal audit pass.** Each mock goes through a critic agent that scores every sub-part for likelihood-given-prof-signals and a solver agent that produces the official-style solution proposal. Disagreements get reconciled before the PDF is published.

## Related

- [[exam_analysis|Exam analysis]] --- canonical synthesis of the prof's signals across all 27 lectures.
- [[lectures/L27-summary|L27 (Apr 28)]] --- the exam-review lecture, the highest-priority single source for "what's actually on the test."
- [[exams|Past exams (2023--2025)]] --- with official solutions inline.
- [[docs/scope|Scope rule]] --- the authoritative "what's in / what's out" reference, derived from slides + lectures + exercises.
