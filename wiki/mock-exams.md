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
| 3 | <a href="/pdfs/mock-exam-3.pdf" target="_blank" rel="noopener">Mock Exam 3 (PDF)</a> | *not yet available* | 10 / 28 / 16 / 22 / 24 |

LaTeX sources live at `mock-exams/mock-exam-N.tex` and `mock-exams/mock-exam-N-solution.tex` in the repo. Each mock deliberately rotates topics relative to the others so they cover complementary slices of the prof's scope (e.g.\ mock 1 has the MLE\,=\,LS derivation and complete-linkage clustering; mock 2 has the LDA/QDA boundary derivation and single-linkage clustering; mocks pair lasso vs.\ ridge\,+\,PCR; mocks rotate which interaction trap appears on the log-odds scale).

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
