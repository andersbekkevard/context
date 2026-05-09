# Lectures manifest

Deterministic mapping from transcript files in `transcripts/` to the lecture
identity each one represents (number, date, module, title, slug). Source of
truth for the parallel `wiki/lectures/L<NN>-<slug>.md` compression pass — every
agent looks up its row here so numbers, slugs, and module tags don't drift.

Titles reflect what the prof **actually taught** (read from the transcripts),
not what the published `course-information.md` schedule claimed. The two
diverge: the prof ran a session behind for most of the semester, the Module 6
sessions were pushed from Feb 16/17 to Feb 23/24, the Compulsory-Exercise-1
weeks (Feb 9–10) were used as regular Resampling lectures, and the
Compulsory-Exercise-2 / Easter weeks have no transcripts.

| #  | File                                  | Date       | Module          | Title                                           | Slug              |
|----|---------------------------------------|------------|-----------------|-------------------------------------------------|-------------------|
| 01 | tma4268-2026-01-05.txt                | 2026-01-05 | 01-intro        | Introduction                                    | intro             |
| 02 | tma4268-2026-01-12.txt                | 2026-01-12 | 02-statlearn    | Statistical Learning 1                          | statlearn-1       |
| 03 | tma4268-2026-01-13.txt                | 2026-01-13 | 02-statlearn    | Statistical Learning 2                          | statlearn-2       |
| 04 | tma4268-2026-01-19.txt                | 2026-01-19 | 02-statlearn    | Statistical Learning 3                          | statlearn-3       |
| 05 | tma4268-2026-01-20.txt                | 2026-01-20 | 03-linreg       | Linear Regression 1                             | linreg-1          |
| 06 | tma4268-2026-01-26.txt                | 2026-01-26 | 03-linreg       | Linear Regression 2                             | linreg-2          |
| 07 | tma4268-2026-01-27.txt                | 2026-01-27 | 04-classif      | Classification 1                                | classif-1         |
| 08 | tma4268-2026-02-02.txt                | 2026-02-02 | 04-classif      | Classification 2 (LinReg wrap-up + LDA)         | classif-2         |
| 09 | tma4268-2026-02-03.txt                | 2026-02-03 | 04-classif      | Classification 3                                | classif-3         |
| 10 | tma4268-2026-02-09.txt                | 2026-02-09 | 05-resample     | Resampling 1                                    | resample-1        |
| 11 | tma4268-2026-02-10.txt                | 2026-02-10 | 05-resample     | Resampling 2                                    | resample-2        |
| 12 | tma4268-2026-02-23.txt                | 2026-02-23 | 06-modelsel     | Model Selection and Regularization 1            | modelsel-1        |
| 13 | tma4268-2026-02-24.txt                | 2026-02-24 | 06-modelsel     | Model Selection and Regularization 2 (Ridge)    | modelsel-2        |
| 14 | tma4268-2026-03-02.txt                | 2026-03-02 | 06-modelsel     | Model Selection and Regularization 3 (PCR/PCA)  | modelsel-3        |
| 15 | tma4268-2026-03-03.txt                | 2026-03-03 | 06-modelsel     | Model Selection and Regularization 4 (PCR wrap) | modelsel-4        |
| 16 | tma4268-2026-03-09.txt                | 2026-03-09 | 07-beyondlinear | Moving Beyond Linearity 1                       | beyondlinear-1    |
| 17 | tma4268-2026-03-10.txt                | 2026-03-10 | 08-trees        | Tree-based Methods 1 (BeyondLinear wrap-up)     | trees-1           |
| 18 | tma4268-2026-03-16.txt                | 2026-03-16 | 08-trees        | Tree-based Methods 2                            | trees-2           |
| 19 | tma4268-2026-03-17.txt                | 2026-03-17 | 09-boosting     | Boosting and Additive Trees 1 (Trees wrap-up)   | boosting-1        |
| 20 | tma4268-2026-04-07.txt                | 2026-04-07 | 09-boosting     | Boosting and Additive Trees 2                   | boosting-2        |
| 21 | tma4268-2026-04-13.txt                | 2026-04-13 | 10-unsuper      | Unsupervised Learning 1 (PCA)                   | unsupervised-1    |
| 22 | tma4268-2026-04-14.txt                | 2026-04-14 | 10-unsuper      | Unsupervised Learning 2 (Clustering)            | unsupervised-2    |
| 23 | tma4268-2026-04-20.txt                | 2026-04-20 | 11-nnet         | Neural Networks 1 (Feedforward)                 | nnet-1            |
| 24 | tma4268-2026-04-21.txt                | 2026-04-21 | 11-nnet         | Neural Networks 2 (CNNs)                        | nnet-2            |
| 25 | tma4268-2026-04-27-1013.txt           | 2026-04-27 | 11-nnet         | Neural Networks 3 (aborted recording)           | nnet-3-aborted    |
| 26 | tma4268-2026-04-27-1015.txt           | 2026-04-27 | 11-nnet         | Neural Networks 3 (RNNs and Double Descent)     | nnet-3            |
| 27 | tma4268-2026-04-28.txt                | 2026-04-28 | 12-final        | Summary and Exam Review                         | summary           |

## Notes

- **The official schedule is one slot off from what was actually delivered.**
  The prof self-reports being "a little behind" around Mar 9. Decisions are
  made from transcript content (opening recap + "today we'll talk about…"),
  not from `course-information.md`. Examples worth knowing:
  - Jan 19 is *Statistical Learning part 3*, not Linear Regression 1 — prof
    explicitly says "still module two, but now part two."
  - Module 6 (Model Selection) ran Feb 23, Feb 24, Mar 2, Mar 3 (four
    sessions), pushing Module 7 to Mar 9 etc. Schedule had Module 6 on
    Feb 16/17.
  - Feb 9 / Feb 10 were used for Resampling (Module 5), not for Compulsory
    Exercise 1 work.
  - There are *no* transcripts for Jan 6 R-course, Mar 23/24 Compulsory
    Exercise 2 weeks, or the Easter break — none get a lecture number.

- **Two-module sessions are tagged by what the lecture mainly delivers.** A
  session that wraps the previous module in the first 10 minutes and then
  spends the rest on the new module is filed under the new module:
  - L08 Feb 2 wraps LinReg then teaches LDA → `04-classif / classif-2`.
  - L17 Mar 10 wraps BeyondLinear then teaches trees → `08-trees / trees-1`.
  - L19 Mar 17 wraps Trees then starts Boosting → `09-boosting / boosting-1`.
  - L22 Apr 14 wraps Unsupervised (clustering) then peeks NN → kept as
    `10-unsuper / unsupervised-2` since the bulk is clustering.

- **Apr 27 has two transcript files for the same date.** The 1013/1015 suffixes
  are start-time stamps (~10:13 / ~10:15). The `-1013` file is a 10-character
  aborted recording (just "Thank you."); the `-1015` file is the real 85-min
  lecture on RNNs and double descent. They are assigned adjacent numbers L25
  and L26 with the aborted file first by timestamp; L25 is flagged as empty
  in its slug (`nnet-3-aborted`) so the compression-pass agent knows to skip
  or stub it rather than try to summarize 10 characters.

- **L24 (Apr 21) slug `nnet-2` covers convolutional networks**, not RNNs. The
  prof defers RNNs to "next time" at the end — so L26 Apr 27 picks up RNNs
  and double descent.

- **L27 Apr 28 is the real "Summary and Outlook"** — exam logistics, scope,
  Q&A. The schedule's "Apr 14 = Summary" entry is stale; Apr 14 is Unsupervised
  Learning 2.

- **Module slugs match `modules/` folder names**, with the digit zero-padded
  and the trailing word lowercased. Note `modules/9SVM` exists but is **not**
  used here — no transcript covers SVMs as a module focus, and `9TreeBoosting`
  is the boosting module that maps to `09-boosting`.

- **No tier/importance ranking is encoded** here — that's derived elsewhere
  per the repo invariants. This file is purely identity.
