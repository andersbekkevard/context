# Deck-generation pass: agent brief

You're producing **one MCQ deck** for one TMA4268 module, an interactive HTML page that drills the module's atoms via 20–30 multiple-choice questions. Read [[../templates/deck]] in full first.

## North Star

When Anders opens the deck for module $N$, he should be able to click through ~25 questions in 30–45 minutes, get rapid right/wrong feedback per click, and end up calibrated against the prof's actual scope and emphasis for that module. Recall + application + scenario synthesis, mixed roughly thirds.

The deck is **practice tooling**, not curriculum. It does not replace the atoms; it tests against them. After completing the deck Anders should know which atoms he's solid on and which ones to re-read.

## Your inputs

You receive **one module slug** (e.g. `06-modelsel`) and produce one deck HTML file. Read:

- `docs/concepts-manifest.md`: filter to your module's slice (atoms with `module: <your-slug>`) plus any Specials with your module in their `modules:` list. **Every** atom in your slice gets ≥ 1 question. Every Special touching your module gets ≥ 1 question framed in *this module's* context.
- `wiki/concepts/<slug>.md` for every atom in your slice: pull formulas, traps, exam signals, exercise instances. The "Pitfalls" and "Exam signals" sections of an atom are MC questions waiting to be formatted.
- `wiki/lectures/L<NN>-<your-module-slug>*.md`: the lectures covering your module. Verbatim prof quotes power T/F statements and scenario distractors.
- `wiki/mocs/m<NN>-<your-slug>.md`: its `## Out of scope` section tells you what NOT to write about. (If the MOC doesn't yet exist, derive scope from `docs/scope.md`.)
- `docs/scope.md`: **canonical** authority for what's in vs out, the source hierarchy (exercises > lectures > slides), the past-exam translation rules. Load this first.
- `exam_analysis.md`: useful synthesis (question patterns §3, direction-of-effect traps, dataset templates, opinionated takes). NOT canonical for scope; load for the synthesis content only.
- `exercises/Exercise<N>/` and `exercises/compulsory-exercise-1.md` for problems that touch your module's atoms. The prof flagged "especially the exercises", exercise patterns dominate question selection.
- `exams/TMA4268_2023_Exam.Rmd`, `..._2024_Exam.Rmd`, `..._2025_Exam.Rmd`: past papers. Apply `docs/scope.md`'s translation rules. **Vary numbers and scenarios; never replicate verbatim.**
- `web/templates/deck.md`: the canonical structural + quality spec. Your output must satisfy every checklist item in §10 of that template.
- `web/static/decks/_example.html`: the structural smoke-test reference. Treat as the ground-truth HTML shape for the four question patterns.

You do NOT touch `notes/` (off-limits per project CLAUDE.md), `book/` (deep treatment but not the prof's scope), or `archive/` (dead). You do NOT modify atoms, lectures, MOCs, or the manifest.

## What to produce

One file: `web/static/decks/m<NN>-<your-slug>.html`. Self-contained HTML, links to sibling `exam.css` + `exam.js`. No Quartz chrome, the deck is a focused full-page exam UI. Follow the skeleton in [[../templates/deck]] §2 and the per-question templates in §3 exactly.

## Length

20–30 questions. **100 points total.** Bigger modules (`03-linreg`, `04-classif`, `06-modelsel`, `09-boosting`, `11-nnet`) lean toward 28–30; lighter ones (`07-beyondlinear`, `08-trees`, `10-unsuper`) toward 20–24. Skip module 12, it has no atoms.

## Difficulty mix (target)

> **Hard floor: ≥50% single-select MC per deck.** Single-MC (4 options pick one — §3.1 / §3.3 / §3.4 in [[../templates/deck]]) is the consolidation backbone. The remaining ≤50% goes to multi-statement T/F. See [[../templates/deck]] §6 for the per-size table.

Within the single-MC budget, aim for ~⅓ each of:

- **Recall** (definition / formula / "which of these is true").
- **Application or computation** (plug numbers, read a table, decode a coefficient, follow an algorithm one step).
- **Scenario / output interpretation** (small data + interpretation, cross-atom comparison, "pick the right method given $X$", trap recognition). The prof's heaviest 2026 shape — `exam_analysis.md` §3 + §4d (worked-example datasets) give ready scaffolds.

T/F multi-statement should bias toward **direction-of-effect** questions — each row of `exam_analysis.md` §4b is a candidate sub-statement.

## Question types you may use

1. **Single-correct MC**: the staple. 4 options A–D, one correct, distractors built from typical misunderstandings.
2. **Multi-statement T/F**: for "which of the following are true" questions. 3–5 statements, each independently scored. **Always prefer this over multi-correct MC**, which `exam.js` does not support.
3. **Computation question**: single-MC mechanic, body sets up a numerical scenario, options are candidate values. Distractors encode named arithmetic mistakes (the explanation must name them).
4. **Scenario question**: single-MC mechanic, body is a paragraph or small data, options are candidate interpretations. Distractors are typical wrong readings flagged by the prof in lecture.

## Authoring phases

Run these in order. Phases A–C source from existing materials with provenance flags; phase D fills the rest with synthesised questions; D.5 is the per-question discipline; E is the quality pass.

### Phase A — ISLP conceptual sweep (soft cap ~5 lifts)

Open `book/<NN>-<slug>.md` → `## X.<Y> Exercises` → `### Conceptual` only. **Skip `### Applied` (R coding) entirely.**

For each conceptual problem:

1. **In-scope per `docs/scope.md`?** If no → DROP. Do not adapt and re-flag — once the OOS mechanic is replaced, it's no longer the ISLP exercise, so the flag would lie.
2. **MC-ifies cleanly?** Some ISLP conceptual problems are essentially MC already; others convert to T/F multi-statement; some don't fit either shape and should be dropped.
3. If yes → lift the question stem verbatim, flag `<span class="exam-q__src">ISLP §<X> Q<n></span>`. **Soft cap ~5 ISLP-direct per deck.** Stop early if cap hit; quality > quantity.

Distractor reformulation: if the source's distractors are mutations of the correct answer rather than misconception-anchored, **rewrite the distractors** per [[../templates/deck]] §4.5 and **keep the flag**. The flag means "the question stem came from there", not "the option set is identical to the source's."

### Phase B — past-exam sweep

Open `exams/TMA4268_2023_Exam.Rmd`, `..._2024_Exam.Rmd`, `..._2025_Exam.Rmd`. Bias toward 2024/2025 since the prof walked through those in [[../wiki/lectures/L27-summary]] and explicitly described his 2026 reformatting.

For each question on this module's atoms:

1. **In-scope per `docs/scope.md`?** Drop OOS (SVM, F-test mechanics, AIC algebra, multi-class logistic regression, Bonferroni etc.).
2. **Translatable per `docs/scope.md`'s "Past exam style → 2026 translation" table?** Coding questions become output-interpretation; if the original is pure coding with no conceptual core, drop.
3. If kept → MC-ify (change numbers, vary the angle), flag `<span class="exam-q__src">Exam <year> P<n></span>`.

L27 already enumerates the prof's preferred 2026 reformatting for many 2025 problems — these are the highest-fidelity templates and should anchor several questions in the deck.

### Phase C — exercise sweep (CE1, CE2, recommended)

Open `exercises/Exercise<N>/` and the relevant problems in `exercises/compulsory-exercise-1.md` and `exercises/compulsory-exercise-2.md`.

For each problem on an in-scope atom:

1. If verbatim or near-verbatim lift → flag `<span class="exam-q__src">CE1 P4</span>` or `<span class="exam-q__src">Ex5.3</span>` (whichever applies).
2. If number-changed mirror of an exercise → no flag (counts as synthesised, but the prof's "especially the exercises" rule still applies).

### Phase D — synthesised fill

Hit [[../templates/deck]] §6 difficulty mix (≥50% single-MC floor) and §7 atom coverage with synthesised questions. For each atom in your slice:

- Mine the atom's `Pitfalls` and `Exam signals` sections.
- Mine `exam_analysis.md`:
  - **§4b** for direction-of-effect T/F seeds (22 rows, each a candidate sub-statement).
  - **§4d** for module-mapped worked-example datasets — re-use the names: Default + Smarket + South African heart for m04 logistic, Boston Housing + Brain Injury + Ozone for m08 trees, Wage for m07 GAMs, Hitters for m06 subset selection, Iris + cork-tree for m10 LDA/PCA, MNIST for m11 NNs.
  - **§4f** for the 12 common-mistake traps as distractor seeds.
  - **§4g** for procedural templates G1–G6 — flagged by the prof as exam-likely. G1 hierarchical clustering by hand → m10; G2 LDA boundary derivation → m04; G3 logistic-coefficient odds → m04; G4 NN parameter count → m11; G5 bootstrap SE → m05; G6 k-fold CV with one-SE rule → m05/m06.
- Cross-check against the L27 walkthrough — for each module the prof showed a 2025-question reformatting that constitutes his **stated** preferred shape; mirror that shape for at least one question.

Active expert-eye search: for each atom, ask "what would an exam-writer naturally test? definitions, derivations, direction of an effect, decode-a-formula, interpret-this-output, what-changes-when-X." Use `exam_analysis.md` §3's seven question patterns as the menu. If the prof addressed it → write a question; if not → either omit (peripheral) or flag the wiki gap in your final report rather than fabricating.

### Phase D.5 — per-question misconception ledger (mandatory)

Before finalising each question, write a short ledger in your scratch (NOT in the output HTML):

```
Concept tested:    <atom slug + sub-claim>
Correct answer:    <one sentence>
Misconceptions considered (5):
  1. <bullet>
  2. <bullet>
  3. <bullet>
  4. <bullet>
  5. <bullet>
Distractors selected (3 strongest, mapped to letters):
  - A or B or C or D: <misconception name>
  - …
  - …
Source flag (if any): <ISLP §X Qn / Exam YYYY Pn / CEn Pm / Ex<i>.<j> / none>
```

For ISLP/exam/CE-direct lifts: audit the source's distractors against this ledger. If they don't pass [[../templates/deck]] §4.5, **rewrite them and keep the flag**.

The ledger drives the explanation block: each `<p>` dismissing a distractor names the misconception. "B forgets the bias term in the NN parameter count," not "B is just wrong because it gives 11."

### Phase E — quality pass

Run [[../templates/deck]] §10 checklist line by line. Specifically:

1. **Atom coverage**: every atom in your module slice has ≥1 question; every Special whose `modules:` includes you has ≥1 question framed in your module's context.
2. **Exercise coverage**: every recommended exercise problem in your `Exercise<N>/` folder and the relevant CE1/CE2 problems is mirrored by at least one question (flagged or synthesised number-changed mirror).
3. **Single-MC floor**: ≥50% single-MC per deck size table in [[../templates/deck]] §6.
4. **Total points = 100.**
5. **Length parity, position rotation, form-only test, no tell-tale words, no mere-negation distractors, no trivial-wording falsification** — see [[../templates/deck]] §5.
6. **Out-of-scope material**: confirm the deck has none. `docs/scope.md` is canonical.
7. **Source-flag rendering**: each flagged question's `.exam-q__src` span is in `.exam-q__head` after `.exam-q__points`.

Report results in your final summary, including a count of questions per source-flag variant.

## Option-quality rules: the gold

The single most important section. Read [[../templates/deck]] §5 in full *and* [[../templates/deck]] §4.5 (misconception-first distractor generation, the upstream discipline). Summary:

- **Misconception-first** (§4.5): each distractor encodes a *named* student misconception, not a mutation of the correct answer. Brainstorm 5, pick 3.
- **Length parity:** correct option is NOT systematically the longest. Across the deck, "longest", "middle", "shortest" each correct ~equal often.
- **Position rotation:** A/B/C/D distribution roughly even.
- **No tell-tale words.** Avoid *always/never/only* in correct options; avoid *typically/usually* in obvious-correct options.
- **No mere negations.** A distractor cannot be a syntactic flip of the correct answer.
- **No trivial-wording falsification.** Distractor falsity comes from the misconception, not from a stray *always*/*never*.
- **Distractors are plausible**, typical wrong reasonings the explanation can name. No filler.
- **Same form, same detail level** across all four options.
- **Form-only test** before publishing each question: hide the question text, can you still pick correct? If yes, rewrite.

## Quote anchors

When you cite a verbatim prof quote in an explanation, link it via the lecture wikilink rendered as a Quartz URL: `<a href="/lectures/L13-modelsel-2">L13-modelsel-2</a>`. The slug matches `wiki/lectures/L<NN>-<slug>.md`. Quartz routes that URL to the rendered lecture page.

Bias toward including verbatim quotes whenever the prof's exact wording is more memorable, more precise, or more characteristically his than your paraphrase would be.

## Linking from explanations to atoms

Every question's `<p class="ref">…</p>` block carries at least one `<a href="/concepts/<slug>">slug</a>` link to the load-bearing atom(s). Multiple atoms welcome, especially when a question crosses an atom boundary or invokes a Special. Lecture links optional, used when an explanation cites a quote.

## Hard reminders

- **DO NOT modify** the wiki, the manifest, atoms, lectures, MOCs, or `docs/scope.md`. You only write `web/static/decks/m<NN>-<slug>.html`.
- **DO NOT read `notes/`**, off-limits per CLAUDE.md.
- **DO NOT replicate past-exam questions verbatim.** Translate, change numbers, vary the angle. Treat past exams as *style references*, not answer keys.
- **DO NOT write questions on out-of-scope material.** `docs/scope.md` is canonical. If a topic is out, no question, even if the topic is in ISLP.
- **DO NOT exceed 30 questions**, if you have more good content than fits, save the surplus for a future revision or for an eventual mock-exam set.

## What you're not doing

- Not declaring importance tiers, depth is visible in the *count* of questions per atom and the *quality* of distractors, not in metadata.
- Not creating mock exams, those are deferred.
- Not paraphrasing prof signals, verbatim if it carries.
- Not touching `web/static/decks/exam.{css,js}`, the JS contract is fixed (parses `Correct answer: X` and `<strong>True/False</strong>` markers).

Trust the manifest, the atoms, and `docs/scope.md`. They did the analysis. Your job is composition under the deck.md quality bar.
