# `web/todo.md` — mock-exam build plan

Working document for turning the per-module deck format into full-length mock exams that mirror the TMA4268 May-18 paper. Read alongside `web/templates/deck.md`, `web/prompts/deck-generation.md`, `docs/scope.md`, `exam_analysis.md`, and `wiki/lectures/L27-summary.md`.

## State of play (2026-05-09)

- **Per-module decks**: 11 to be built (`m01-intro` through `m11-nnet`). Spec locked, agents not yet fanned out. Each deck has a ≥50% single-MC floor, ≤50% multi-statement T/F, source-flag pills (ISLP / past-exam / exercise / synthesised), misconception-first distractors.
- **Mock exams**: deferred per `web/templates/deck.md §11`. This file plans that follow-up.
- **Visual shell already mock-exam-ready**: `web/static/decks/exam.css` ships unused `.exam-part`, `.exam-part h2`, `.exam-part .part-desc` styling for sectioned papers. No CSS work needed for the section structure.

## Hard prerequisite

Build the 11 per-module decks first. They are the question pool the mock exam draws from. Without them every mock-exam question would be authored from scratch, ~3× the effort.

## Paths

Three options, in increasing fidelity to the actual exam.

| Path | What you get | Infra effort | Per-paper content effort | When to pick |
|---|---|---|---|---|
| **A** — MVP MC/T-F only | 4-hour, ~80–100-question paper across all modules using only the existing 4 deck question types. No free-text response. | ~3–4h | ~6–8h | Default. Practice value is timed cross-module recall. |
| **B** — Authentic | Path A + free-text response, derivation/pseudocode, hand-calculation question types with self-grade buttons. Matches the prof's stated 7-type 2026 format. | ~10–15h | ~8–12h | One dress rehearsal before exam. |
| **C** — Past-paper translated | Path B + verbatim 2024/2025 papers re-cast in 2026 format per `docs/scope.md` translation table. | Path B only | ~6–10h per past paper | Diminishing returns; the `.Rmd` papers + answer keys also work as plain self-graded study. |

Recommendation: do **Path A first** after the 11 decks land. Re-evaluate before committing to Path B.

---

## Path A — MVP MC/T-F-only mock exam

A long page that wraps cross-module questions into Parts. Reuses the deck HTML/JS contract wholesale.

### A1. Template — `web/templates/mock-exam.md`

Sibling of `deck.md`. Captures:

- Frame: 4 hours, ~80–100 questions, 100 points, ~all 11 modules represented roughly proportional to lecture-hours.
- File layout: `web/static/decks/mock-exam-<NN>.html` (e.g. `mock-exam-01.html`).
- Reuses the same `<!doctype html>` skeleton as a deck (theme-sync, Quartz fonts, KaTeX includes, `exam.css` + `exam.js`). Only difference: longer body and `<section class="exam-part">` wrappers.
- Sectioning: 3 parts, each with `<h2>` and a `.part-desc` italic blurb.
  - **Part 1 — Multiple choice** (~40 questions, 35 points, ~75 min): the §3.1 / §3.3 / §3.4 single-MC mechanics. Fast, high-volume.
  - **Part 2 — True/false** (~25 questions × 4 statements, 35 points, ~75 min): the §3.2 multi-statement T/F mechanic. Direction-of-effect heavy.
  - **Part 3 — Scenario interpretation** (~15 questions, 30 points, ~90 min): single-MC mechanic over output tables / plots / dendrograms / scree plots. The prof's heaviest 2026 shape.
- Module coverage rule: every module 02–11 has ≥4 questions; tier-1 modules (m03 linreg, m04 classif, m06 modelsel, m11 nnet) get ≥10. Module 01 (intro) gets 1–2. Bias-variance gets ≥3 questions across the paper (prof's Apr-28 commitment).
- Source-flag rules carry over from `deck.md §3.5`. The whole 2024/2025 papers are fair game for `Exam YYYY P<n>` flags after translation per `docs/scope.md`.
- Quality bar inherits `deck.md §4.5` (misconception-first), `§5` (option quality), `§6` (50% single-MC floor still applies).

**Effort:** ~1h.

### A2. Prompt — `web/prompts/mock-exam-generation.md`

Sibling of `deck-generation.md`. Tells an agent how to assemble one mock exam:

1. **Question pool sweep**: read all 11 module decks at `web/static/decks/m<NN>-<slug>.html`. Score each question on (a) atom centrality, (b) prof-flagged exam-likeness, (c) difficulty.
2. **Module quota**: pick questions per module roughly proportional to atom count + lecture hours. No question repeats from a deck verbatim — change numbers, swap dataset, vary the angle.
3. **Specials sweep**: bias-variance, regularization, cross-validation, standardization should each appear ≥2× across the paper.
4. **Past-exam direct lifts**: per the `Exam <year> P<n>` flag rule, lift from 2024 + 2025 after translation. Bias toward L27's walked-through reformatting.
5. **Synthesise + lift balance**: target ~30% past-exam-direct, ~10% ISLP-direct, ~10% exercise-direct, ~50% synthesised.
6. **Authoring phases A–E** carry over from `deck-generation.md`.

**Effort:** ~1h.

### A3. First mock-exam HTML — `web/static/decks/mock-exam-01.html`

The first concrete artefact. Built by an agent following the template + prompt above, drawing from the 11 module decks.

**Effort:** ~6–8h of agent work.

### A4. Optional — `exam.js` per-part scoring

Default `exam.js` aggregates score across all questions on the page. For mock exams it would help to also see per-part breakdown. Two ways:

- **Lightweight**: read the existing `.exam-part` wrapper and emit a per-section sub-tally in the FAB panel.
- **Skip**: keep the existing single tracker. Anders can eyeball per-part performance from the question colours.

Pick lightweight only if it's clearly useful after running mock-exam-01 once.

**Effort:** ~1–2h if done.

### A5. Linking from MOC + index

- Add a "Mock exams" section to `wiki/index.md` linking to `/decks/mock-exam-01.html`.
- (Optional) Add a per-module "Try this question on the mock exam →" link from each MOC's `## Practice` section.

**Effort:** ~30 min.

### A6. Pre-publish checklist (Path A)

Run before declaring `mock-exam-01.html` done:

- [ ] Total points = 100.
- [ ] ~80–100 questions; section split ~40 / ~25 / ~15.
- [ ] Every module 02–11 has ≥4 questions; tier-1 modules ≥10.
- [ ] Bias-variance appears in ≥3 questions.
- [ ] No verbatim repeats from any module deck.
- [ ] Source-flag rendering correct (deck.md §3.5).
- [ ] Length parity, position rotation, form-only test (deck.md §5).
- [ ] No mere-negation distractors, no trivial-wording falsification (deck.md §5 + §4.5).
- [ ] Out-of-scope check (`docs/scope.md`).
- [ ] Browser smoke test: click every question, FAB updates, explanations open, KaTeX renders, theme toggle works.

**Total Path A effort: ~10–14h** (mostly the one-time content build for paper #1; subsequent papers are ~6–8h each).

---

## Path B — Authentic open-response support

Adds the three real-exam question types the deck format doesn't have. All use the same self-grade UX: stem → optional `<textarea>` for typed scratch → `<details>` with model answer → 3 buttons (Correct / Partial / Wrong) feeding a points fraction.

### B1. New question type spec — `deck.md §3.6 Open-response`

```
§3.6 Open-response (free-text / derivation / hand-calc)

For questions where the answer is prose, math, pseudocode, or worked
arithmetic. Student types in a textarea (or scribbles on paper, optional),
clicks "Show model answer", self-grades against it. Three buttons map to
100% / 50% / 0% of the question's points.

HTML pattern:

<article class="exam-q exam-q--open">
  <header class="exam-q__head">
    <span class="exam-q__num">Question 14</span>
    <span class="exam-q__points">8 points</span>
    <span class="exam-q__src">Exam 2024 P3</span>
  </header>
  <p class="q-text">…question text…</p>
  <textarea class="exam-q__scratch" rows="6" placeholder="Optional: type your work…"></textarea>
  <details class="fasit-details">
    <summary>Show model answer</summary>
    <div class="fasit-body">
      <p>…model answer with full reasoning…</p>
    </div>
  </details>
  <div class="exam-q__selfgrade">
    <button data-grade="correct">Got it</button>
    <button data-grade="partial">Partial</button>
    <button data-grade="wrong">Missed it</button>
  </div>
</article>
```

Rules:
- Self-grade buttons lock on first click (parallel to MC option locking).
- Points awarded: Correct = full, Partial = half, Wrong = 0.
- The `<textarea>` is optional — students hand-write on paper anyway. It's there for typing fluency drills if useful.
- Source-flag rules apply identically.
- `deck.md §4.5` misconception-first does NOT apply (no distractors). Quality bar shifts to: model answer must be complete, name the rubric for partial credit, cite the prof's framing.

**Effort:** ~1h (spec only).

### B2. CSS — `web/static/decks/exam.css`

Add ~30 lines: textarea styling, self-grade button row, locked/correct/partial/wrong states. Match the existing correct/wrong colour scheme.

**Effort:** ~1h.

### B3. JS — `web/static/decks/exam.js`

The non-trivial chunk. New behaviour:

- Detect `.exam-q--open` articles on load.
- Wire up self-grade buttons: lock on first click, mark the chosen state, open `<details>`, award points fraction.
- Update FAB tracker to mix auto-scored (MC + T/F) and self-scored (open-response) points into a single running total.
- Optional: separate the panel display so user sees "auto: 45/60, self-graded: 22/40" rather than only the sum.

Risk: the FAB-tracker code currently assumes deterministic scoring. Adding a self-graded path has to not break the existing MC + T/F flow. Test all four question shapes after the change.

**Effort:** ~3–4h including tests.

### B4. Numeric-input question type (cheap, optional)

`<input type="number">` + `data-expected` + `data-tolerance`. exam.js parses, compares with absolute tolerance, marks correct/wrong. ~50 lines including CSS. Useful for hand-calc questions where the answer is one number (odds → probability, MSE from a small table).

**Effort:** ~1–2h.

### B5. Multi-part question wrapping

For Q3a / Q3b / Q3c sharing a stem. HTML: `<article class="exam-q-multi">` containing one shared stem and N `.exam-q-sub` children. CSS: indent children, share number prefix. Each child is a self-contained question of any of the supported types — no JS changes needed.

**Effort:** ~1h.

### B6. Update `web/templates/mock-exam.md` with Path-B types

Add §s for §3.6 open-response, multi-part wrapping, numeric input. Update the part split to include a Part 4: Open response (~2 questions, ~10 points), since the prof guarantees "at least one mathy/derivation question."

**Effort:** ~1h.

### B7. First authentic mock exam — `mock-exam-02.html`

Built using the new question types. Should include:
- ≥1 derivation question (bias-variance decomposition, MLE=LS, LDA boundary, NN parameter count).
- ≥2 prose-interpretation questions (regression output table, ROC curve, dendrogram).
- ≥1 hand-calc question (odds↔prob, hierarchical clustering 4×4, k-means by hand).

**Effort:** ~8–12h of agent work.

**Total Path B effort: ~16–22h** infra + first authentic paper.

---

## Path C — Past-paper translation

Only after Path B. Translate `exams/TMA4268_2024_Exam.Rmd` and `..._2025_Exam.Rmd` per `docs/scope.md`'s "Past exam style → 2026 translation" table:
- Coding tasks → output-interpretation prose.
- "Memorise the `gbm()` arguments" → "Explain the boosting hyperparameters' bias-variance effect."
- Drop OOS questions silently.
- Each kept question gets `<span class="exam-q__src">Exam <year> P<n></span>`.

Output: `web/static/decks/mock-exam-2024-translated.html` and `mock-exam-2025-translated.html`.

**Effort: ~6–10h per past paper.**

---

## Question pool: where content comes from

In priority order:

1. **The 11 module decks** (after they land): the primary pool. Vary numbers, swap datasets, change angle — never verbatim repeat.
2. **Past TMA4268 exams** (`exams/TMA4268_{2023,2024,2025}_Exam.Rmd`): translate per `docs/scope.md`. Bias toward 2024 + 2025 since the prof walked through them in `wiki/lectures/L27-summary.md`.
3. **L27 walkthrough** (`wiki/lectures/L27-summary.md` § "Walkthrough: 2025 exam, problem-by-problem"): the prof's own re-formattings of 2025 questions for 2026. Highest-fidelity templates per module.
4. **`exam_analysis.md` §4d worked datasets**: scenario scaffolds — Default + Smarket for m04, Boston Housing for m08, Wage for m07, Hitters for m06, Iris + cork-tree for m10, MNIST for m11.
5. **`exam_analysis.md` §4g procedural templates G1–G6**: hand-calc question recipes the prof flagged as exam-likely.
6. **ISLP `### Conceptual` end-of-chapter exercises** (per-chapter): same soft cap ~5 lifts as decks, applied across the whole paper.

## Out-of-scope reminder

`docs/scope.md` is canonical. SVM, F-test mechanics, AIC/BIC algebra, multi-class logistic, Bonferroni / FDR, survival, time series, weight initialisation schemes, BPTT, RNN/CNN architecture details, history questions — none of these get a question, regardless of how good the source material is.

## Decision log

- **2026-05-09**: scoped this file. Decision to defer mock-exam build until the 11 module decks land. Path A picked as default; Path B reserved for one dress-rehearsal paper if useful.

---

## Quick reference: what to read before starting

- `web/templates/deck.md` (full): structural + quality spec, including new §3.5, §4.5, §6 floor.
- `web/prompts/deck-generation.md` (full): authoring phases A–E, misconception ledger.
- `docs/scope.md` (full): canonical scope authority.
- `exam_analysis.md` §3, §4b, §4d, §4f, §4g: question-pattern menu, direction-of-effect, datasets, traps, procedural templates.
- `wiki/lectures/L27-summary.md` § "Walkthrough: 2025 exam" + "The mathy question (2024)": the prof's own 2026 reformattings.
