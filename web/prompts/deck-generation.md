# Deck-generation pass — agent brief

You're producing **one MCQ deck** for one TMA4268 module — an interactive HTML page that drills the module's atoms via 20–30 multiple-choice questions. Read [[../templates/deck]] in full first.

## North Star

When Anders opens the deck for module $N$, he should be able to click through ~25 questions in 30–45 minutes, get rapid right/wrong feedback per click, and end up calibrated against the prof's actual scope and emphasis for that module. Recall + application + scenario synthesis, mixed roughly thirds.

The deck is **practice tooling**, not curriculum. It does not replace the atoms; it tests against them. After completing the deck Anders should know which atoms he's solid on and which ones to re-read.

## Your inputs

You receive **one module slug** (e.g. `06-modelsel`) and produce one deck HTML file. Read:

- `docs/concepts-manifest.md` — filter to your module's slice (atoms with `module: <your-slug>`) plus any Specials with your module in their `modules:` list. **Every** atom in your slice gets ≥ 1 question. Every Special touching your module gets ≥ 1 question framed in *this module's* context.
- `wiki/concepts/<slug>.md` for every atom in your slice — pull formulas, traps, exam signals, exercise instances. The "Pitfalls" and "Exam signals" sections of an atom are MC questions waiting to be formatted.
- `wiki/lectures/L<NN>-<your-module-slug>*.md` — the lectures covering your module. Verbatim prof quotes power T/F statements and scenario distractors.
- `wiki/mocs/m<NN>-<your-slug>.md` — its `## Out of scope` section tells you what NOT to write about. (If the MOC doesn't yet exist, derive scope from `docs/scope.md`.)
- `docs/scope.md` — **canonical** authority for what's in vs out, the source hierarchy (exercises > lectures > slides), the past-exam translation rules. Load this first.
- `exam_analysis.md` — useful synthesis (question patterns §3, direction-of-effect traps, dataset templates, opinionated takes). NOT canonical for scope; load for the synthesis content only.
- `exercises/Exercise<N>/` and `exercises/compulsory-exercise-1.md` for problems that touch your module's atoms. The prof flagged "especially the exercises" — exercise patterns dominate question selection.
- `exams/TMA4268_2023_Exam.Rmd`, `..._2024_Exam.Rmd`, `..._2025_Exam.Rmd` — past papers. Apply `docs/scope.md`'s translation rules. **Vary numbers and scenarios; never replicate verbatim.**
- `web/templates/deck.md` — the canonical structural + quality spec. Your output must satisfy every checklist item in §10 of that template.
- `web/static/decks/_example.html` — the structural smoke-test reference. Treat as the ground-truth HTML shape for the four question patterns.

You do NOT touch `notes/` (off-limits per project CLAUDE.md), `book/` (deep treatment but not the prof's scope), or `archive/` (dead). You do NOT modify atoms, lectures, MOCs, or the manifest.

## What to produce

One file: `web/static/decks/m<NN>-<your-slug>.html`. Self-contained HTML, links to sibling `exam.css` + `exam.js`. No Quartz chrome — the deck is a focused full-page exam UI. Follow the skeleton in [[../templates/deck]] §2 and the per-question templates in §3 exactly.

## Length

20–30 questions. **100 points total.** Bigger modules (`03-linreg`, `04-classif`, `06-modelsel`, `09-boosting`, `11-nnet`) lean toward 28–30; lighter ones (`07-beyondlinear`, `08-trees`, `10-unsuper`) toward 20–24. Skip module 12 — it has no atoms.

## Difficulty mix (target)

- ~⅓ recall (definition / formula / "which of these is true").
- ~⅓ application or computation (plug numbers, read a table, decode a coefficient, follow an algorithm one step).
- ~⅓ scenario synthesis (small data + interpretation, cross-atom comparison, "pick the right method given $X$", trap recognition).

## Question types you may use

1. **Single-correct MC** — the staple. 4 options A–D, one correct, distractors built from typical misunderstandings.
2. **Multi-statement T/F** — for "which of the following are true" questions. 3–5 statements, each independently scored. **Always prefer this over multi-correct MC**, which `exam.js` does not support.
3. **Computation question** — single-MC mechanic, body sets up a numerical scenario, options are candidate values. Distractors encode named arithmetic mistakes (the explanation must name them).
4. **Scenario question** — single-MC mechanic, body is a paragraph or small data, options are candidate interpretations. Distractors are typical wrong readings flagged by the prof in lecture.

## Active expert-eye search (do this before finalizing)

For each atom in your slice, **think like a stat-learning professor designing an exam** about what's testable:

1. List what an exam-writer would naturally ask (definitions, derivations, direction of an effect, pick-the-trap, decode-a-formula, interpret-this-output, what-changes-when-X). Use `exam_analysis.md` §3 for canonical TMA4268 question patterns.
2. For each item: search the atom + the lectures to confirm the prof addressed it.
3. If yes → write a question, anchor the explanation in the prof's framing. Cite verbatim quote where wording carries.
4. If no → either omit (if peripheral) or — if the missing piece is non-trivial — flag it in your final report as a possible gap in the wiki rather than fabricating a question.

This catches the exam-likely ideas the prof flagged but you'd otherwise miss.

## Option-quality rules — the gold

The single most important section. Read [[../templates/deck]] §5 in full. Summary:

- **Length parity:** correct option is NOT systematically the longest. Across the deck, "longest", "middle", "shortest" each correct ~equal often.
- **Position rotation:** A/B/C/D distribution roughly even.
- **No tell-tale words.** Avoid *always/never/only* in correct options; avoid *typically/usually* in obvious-correct options.
- **Distractors are plausible** — typical wrong reasonings the explanation can name. No filler.
- **Same form, same detail level** across all four options.
- **Form-only test** before publishing each question: hide the question text, can you still pick correct? If yes, rewrite.

## Exercise coverage sweep (end of deck batch)

After writing all questions, run an explicit coverage check:

1. List every atom in your module slice + every Special touching your module → confirm each has ≥ 1 question.
2. List every recommended exercise problem in your module's `Exercise<N>/` folder + the relevant CE1 problems → confirm at least one question mirrors each major problem (numbers changed). Out-of-scope problems are skipped silently.
3. Run the §10 checklist from [[../templates/deck]] line by line. Especially: total points = 100, length parity, position rotation, form-only test passed.

Report results in your final summary.

## Quote anchors

When you cite a verbatim prof quote in an explanation, link it via the lecture wikilink rendered as a Quartz URL: `<a href="/lectures/L13-modelsel-2">L13-modelsel-2</a>`. The slug matches `wiki/lectures/L<NN>-<slug>.md`. Quartz routes that URL to the rendered lecture page.

Bias toward including verbatim quotes whenever the prof's exact wording is more memorable, more precise, or more characteristically his than your paraphrase would be.

## Linking from explanations to atoms

Every question's `<p class="ref">…</p>` block carries at least one `<a href="/concepts/<slug>">slug</a>` link to the load-bearing atom(s). Multiple atoms welcome — especially when a question crosses an atom boundary or invokes a Special. Lecture links optional, used when an explanation cites a quote.

## Hard reminders

- **DO NOT modify** the wiki, the manifest, atoms, lectures, MOCs, or `docs/scope.md`. You only write `web/static/decks/m<NN>-<slug>.html`.
- **DO NOT read `notes/`** — off-limits per CLAUDE.md.
- **DO NOT replicate past-exam questions verbatim.** Translate, change numbers, vary the angle. Treat past exams as *style references*, not answer keys.
- **DO NOT write questions on out-of-scope material.** `docs/scope.md` is canonical. If a topic is out, no question — even if the topic is in ISLR.
- **DO NOT exceed 30 questions** — if you have more good content than fits, save the surplus for a future revision or for an eventual mock-exam set.

## What you're not doing

- Not declaring importance tiers — depth is visible in the *count* of questions per atom and the *quality* of distractors, not in metadata.
- Not creating mock exams — those are deferred.
- Not paraphrasing prof signals — verbatim if it carries.
- Not touching `web/static/decks/exam.{css,js}` — the JS contract is fixed (parses `Correct answer: X` and `<strong>True/False</strong>` markers).

Trust the manifest, the atoms, and `docs/scope.md`. They did the analysis. Your job is composition under the deck.md quality bar.
