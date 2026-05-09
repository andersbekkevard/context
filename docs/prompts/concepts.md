# Concepts pass: agent brief

You're producing self-contained expert briefings for the TMA4268 exam-prep context bank, one atom per named concept assigned to your module slice. Read [[../overview]] and [[../concepts]] first.

## North Star

When Claude (the tutor) loads one of your atoms, it should be able to answer any reasonable exam question about that concept without traversing other files. Your atom is **Claude's raw material for tutoring Anders.** It is not a textbook recreation; it is the prof-specific signal layered on top of what Claude already knows.

## Your inputs

You receive a slice of `docs/concepts-manifest.md` listing the atoms you are responsible for. The manifest has two kinds of slice:

- **Module slice**: atoms with a singular `module:` field. You are one of 12 module agents, each handling its own module's atoms.
- **Specials slice**: atoms with a plural `modules:` field (cross-module concepts like bias-variance, regularization, CV). You are the 13th agent, the specials agent.

For each atom in your slice, read:
- The relevant `wiki/lectures/L<NN>-<slug>.md` files (for Specials, this spans modules)
- The slide deck(s) under `modules/<...>/`
- The relevant book chapter(s) under `book/<NN>-<slug>.md`, use to fill `isl-ref:` and to flesh out the deep treatment Claude can point Anders at
- The relevant exercise files under `exercises/Exercise<N>/` and (for problems that touch this concept) the compulsory exercises `exercises/compulsory-exercise-1.md` and `exercises/compulsory-exercise-2.md`
- Past exams `exams/TMA4268_2023_Exam.Rmd`, `..._2024_Exam.Rmd`, `..._2025_Exam.Rmd`, for exam-pattern hints (with translation rules per `docs/scope.md`)
- `docs/scope.md`: **canonical** authority for what's in scope, what's out, and the source hierarchy. Load this first.
- `exam_analysis.md`: useful synthesis (tier rankings, traps, datasets, opinionated takes) but **not canonical** for scope. Use for those auxiliary signals, not for in/out decisions.

## What to produce

For each atom in your slice, write `wiki/concepts/<slug>.md` following `docs/templates/concepts.md` exactly.

The atom is a **briefing for a knowledgeable peer**, imagine handing it to another stat learning professor who needs to teach *this* prof's exam. They already know what ridge regression is, what backprop does. They don't know:

- What's on *this* curriculum (vs the textbook)
- The exact definitions *this* prof uses
- Where *he* lays the emphasis
- The insights and framings *he* brought up
- The traps *he* flagged
- Which exercises drill this concept
- How *this* exam might ask about it

That gap is the atom's job to fill. **Skip generic stat learning content**: Claude knows it from training, and ISLR holds the deep treatment. The atom's value is everything specific to *this* course.

## Atom depth scales with prof's treatment

There is no external tier ranking. Read the lectures + slides for your atom's concept and let the prof's actual treatment dictate depth:

- Heavy treatment + multiple exam-flag quotes + recurring across lectures → richly developed atom (toward 200–250 lines)
- Standard treatment, one solid lecture pass → standard atom (~120 lines)
- Brief mention or "you can read about this in the book" → stub (40–80 lines: definition + ISLR pointer + scope note)

Length bound: **80–250 lines** (cross-cutting atoms can exceed). Don't pad to hit a number; don't over-compress to hit a number.

## Granularity (when in doubt)

One atom = **one named idea Anders would naturally ask Claude about as a single question**.

- "Ridge regression": yes (you'd ask "explain ridge")
- "L2 norm": no (you'd never ask standalone; lives inside ridge)
- "Regularization techniques": no (too coarse; that's a MOC topic)
- "Walk me through hierarchical clustering by hand": yes (procedural, but standalone-askable)
- "Difference between LDA and QDA": no (Claude composes from lda atom + qda atom)

The manifest has already drawn the lines. If you find a manifest entry that genuinely shouldn't be its own atom, flag it in your final report rather than silently skipping or merging.

## Specials atoms (if you are the specials agent)

The Specials section of the manifest contains concepts that genuinely span multiple modules with no natural single home (bias-variance, regularization, cross-validation, standardization, etc.). These atoms have a plural `modules:` field rather than singular `module:`.

If you are the specials agent:
- Read all lectures listed in `lectures:` (will span modules)
- Skim slide decks for every module in `modules:` for relevant treatment
- The atom captures the prof's treatment across **every** module that touches it, in chronological order
- Include a `## Returns in other modules` section showing how the prof revisits the concept later in the course
- Per-module MOCs will wikilink back to your atom, your job is to ensure the atom is rich enough to compose well from those wikilinks
- The atom's frontmatter uses plural `modules:` (matching the manifest), not singular `module:`

## Exercise coverage sweep (end of module batch)

After writing all atoms in your slice, do an explicit coverage check:

1. List every exercise problem in `exercises/Exercise<N>/` for your module's relevant exercise folder(s) and every problem in the compulsory exercises that touches your module's concepts.
2. For each problem, verify it is referenced in some atom's `## Exercise instances` section.
3. Any unreferenced problem is a signal: either (a) you missed adding it to the right atom (fix), or (b) it tests something that should be its own atom but isn't in the manifest (report), or (c) it tests something out-of-scope (report).

This sweep is non-negotiable. Report results in your final summary.

## Quote anchors

When you cite a verbatim prof quote, anchor it with the lecture wikilink: `> "<quote>" — [[L13-modelsel-2]]`. Date+timestamp is unnecessary. The lecture page is the grep handle. Anders (or a future agent) opens the linked lecture, finds the quote, and reads the surrounding context.

Bias toward including quotes whenever the prof's exact wording is more memorable, more precise, or more characteristically his than your paraphrase would be.

## Active expert-eye search (do this before finalizing each atom)

For each concept, **think like an ISLP-trained stat learning expert** about what should be covered:

1. List what an expert would expect to see for this concept: standard formulas, conventions, derivations, edge cases, computational considerations, comparisons. (For ridge: standardization, intercept-not-penalized, closed-form solution, MAP interpretation, ridge trace, λ via CV, contrast with lasso. For backprop: chain rule, computational graph, vanishing/exploding gradients, vectorized form, mini-batch.)

2. For each item, search lectures and slides to see whether the prof addressed it.

3. If yes → include with a lecture/slide anchor.

4. If no → either omit (if peripheral) or note explicitly in `## Scope vs ISLR` as "Skip in ISLR" so Anders knows the textbook covers it but it's not on this exam.

This catches concepts the prof mentioned briefly that matter, and tells Anders what to skip in the textbook.

## Format rules (enforced by template)

- Filename = manifest slug (hyphen-case)
- Frontmatter exactly as in `docs/templates/concepts.md`
- Wikilinks to related concepts: `[[lasso]]`
- Wikilinks to lectures: `[[L13-modelsel-2]]`
- Math in LaTeX (`$...$`, `$$...$$`)
- Section ordering as in the template

## Hard reminders

- **DO NOT modify `docs/concepts-manifest.md`.** It is the deterministic slug source. If you find a manifest entry that shouldn't exist or a missing concept, report it in your final summary, do not act unilaterally.
- **Overwrite freely.** If a `wiki/concepts/<slug>.md` already exists, overwrite it. Earlier passes may have produced stubs.
- **Module agents do NOT write Specials atoms.** Specials atoms (with plural `modules:` in the manifest) are owned by the specials agent. Module agents wikilink to them via `[[bias-variance-tradeoff]]` etc. but never create or modify them.
- **Do NOT read `notes/`**, off-limits per `CLAUDE.md`.

## What you're not doing

- Not declaring importance tiers, depth is visible in the signals you cite, the section richness, and the exam-pattern count
- Not creating atoms for out-of-scope material (per `docs/scope.md`, the canonical authority)
- Not paraphrasing prof signals, verbatim if it carries
- Not recreating textbook content, the book is on the exam table; point to it
- Not enforcing word counts within an atom (the 80–250 line bound is the only gross-length rule)

Trust your judgment on filler-vs-signal. You're a smart reader.
