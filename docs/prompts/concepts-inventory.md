# Concepts inventory pass: agent brief

You're the single most consequential agent in the wiki pipeline. Your job is to read the entire course corpus and produce `docs/concepts-manifest.md`, the deterministic atom list that 13 downstream fan-out agents will use to write the wiki's concept atoms. Every decision you make cascades. Get it right.

Read [[../overview]], [[../concepts]], and [[../templates/concepts]] first.

## North Star

When the fan-out agents finish, the wiki should have one atom per named idea Anders would naturally ask Claude about as a single question, no fragmentation, no duplicates, no atoms for out-of-scope material, no missing in-scope concepts. Your manifest is what makes that possible.

## Your operating principle

**You are the instantiation of the prof's scope rule:**

> "If it was covered in the slides or the exercises, it's fair game. If it's only in the book and we didn't talk about it in class or exercises, it won't be on the test."

The prof additionally emphasized **"especially the exercises"** as the strongest exam-relevance signal. Treat coverage in the recommended exercises and the two compulsory exercises as the *highest*-weight indicator that a concept is in scope and exam-likely. If something appears in an exercise, it gets an atom, period. If something appears only in slides without lecture or exercise reinforcement, still in scope, but less load-bearing.

Every decision you make about whether something becomes an atom flows from this rule. Slides + lectures + exercises are the universe of in-scope concepts. The book and the past exams play no role at this stage, the book is for *later* (the fan-out agents use it to flesh out treatment of in-scope ideas), and past exams are not on the table because the prof has explicitly redesigned this year's exam. Don't read either.

## Your inputs

Read all of these. **Do not read `notes/`: it is off-limits per `../overview` and `CLAUDE.md`. Do not read `book/`: ISLP is for the fan-out stage, not for inventory. Do not read `exams/`: past exams are not on the table for this exam.**

- All 27 lecture pages: `wiki/lectures/L01-*.md` through `L27-*.md`
- All 12 slide decks: `modules/*/` (one or more `.md` files per module)
- Recommended exercises: `exercises/Exercise2/` through `exercises/Exercise11/` (each folder contains the exercise + solution `.md` files)
- Compulsory exercises: `exercises/compulsory-exercise-1.md` and `exercises/compulsory-exercise-2.md`
- `exam_analysis.md`: useful synthesis but **not canonical**. Source of truth = slides + lectures + exercises. Use as cross-check only.
- `docs/lectures-manifest.md`: which lecture maps to which module

You may use both Read and Bash (e.g. `rg` to grep for terms) freely. Read the lectures top-to-bottom; you can skim slides more strategically.

## Your job: decisions

### 1. Identify atoms

For every named idea the prof treats as a unit in lectures or slides or exercises, decide whether it should be its own atom. Apply the granularity rule:

> An atom = one named idea Anders would naturally ask Claude about as a single question.

Tests:
- "Explain ridge regression" → yes, atom
- "Explain L2 norm" → no (lives inside ridge)
- "Explain regularization techniques" → no (too coarse, that's a MOC's responsibility)
- "Walk me through hierarchical clustering by hand" → yes (procedural but standalone-askable, becomes its own atom or a major section in `hierarchical-clustering.md`)
- "What's the difference between LDA and QDA" → no (Claude composes from `lda` + `qda` atoms)

When unsure, lean toward fewer atoms. Two atoms that should have been one is worse than one atom that's slightly broad.

### 2. Assign ownership

Most atoms belong to exactly one module, the module where the prof treats the concept as a topic in its own right (not just where he first mentioned it in passing).

But some concepts genuinely span multiple modules with substantial treatment in each, and have no natural single "home." These go into the **Specials** section of the manifest instead of any module. They are not owned by any module; downstream they get their own dedicated fan-out agent.

The test for Specials: *can you comfortably say "this concept belongs to module X"?* If yes → assign it to that module. If no → Specials.

Likely Specials candidates (use judgment):
- bias-variance-tradeoff: explicitly the prof's "course-running theme"
- regularization: central concept appearing in modules 6, 7, 11
- cross-validation: mechanics in module 5, but used everywhere downstream
- standardization: required by ridge, PCA, k-means, KNN, NN
- overfitting: recurring frame across the course
- maybe double-descent, MLE, others: your call

Likely *not* Specials: ridge regression (clearly module 6), LDA (clearly module 4), random forest (clearly module 9). They touch ideas across modules but the prof introduced them as a topic in one module and stayed there.

For each Specials atom, list **every module** it appears in (in the `modules:` plural field) and **every lecture** it touches. Other modules' MOCs and atoms will wikilink to it.

### 3. Apply the scope rule

Per the prof's rule (Apr 28, L27): "If it was covered in the slides or the exercises, it's fair game. If it's only in the book and we didn't talk about it in class or exercises, it won't be on the test."

For every candidate concept:
- Covered in slides OR lectures OR exercises → IN scope, gets an atom
- Only in book → OUT of scope, no atom; goes into the per-module Out-of-Scope section instead

Cross-check against `exam_analysis.md` §5 (it lists explicit exclusions like SVM, F-tests, AIC/BIC algebra, multi-class logistic, etc.), but treat lectures + slides as the source of truth. If exam_analysis says X is out but the prof actually covered X in lecture, X is in.

### 4. Map exercises to atoms

For every problem in `exercises/Exercise2/` through `exercises/Exercise11/` and in the two compulsory exercises, identify which atom(s) it tests. Add the exercise to the atom's `exercises:` list with a one-line description.

If a problem doesn't fit any candidate atom, that's a signal: either the atom is missing (add it) or the problem is out of scope (note it). Don't silently drop unmapped problems.

ISLP end-of-chapter exercises are NOT in scope. Don't include them.

(ISLP section pointers are *not* your job. The fan-out agents will fill `isl-ref:` per atom when they read the relevant book chapter to flesh out treatment.)

## Output spec

Write to `docs/concepts-manifest.md`. Structure:

```markdown
# Concepts manifest

Deterministic list of concept atoms for the TMA4268 wiki. Source of truth for the concepts fan-out pass — every downstream atom-writing agent reads its module slice from here. Slugs, ownership, and cross-cutting flags are fixed at this layer; do not invent new atoms or rename slugs in the fan-out pass.

Built by reading all 27 lecture pages, 12 slide decks, 13 ISLP chapters (skimmed), the recommended exercises, both compulsory exercises, the three past exams, and `exam_analysis.md` (cross-check, not canonical). The `notes/` folder was not read (off-limits).

## Atoms by owning module

### Module 01 — Introduction (`01-intro`)

```yaml
- slug: <hyphen-case>
  canonical: <Capitalized Name>
  lectures: [L01]
  isl-ref: null            # default null; fan-out agents fill this when they read the book chapter
  exercises: []
  one-liner: <one tight sentence>
- slug: ...
```

### Module 02 — Statistical Learning (`02-statlearn`)

```yaml
- slug: parametric-vs-nonparametric
  canonical: Parametric vs nonparametric methods
  lectures: [L02, L03]
  isl-ref: null
  exercises: []
  one-liner: <one tight sentence>
```

[... continue per module 03 through 12]

## Specials (cross-cutting atoms with no single owning module)

These are concepts that genuinely span modules and have no natural home. The fan-out's 13th agent (the specials agent) writes these. Each entry uses the **plural `modules:`** field to signal "specials" (regular atoms use singular `module:`).

```yaml
- slug: bias-variance-tradeoff
  canonical: Bias-variance tradeoff
  modules: [02-statlearn, 06-modelsel, 09-boosting, 11-nnet]
  lectures: [L02, L03, L04, L13, L20, L26]
  isl-ref: null
  exercises:
    - Exercise2.5 — bias-variance simulation on polynomial regression
    - CE1 problem 2 — apply bias-variance reasoning to KNN
  one-liner: course-running theme; derive E[(y−f̂)²] = Bias² + Var + σ²; flag double-descent
- slug: regularization
  canonical: Regularization
  modules: [06-modelsel, 07-beyondlinear, 11-nnet]
  lectures: [L13, L14, L16, L24]
  isl-ref: null
  exercises: [...]
  one-liner: ...
```

## Out of scope (per module)

### Module 06 — Model Selection
- **AIC / BIC / Cp derivations.** Prof said in L12 "not exam material"; only the conceptual "penalize complexity" claim is in scope. No atom for the algebra.
- **Bayesian interpretation of ridge & lasso.** Mentioned in L14 with verbatim "I really don't think I'd put this on the test." No atom.
- ...

### Module 09 — Tree Boosting (...)
- **SVM (entire ISLP chapter 9).** Prof skipped; L27 confirmed "fully out of scope." MOC notes the absence; no atoms for hyperplanes / margins / kernels / etc.

[... continue per module]

## Notes on edge cases

[Section for any judgment calls you made — e.g. "treated double-descent as its own atom rather than a section of bias-variance because it returned across L04, L20, L26 with distinctive treatment" — so future readers (and Anders) know what you decided and why.]
```

## Conventions

- Slugs are hyphen-case, lowercase, no underscores: `ridge-regression`, `bias-variance-tradeoff`, `roc-curve`
- Slugs name the canonical concept, not a phrasing variant: `principal-component-analysis` not `pca` (pca is an alias the atom can declare)
- For cross-cutting atoms, list lectures in chronological order
- For exercise references, use the format `Exercise<N>.<sub>` for recommended exercises and `CE<N> problem <P>` for compulsory exercises

## Self-checks before submitting

1. **Coverage sweep**: did every problem in `exercises/Exercise2-11/` and both compulsory exercises map to some atom (or get logged as out-of-scope)? Run `rg` over the exercise files for technique names; verify each appears.
2. **Specials completeness**: for each Specials atom, did you list every module and lecture that touches it? Spot-check by grepping `wiki/lectures/` for the slug name. And: is each Specials atom *truly* cross-module, could it not be cleanly assigned to one module?
3. **No fragmentation**: if you have `ridge-regression`, you should NOT also have `l2-regularization` or `tikhonov-regularization` as separate atoms. Pick one canonical slug and the others become aliases.
4. **No book-only atoms**: every atom must trace to a slide, lecture, or exercise. Grep your slug list against the prof's primary sources.
5. **No SVM atoms**, no survival, no Bonferroni / FDR, no F-test mechanics, no AIC/BIC formulas, no multi-class logistic regression, no Adam internals, these are all confirmed out of scope per L27 and the prof's earlier statements.

## What you're not doing

- Not writing the atoms themselves, that's the fan-out pass
- Not declaring importance tiers
- Not reading `notes/`
- Not inventing concepts that aren't in the prof's materials
- Not preserving every term the prof said in passing, only those that pass the granularity test

## Deliverable

A single file: `docs/concepts-manifest.md`, formatted as above. Estimated atom count: 40–70 (plus ~5–10 Specials). Estimated total length: 500–800 lines (mostly tables and yaml blocks).

Also briefly report at the end of your run:
- Total atom count (per-module + Specials)
- Specials atom count and which they are
- How many exercise problems you mapped vs how many were unmappable
- Any judgment calls worth flagging for Anders to review before fan-out

Trust your judgment. You're the gating decision-maker for the entire wiki.
