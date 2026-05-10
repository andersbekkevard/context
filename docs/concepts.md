# Concept atoms

A concept atom is a **self-contained expert briefing on one named idea**. One per concept, in `wiki/concepts/`.

## What it is

The atom is **Claude's raw material when tutoring Anders.** When Anders asks Claude "explain ridge regression like the prof teaches it," Claude loads the ridge atom and answers from it.

Imagine handing the atom to **another stat learning professor** who needs to brief themselves on how *this* prof teaches one named idea. The audience is a knowledgeable peer, they already know what ridge regression is, what backpropagation does, what cross-validation means in the abstract. What they don't know:

- What's on *this* curriculum
- The exact definitions *this* prof uses
- Where *he* lays the emphasis
- The insights and framings *he* brought up
- The traps *he* flagged
- Which exercises drill this concept
- How the topic might appear on *this* exam

That gap is what the atom fills.

After loading the atom, Claude should be able to answer a reasonable exam question about the concept the way *this* exam would ask it, without traversing other files.

## Division of labor between wiki and book

The atom is **not a textbook recreation.** It captures:

- **What's in scope** (from slides + lectures + exercises, the prof's curriculum)
- **The prof-specific framing** (definitions he uses, traps he flagged, his emphasis, his insights)

The book (`wiki/book/`) holds the **deep treatment** of any in-scope idea. The atom carries an `isl-ref:` pointer so Claude can tell Anders "for the full derivation, see ISLP §6.2.1." Atoms intentionally don't recreate textbook content, as Claude already knows it from training, and the book is on the exam table for lookup.

## Standard contents

A typical atom contains (per `docs/templates/concepts.md`):

- **Definition**: the prof's framing, verbatim or near-verbatim from slides/lecture
- **Notation & setup**: symbols and conventions the prof uses
- **Formulas to know cold**: equations the prof put on the board or stressed; flag the few he explicitly said to memorize
- **Insights & mental models**: the prof's angles, verbatim where wording carries
- **Exam signals**: verbatim prof quotes flagging exam relevance, anchored to the lecture wikilink
- **Pitfalls**: inline, verbatim where prof flagged them
- **Scope vs ISLP**: what's IN, ISLP pointer for full treatment, what's in the book but the prof skipped (so Anders knows what to skip in his ISLP look-ups)
- **Exercise instances**: one line per exercise problem touching this concept (`exercises/Exercise<N>/` plus the two compulsory exercises). ISLP end-of-chapter exercises are NOT in scope and are not referenced.
- **Exam-appearance patterns**: how it might show up on the exam, drawing on past papers and prof's preferences
- **Related**: wikilinks to related concept atoms

Dense prose. Skip what a peer already knows from textbooks; the atom's value is everything specific to *this* course's treatment.

## Atom depth scales with prof's treatment

There is no external tier ranking. The agent reads the lectures + slides for a concept and lets the prof's actual treatment dictate depth:

- Heavy treatment + multiple exam-flag quotes + recurring across lectures → richly developed atom
- Standard treatment, one solid lecture pass → standard atom
- Brief mention or "you can read about this in the book" → stub

Length bound: 80–250 lines (cross-cutting atoms may exceed). Importance is visible in the depth and signal count, not declared in metadata.

## Granularity

One atom = **one named idea Anders would naturally ask Claude about as a single question.**

- "Ridge regression": yes
- "L2 norm": no, lives inside ridge
- "Regularization techniques": no, too coarse (MOC scope)
- "Walk me through hierarchical clustering by hand": yes, procedural but standalone-askable
- "Difference between LDA and QDA": no, Claude composes from `lda.md` + `qda.md`

Pitfalls and derivations live **inside** the relevant concept atom, not as separate files. If an atom is getting unwieldy beyond 250 lines, the concept is probably actually two. Split.

The `docs/concepts-manifest.md` file (produced by the inventory pass) is the authoritative slug list. Atom-writing agents work from that manifest, not from their own granularity calls.

## Cross-cutting concepts

Some concepts appear in multiple modules with significant treatment in each (bias-variance, regularization, cross-validation, standardization). These live as **one global atom**, owned by the first-introducing module:

- The atom captures the prof's treatment across all modules
- It includes a `## Returns in other modules` section linking to lectures where the concept reappears
- Other modules' MOCs and atoms wikilink back via `[[bias-variance]]` etc., and bidirectional linking makes the cross-cutting structure visible in the Obsidian graph

This means Claude loads ONE file when tutoring on bias-variance and gets the complete picture, including how the prof escalated his treatment as the course progressed.

## Where it lives

`wiki/concepts/{hyphen-case-slug}.md`, slug from `docs/concepts-manifest.md`. Never invented.

## What concept atoms are not

- Not for out-of-scope material. The scope rule (slides + lectures + exercises = in; book-only = out) is canonical and derived from the prof's primary materials, not from `exam_analysis.md` (which is useful synthesis but not canonical).
- Not declared by importance tier. Depth and signal-count carry the importance signal.
- Not browsing surfaces. The MOC is the browsing surface.
- Not textbook recreations. The book holds the deep treatment; atoms point to it via `isl-ref:`.
