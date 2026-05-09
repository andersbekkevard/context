# Concepts pass — agent brief

You're producing self-contained expert briefings, one per named concept, for the TMA4268 exam-prep context bank. Read [[../overview]] and [[../concepts]] first.

## North Star

When an LLM loads one of your atoms, it should be able to answer any reasonable exam question about that concept without traversing other files.

## What to do

Read `wiki/lectures/*.md` and `modules/*/*.md`. Identify atomic concepts: named ideas, slide-title-sized — e.g. "ridge regression," "bias-variance tradeoff," "ROC curve." Skip out-of-scope material (`exam_analysis.md` §5).

For each, write `wiki/concepts/{hyphen-case-name}.md` as a **briefing for a knowledgeable peer**. Imagine handing it to another stat learning professor who needs to step in and teach *this* exam. They already know the topic in the abstract; they don't know what's on this curriculum, the definitions this prof uses, his framings, his emphasis, his insights, the traps he flagged, or how it might appear on the exam.

That gap is the atom's job to fill.

Each atom typically contains:

- The prof's definition (verbatim or near-verbatim from slides / lecture)
- His framing and notation (the symbols and setup he uses)
- Formulas and conventions worth memorizing — board equations, identities, the calculations he stressed; flag the few he explicitly said to know cold
- His insights and mental models — verbatim where wording carries
- What he flagged as exam-relevant — verbatim quotes from lectures, with date+timestamp
- Pitfalls he flagged (inline, not separate files)
- Scope notes — applying the prof's rule: slides or exercises → fair game; book-only → out
- How it might appear on the exam (drawing on past-exam patterns + his stated preferences)
- Out-links to related concepts and lectures

Dense prose. Skip what a peer already knows from textbooks; the atom's value is everything specific to *this* course's treatment.

## Active expert-eye search (do this before finalizing)

Don't just passively transcribe what's in the lectures. **Think like an ISLP-trained stat learning expert** about the concept you're writing up:

1. List what an expert would expect to see covered for this concept — standard formulas, conventions, derivations, edge cases, computational considerations, comparisons, identities, notation conventions. (For ridge regression: standardization, why intercept isn't penalized, closed-form cost, condition-number motivation, MAP interpretation, ridge trace, λ via CV, contrast with lasso. For backprop: chain rule, computational graph view, vanishing/exploding gradients, vectorized form, mini-batch gradient. Etc.)

2. For each item, **search lectures and slides** to see whether the prof addressed it.

3. If yes → include it, sourced to the right lecture or slide ref.

4. If no → either omit (if peripheral) or note explicitly as "not covered in this course" so the user knows to skip it in ISLR.

This catches things a passive read would miss — moments where the prof mentioned something briefly but it matters. The agent's job isn't just to summarize the lectures; it's to verify coverage of the standard expert checklist for the concept.

## Granularity

One atom = one named idea. If you'd put it on a flashcard, atom; if it's the topic of a deck, MOC.

Pitfalls and derivations live inside the relevant concept atom, not as separate files. If an atom is getting unwieldy, the concept is probably actually two — split.

## Format

- Filename hyphen-case (e.g. `ridge-regression.md`)
- Light frontmatter (name, module, related, source lecture dates)
- Wikilinks to related concepts and lectures
- Math in LaTeX (`$...$`, `$$...$$`)

## What you're not doing

- Not declaring importance tiers — importance is visible in the signals you cite
- Not creating atoms for out-of-scope material
- Not paraphrasing prof signals — verbatim if it carries
- Not enforcing word counts

Trust your judgment on granularity.
