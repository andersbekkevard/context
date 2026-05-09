# Concept atoms

A concept atom is a **self-contained expert briefing on one named idea**. One per concept, in `wiki/concepts/`.

## What it is

Imagine handing the atom to **another stat learning professor** who needs to brief themselves on how *this* prof teaches one named idea. The audience is a knowledgeable peer — they already know what ridge regression is, what backpropagation does, what cross-validation means in the abstract. What they don't know:

- What's on *this* curriculum
- The exact definitions *this* prof uses
- Where *he* lays the emphasis
- The insights and framings *he* brought up
- The traps *he* flagged
- How the topic might appear on *this* exam

That gap is what the atom fills.

After loading the atom, an LLM should be able to answer a reasonable exam question about the concept the way *this* exam would ask it — without traversing other files.

A typical atom contains:

- **Definition** — the prof's exact definition, verbatim or near-verbatim from slides or lecture
- **Framing and notation** — how he sets the topic up, the symbols and conventions he uses
- **Formulas and conventions worth memorizing** — the equations and identities he put on the board or stressed in slides; tag the few he explicitly said to know cold
- **Insights and mental models** — angles he brought up that aren't generic; verbatim where wording carries
- **What he emphasized as exam-relevant** — verbatim signals from lectures, with date+timestamp
- **Pitfalls he flagged** — inline
- **Scope notes** — applying the prof's rule: covered in slides or exercises → fair game; book-only → won't be on the test. Note where ISLR treats material he didn't.
- **How it might show up on the exam** — drawing on past-exam patterns and his stated preferences
- Wikilinks to related concepts and the lectures the atom draws from

Dense prose. Skip what a peer already knows from textbooks; the atom's value is everything specific to *this* course's treatment.

## Granularity

One atom = one named idea, slide-title-sized.

- "Ridge regression" — yes
- "Regularization techniques" — too coarse, that's a MOC topic
- "L2 norm" — too fine, lives inside ridge regression

Pitfalls and derivations live **inside** the concept they belong to, not as separate files. A pitfall is a few sentences; a derivation is a code block or short section. We don't shard.

If an atom is getting unwieldy, the concept is probably actually two — split.

## Where it lives

`wiki/concepts/{hyphen-case-name}.md` — e.g. `wiki/concepts/ridge-regression.md`.

## What concept atoms are not

- Not for out-of-scope material (per `exam_analysis.md` §5). If the prof excluded it, we don't atomize it.
- Not declared by importance tier. Importance is visible in the signals the atom cites, not in a frontmatter field.
- Not browsing surfaces. The MOC is the browsing surface.
