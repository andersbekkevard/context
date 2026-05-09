# Lectures

A lecture page is a **compressed lecture transcript**. One per class session, in `wiki/lectures/`.

## What it is

The same lecture, in a fraction of the tokens. Filler, repetition, banter, admin, and long-winded phrasings stripped. Every claim, example, intuition, idea, and signal preserved. Still reads like a lecture — flowing prose, just denser.

The order of presentation matters and stays. The prof's flow shows what builds on what.

Verbatim quotes for things that carry weight: signals (exam relevance flags, common-mistake warnings), the prof's mental models, his exact wording where it matters. Outside those, tighter prose is fine.

## Why lectures exist

Bronze transcripts are too long to load into LLM context efficiently. Lecture pages let an agent load one class session's worth of signal at roughly a third of the tokens. The agent gets the prof's voice and emphasis without the noise.

## Shape

Each lecture follows the canonical template at [[templates/lectures]]:

- Frontmatter (lecture number, date, module, title, topics, tags, aliases)
- H1 lecture title
- A 1–3 sentence prose summary right after the H1
- `## Key takeaways` — 3–6 bullets
- Body, with H2 headings marking the prof's actual topic transitions and H3 for sub-segments. Never H4.

The headings reflect the lecture's natural shape, not an imposed template.

## Where it lives

`wiki/lectures/L{NN}-{slug}.md` — e.g. `wiki/lectures/L04-linreg-1.md`. Sequential lecture number 01..27 plus a slug from the title.

## Linking

When concepts come up, wikilink them: `[[ridge-regression]]`. Cite slide and ISL refs inline when the prof points to them.

## What lectures are not

- Not structured artifacts with imposed sections like "Outline / Claims / Pitfalls / Q&A." The body's H2 headings mirror the prof's actual topic transitions; we don't flatten lectures into a template.
- Not summaries. The summary at the top is metadata for routing; the body still contains the lecture, denser.
- Not transformations. The lecture is still recognizable as itself, just denser.
