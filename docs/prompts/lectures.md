# Lectures pass — agent brief

You're compressing one lecture transcript for the TMA4268 exam-prep context bank. Read [[../overview]] and [[../lectures]] first.

## North Star

An LLM should be able to read your output and have nearly the same understanding of what mattered in this lecture as it would from the raw transcript — at a fraction of the tokens.

## What to do

Read `transcripts/{date}.txt`. Output a markdown file at `wiki/lectures/L{NN}-{slug}.md` — see Output below.

Strip:

- Filler, false starts, "ums," repetition
- Off-topic asides and banter
- Course admin (deadlines, room changes, scheduling chatter)
- Long-winded phrasings — tighten them

Preserve:

- Every factual claim the prof made (verbatim or near-verbatim where wording is precise)
- Every example, derivation sketch, mental model, intuition
- Every signal flagging exam relevance, importance, or common student mistakes — verbatim where the wording carries weight
- The order of presentation (matters for what builds on what)

The output reads like a lecture, just denser. Roughly a third of the bronze length is a reasonable rule of thumb — not a target.

## Output

Write a markdown (`.md`) file at `wiki/lectures/L{NN}-{slug}.md` — sequential lecture number 01..27 (cross-reference `course-information.md` to determine N) plus a slug from the lecture title. E.g. `L04-linreg-1.md`.

Follow the canonical shape in [[../templates/lectures]]:

- Frontmatter exactly as in the template (lecture, date, module, title, topics, tags, aliases)
- `# L<NN> — <Title>` (H1, one only)
- Immediately after H1: a 1–3 sentence prose summary of the lecture — what was covered, where it went, what to flag at a glance
- `## Key takeaways` with 3–6 bullets capturing what mattered most
- Then the body, with `## ` headings marking each major topic the prof transitioned into, `### ` for sub-segments. **Never H4.**

Inside the body:
- Wikilink concept names when they come up: `[[ridge-regression]]`
- Cite slide refs and ISL sections inline when the prof points to them
- Use Obsidian callouts when they fit naturally (e.g. for a verbatim must-know quote); don't impose a callout taxonomy

## What you're not doing

- Not imposing template sections like "Outline / Claims / Q&A" — your H2 headings reflect the prof's actual topic transitions, not a fixed shape
- Not paraphrasing prof signals — verbatim if it carries weight
- Not adding content not in the transcript
- Not classifying signals into tiers or strengths
- Not enforcing a target word count or bullet count

Trust your judgment on filler-vs-signal. You're a smart reader.
