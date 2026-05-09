# Lectures pass: agent brief

You're compressing one lecture transcript for the TMA4268 exam-prep context bank. Read [[../overview]] and [[../lectures]] first.

## North Star

An LLM should be able to read your output and have nearly the same understanding of what mattered in this lecture as it would from the raw transcript, at a fraction of the tokens.

## What to do

**Read the entire transcript first, top to bottom, no skipping.** This is non-negotiable. You can't strip filler from what you haven't read, and signals near the end of a lecture often reframe what was said earlier (a closing trap-flag can recast an earlier definition). Partial reads produce broken lecture pages. Use whatever tooling you need to ingest the full file before writing a single word.

**Identify the slide deck the prof walked through.** Inspect the transcript content and match it against the decks in `modules/`. Most lectures map to one deck (e.g. `modules/3LinReg/3LinReg.md`); a wrap-up session that transitions modules may touch two, pick the deck hosting the bulk of the content. The path goes in the `slides:` frontmatter field.

Then compress. Source: `transcripts/{date}.txt` → output: `wiki/lectures/L{NN}-{slug}.md` (see Output below).

Strip:

- Filler, false starts, "ums," repetition
- Off-topic asides and banter
- Course admin (deadlines, room changes, scheduling chatter)
- Long-winded phrasings, tighten them

Preserve:

- Every factual claim the prof made (verbatim or near-verbatim where wording is precise)
- Every example, derivation sketch, mental model, intuition
- Every signal flagging exam relevance, importance, or common student mistakes, verbatim where the wording carries weight
- The order of presentation (matters for what builds on what)

**Use direct quotes from the transcript heavily, especially for emphasis.** When the prof's wording is memorable, pithy, or carries a distinctive framing, even if you could paraphrase it tightly, quote him verbatim. Quotes function as **grep anchors**: Anders (or you in a later session) can copy a phrase, search the bronze transcript, locate it precisely, and read the surrounding context for fuller meaning. A compressed paragraph distilling three minutes of lecture loses that anchor; a quote keeps it. Don't be stingy with quotes, bias toward including them whenever the prof's exact wording is more memorable, more precise, or more characteristically his than your paraphrase would be.

The output reads like a lecture, just denser.

## Output

Write a markdown (`.md`) file at `wiki/lectures/L{NN}-{slug}.md`. Lecture number and slug are deterministic, look them up in [[../lectures-manifest]] for your transcript file (your wrapper prompt should also tell you).

Follow the canonical shape in [[../templates/lectures]]:

- Frontmatter exactly as in the template (lecture, date, module, title, slides, topics, tags, aliases)
- `# L<NN> — <Title>` (H1, one only)
- Immediately after H1: a 1–3 sentence prose summary of the lecture, what was covered, where it went, what to flag at a glance
- `## Key takeaways` with 3–6 bullets capturing what mattered most
- Then the body, with `## ` headings marking each major topic the prof transitioned into, `### ` for sub-segments. **Never H4.**

**Length: 250–500 lines.** Below 250 means you stripped signal; above 500 means filler leaked through. If you finish outside this range, re-evaluate which way you're erring.

Inside the body:
- Wikilink concept names when they come up: `[[ridge-regression]]`
- Cite slide refs and ISL sections inline when the prof points to them
- Use Obsidian callouts when they fit naturally (e.g. for a verbatim must-know quote); don't impose a callout taxonomy

## What you're not doing

- Not imposing template sections like "Outline / Claims / Q&A", your H2 headings reflect the prof's actual topic transitions, not a fixed shape
- Not paraphrasing prof signals, verbatim if it carries weight
- Not adding content not in the transcript
- Not classifying signals into tiers or strengths
- Not enforcing a target bullet count (the 250–500 line range above is the only length bound)

Trust your judgment on filler-vs-signal. You're a smart reader.
