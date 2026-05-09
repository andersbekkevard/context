# TMA4268 — exam-prep context bank

Final exam: **2026-05-18**. Open-book, 4h.

This repo is a study-assistant context bank. **Primary reader: LLMs.** The wiki is engineered so an agent can load one or two small files and instantly be an expert on what's actually relevant for *this* prof's *this* exam. Human navigation (Obsidian) is a side benefit.

## Read first

- [[docs/overview]] — North Star, layers, how it ties together
- [[exam_analysis]] — prior synthesis: logistics, scope, past-exam rules. Reference, not canon.
- [[course-information]] — schedule + curriculum

## Page kinds

- [[docs/lectures]] — compressed lecture transcripts
- [[docs/concepts]] — atomic concept briefings
- [[docs/mocs]] — module routers

## Agent prompts

- [[docs/prompts/lectures]]
- [[docs/prompts/concepts]]
- [[docs/prompts/mocs]]

## Hard invariants

- Bronze (`modules/`, `transcripts/`, `exercises/`, `exams/`, `archive/`) is immutable.
- `wiki/` is LLM-generated; user edits welcome.
- Verbatim quotes for prof signals — never paraphrase.
- Importance is derived from signals, not declared. No tier system inside atoms.
- Out-of-scope material (per `exam_analysis.md` §5) gets no atom.
