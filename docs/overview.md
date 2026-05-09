# Overview

This is the context bank for TMA4268 Statistisk læring (2026 vår), final exam 2026-05-18. The repo serves one purpose: produce a context layer that an LLM can query to act as an expert tutor calibrated to *this* prof's *this* exam.

## North Star

When an LLM is asked a question about this course, it should be able to load **one or two small files** from `wiki/` and have everything it needs to answer well, including what the prof actually emphasized, the worked examples he used, the traps he flagged, and what's out of scope.

This is the post-RAG, agentic-search-friendly form of a personal knowledge base. We're optimizing for **token-efficient context loading at query time**, not vector search, not browse-friendly hyperlinking, not (primarily) human reading.

## The scope rule

The prof was explicit on Apr 28:

> "If it was covered in the slides or the exercises, it's fair game. If it's only in the book and we didn't talk about it in class or exercises, it won't be on the test."

Strengthened by his own emphasis: **"especially the exercises."**

[[scope]] is the canonical authority for "is X in scope?". It spells out the source hierarchy (exercises > lectures > slides; ISLR for fleshing out in-scope ideas, not for scoping), the explicit out-of-scope list with verbatim anchors, the programming policy, the 2026 question patterns, and the past-exam translation rules. Atoms apply this rule: include what the prof covered, note where the textbook adds material he didn't, don't atomize anything that lives only in the book.

## The corpus

| Source | What it gives |
|---|---|
| `transcripts/` | What the prof actually said and emphasized, the **signal** |
| `modules/` | Slides, the **structure** of the curriculum |
| `exercises/` | Recommended + compulsory exercises, the **form** of exam-style problems |
| `exams/` | Past papers (2023, 2024, 2025), the **historical question patterns** |
| `book/` | ISLR as markdown, one file per chapter (slugs match module slugs). Content reference; available at the exam. Used for citation and look-up, not memorization. |

Each source has a distinct role. We mine each for what only it gives.

## Layers

### Bronze: immutable

The raw sources above, plus `archive/` (old materials) and `exam_analysis.md` (prior synthesis of exam logistics + scope from a previous transcript pass, useful as scaffolding, not canonical; superseded for scope by `docs/scope.md`).

Nothing modifies bronze.

### Off-limits: `notes/`

The `notes/` folder contains Anders's own private notes. **Claude does not read, reference, or modify this folder.** It is invisible to all agents and to the tutor. No file under `notes/` should appear in any wiki output, agent prompt, or tutor response.

### Wiki: LLM-generated

`wiki/` holds three kinds of file, each described in its own doc:

- **Lectures** ([[lectures]]): one compressed transcript per lecture. Filler stripped, all signal preserved.
- **Concepts** ([[concepts]]): one self-contained expert briefing per named idea.
- **MOCs** ([[mocs]]): one router per curriculum module.

The user edits wiki freely. Agents regenerate as instructed.

### Docs: this folder

`docs/` describes the system. `docs/prompts/` holds the briefs given to agents that build the wiki. The docs are themselves text files we instruct ad-hoc, not strict specs.

## How agents use this

1. Land on `CLAUDE.md` → read [[overview]] for orientation.
2. Route via [[../exam_analysis]] (logistics, scope) and the relevant MOC.
3. Load 1–2 concept atoms.
4. Drop into the relevant lecture(s) only when verbatim prof voice is needed.
5. Open bronze only to verify a quote.

## How the human uses this

Obsidian as the interface. Open `wiki/README.md`, navigate via MOCs and the graph view. Build the A5 cheat-sheet by reading concepts.

## Typical user journey

The human picks a module. From there:

1. "What's module 6 about?" → agent loads the MOC, gives a routed overview.
2. "What's on the curriculum for module 6?" → agent reads MOC + linked lectures + concepts, summarizes what the prof actually covered and emphasized, flags what's only in the book.
3. "What's important vs. not?" → derived from signals; agent answers from atoms and lectures.
4. "Write me a primer for module 6" → query-time synthesis from MOC + concepts + lectures. Not pre-baked.
5. "Quiz me / give me an exercise" → bronze exercises + agent generation.
6. "Clarify X" → agent loads the relevant atom, answers using the prof's framing.

The wiki doesn't pre-bake primers, quizzes, or summaries. Those are query-time outputs. The wiki holds the *raw material in compressed form*: lectures, concepts, MOCs, each capturing how *this* prof teaches and what *this* exam tests.

## What this is not

- Not a textbook. ISLR is the textbook.
- Not a strict spec. Conventions are loose; agents are smart; we instruct ad-hoc.
- Not a tier-classified knowledge graph. Importance lives in the verbatim signals an atom cites, not in metadata.
