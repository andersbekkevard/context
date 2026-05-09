# TMA4268 — exam-prep context bank

Personal exam-prep system for **TMA4268 *Statistisk læring*** (NTNU, final exam 2026-05-18). The repo holds the raw course corpus alongside an LLM-generated wiki; together they form a context layer an LLM tutor can query, and a Quartz-rendered site friends can browse.

## Layout

| Folder | Contents |
|---|---|
| `CLAUDE.md` | Orientation for Claude — read first |
| `docs/` | System docs: scope rules, agent prompts, page templates, manifests |
| `wiki/` | Generated content: lectures (compressed), concepts (atoms), MOCs (routers) |
| `web/` | Quartz config + practice-deck layer (`web/static/decks/`) |
| `transcripts/`, `modules/`, `exercises/`, `exams/`, `book/` | Bronze: immutable course sources |
| `pdfs/` | Original course PDFs served by the rendered site |
| `notes/` | Private notes — **off-limits to all agents** |
| `archive/` | Deprecated material |

## Start here

- **Orientation:** read [CLAUDE.md](CLAUDE.md), then [docs/overview.md](docs/overview.md).
- **Scope questions:** [docs/scope.md](docs/scope.md) is canonical.
- **Browsing content:** open `wiki/` in Obsidian, or visit the rendered Quartz site for the public view.

The wiki is LLM-generated and freely regenerated; bronze is never modified.
