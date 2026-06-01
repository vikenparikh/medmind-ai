# INDEX — medmind-ai

> One-page context graph. Read this BEFORE reading anything else in this repo.
> Rebuild with: `bash .claude/scripts/build-index.sh > .claude/INDEX.md`
>
> Generated 2026-05-28T22:41Z.

## About

**MedMind AI - Medical Intelligence Platform**

> **Built by Viken Parikh** - AI-powered clinical decision support system with advanced medical knowledge processing

## Top-level layout

| Dir | Purpose |
|-----|---------|
| [backend](backend/) | backend |
| [tests](tests/) | test suite |

## Key files at root

- [README.md](README.md) — MedMind AI - Medical Intelligence Platform
- [ARCHITECTURE.md](ARCHITECTURE.md) — MedMind AI — Architecture
- [docker-compose.yml](docker-compose.yml) — —
- [requirements.txt](requirements.txt) — MedMind AI - Clinical Intelligence Platform

## Code surface

- Python: 27 files
- Shell: 4 files
- Markdown: 16 files

## Memory pointers

- Active focus: [.claude/memory-bank/activeContext.md](.claude/memory-bank/activeContext.md)
- Past sessions: [.claude/memory-bank/progress.md](.claude/memory-bank/progress.md)
- Decisions log: [.claude/memory-bank/decisions.md](.claude/memory-bank/decisions.md)
- Glossary: [.claude/memory-bank/glossary.md](.claude/memory-bank/glossary.md)

## How to run things

- Compose: `docker compose up -d` (see [docker-compose.yml](docker-compose.yml))
- Python deps: [requirements.txt](requirements.txt)

## Claude shared content (symlinked from root)

- Hooks: [.claude/hooks/](.claude/hooks/) (block-dangerous, block-secret-commit)
- Rules: [.claude/rules/](.claude/rules/) (safety, style)
- Skills: [.claude/skills-shared/](.claude/skills-shared/) (graphify, tdd, handoff, …)
- Agents: [.claude/agents-shared/](.claude/agents-shared/)
- Plugins: [.claude/plugins/](.claude/plugins/) — 49 wshobson plugins covering python/frontend/k8s/security/ML/…

Edit the root copies at `~/Downloads/Projects/.claude/` to update every repo at once.
