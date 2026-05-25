---
concept: scaffold-origins
tags: [scaffold, naming, rename, history]
related: [api-routes, medical-ai-engine]
files: [backend/app/main.py, backend/app/api/endpoints.py, backend/app/services/ai_engine_mock.py, backend/app/core/config.py, setup.py]
last-updated: 2026-05-25
---

# Scaffold Origins

## What happened

This repo was bootstrapped from a shared multi-product template. The template carried two prior product names — **NeuralVerse AI** and **MindForge AI**. Both have been renamed to **MedMind AI** as of 2026-05-25.

## What got renamed

| Layer | Old | New |
| --- | --- | --- |
| Class names | `NeuralVerseAIMockEngine` | `MedMindAIMockEngine` |
| FastAPI title | `MindForge AI` | `MedMind AI` |
| Module docstrings | `NeuralVerse AI - ...` | `MedMind AI - ...` |
| Storage paths | `neuralverse.db`, `mongodb://.../neuralverse` | `medmind.db`, `mongodb://.../medmind` |
| URLs / hosts | `api.neuralverse.ai` (mock URL) | `api.medmind.ai` |
| Secret key default | `neuralverse-ai-secret-key-2024` | `medmind-ai-secret-key-2024` (still env-overridable in prod) |
| Test file | `tests/test_neuralverse_ai.py` | `tests/test_medmind_ai.py` |

## What you'll still see

The `.claude/kg/` concept files (this one, `index.md`, `api-routes.md`) intentionally retain references to the old names as a record of the rename. They are NOT residue — they document the history.

## Why this matters

The FastAPI app title shows up in `/docs` and the `/` info route. Auditors or new engineers opening those URLs now see "MedMind AI" — matching the repo, README, and ARCHITECTURE.md. No more brand confusion.

## See also

- `api-routes` — `/docs` OpenAPI surface that reads from `app.title`
- `medical-ai-engine` — the renamed `MedMindAIMockEngine` class
- `data-storage` — the storage-path renames invalidated any pre-existing dev DBs
