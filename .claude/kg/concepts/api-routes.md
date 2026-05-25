---
concept: api-routes
tags: [api, routes, fastapi, rest, websocket]
related: [medical-ai-engine, auth-middleware, scaffold-origins]
files: [backend/app/main.py, backend/app/api/endpoints.py, backend/app/api/medical_endpoints.py, backend/app/api/knowledge_endpoints.py]
last-updated: 2026-05-25
---

# API Routes

## Core routes (main.py)

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Platform info (returns NeuralVerse/MindForge scaffold strings) |
| GET | `/health` | Service health + model count (static strings) |
| GET | `/status` | Detailed metrics (static strings, not live) |
| WS | `/ws/{client_id}` | Real-time NLP / prediction via ai_engine_mock |

## Medical routes (medical_endpoints.py — /api/v1/medical/*)

| Method | Path | Backed by |
|---|---|---|
| POST | `/symptom-analysis` | SimplifiedMedicalAI |
| POST | `/drug-interactions` | SimplifiedMedicalAI |
| POST | `/outcome-prediction` | SimplifiedMedicalAI |
| POST | `/clinical-decision-support` | SimplifiedMedicalAI |

**None of these routes require authentication.**

## AI / framework routes (endpoints.py — /api/v1/*)

| Prefix | Purpose | Status |
|---|---|---|
| `/crew/*` | CrewAI crew instantiation | Mock if no API key |
| `/index/*` | LlamaIndex knowledge query | Mock if no API key |
| `/nlp/*` | NLP tasks | Mock engine |
| `/generate/*` | Text / image generation | Mock engine |
| `/vision/*` | Vision tasks | Mock engine |
| `/audio/*` | Audio transcription | Mock engine |

## Knowledge routes (knowledge_endpoints.py — /api/v1/knowledge/*)

Not fully audited. NetworkX / neo4j driver declared in requirements; no live DB wired by default.

## Key invariant

`/docs` (OpenAPI UI) is enabled in all environments — no auth on the spec itself.

## See also

- `medical-ai-engine` — what backs the medical routes
- `auth-middleware` — why all routes above are unauthenticated
