---
concept: data-storage
tags: [storage, sqlite, postgres, redis, encryption, hipaa]
related: [compliance-gaps, auth-middleware]
files: [backend/app/core/config.py, docker-compose.yml]
last-updated: 2026-05-25
---

# Data Storage

## What is configured

| Store | Config value | Status |
|---|---|---|
| SQLite | `sqlite:///./neuralverse.db` (default `DATABASE_URL`) | Configured; schema not audited |
| PostgreSQL | `DATABASE_URL` env var override | Optional; not provisioned in repo |
| Redis | `redis://localhost:6379` | Optional; declared in config, use not verified |
| ChromaDB | `./chroma_db/` local dir | Optional; no API key required |
| `./models/` | Trained model artifact path | Declared; directory empty / no artifacts |
| `./data/` | Data storage path | Declared; directory empty / no datasets |

## Key invariants

- **Stateless requests.** No conversation history is persisted per request; each call is independent.
- **SQLite filename is `neuralverse.db`** — scaffold naming from the template origin.
- No ORM models or migration scripts are present in the audited codebase; schema is TBD.

## Known gaps (encryption / compliance)

- **No encryption at rest.** SQLite file is plaintext on disk.
- **No field-level encryption.** PHI submitted in request bodies (symptoms, medications, patient context) is not encrypted before any potential persistence.
- **No audit trail.** No write to any audit log when patient data is processed.
- **PII in query params.** Medical endpoint requests pass patient context in JSON body, but there is no scrubbing, masking, or access log that would satisfy HIPAA §164.312.

## See also

- `compliance-gaps` — full gap inventory
- `auth-middleware` — who can submit data (anyone)
