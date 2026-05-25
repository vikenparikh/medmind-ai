# MedMind AI — Knowledge Graph Index

Entry point for the Karpathy-style concept graph. Walk related: links; don't grep the codebase cold.

## Concepts

| Slug | One-line summary |
|---|---|
| [medical-ai-engine](concepts/medical-ai-engine.md) | What actually serves predictions — hardcoded dict + NumPy scoring |
| [auth-middleware](concepts/auth-middleware.md) | JWT scaffolding exists but is NOT enforced on medical routes |
| [api-routes](concepts/api-routes.md) | Full backend route inventory |
| [data-storage](concepts/data-storage.md) | What's stored, where, encryption status (honest) |
| [compliance-gaps](concepts/compliance-gaps.md) | Explicit HIPAA aspiration vs reality gap list |
| [scaffold-origins](concepts/scaffold-origins.md) | NeuralVerse/MindForge naming residue and where it appears |

## Data flow (short form)

```
Client → FastAPI (main.py) → /api/v1/medical/* (medical_endpoints.py)
       → SimplifiedMedicalAI (real_medical_ai_simplified.py)
       → hardcoded dict lookup + NumPy scoring → JSONResponse
```

No trained model is invoked at runtime on any production path.

## Navigation tip

Read one concept file (~2 KB), follow at most two `related:` links, then open source files listed in `files:`. Do not open files speculatively.
