---
concept: auth-middleware
tags: [auth, security, jwt, gap]
related: [compliance-gaps, api-routes]
files: [backend/app/main.py, backend/app/core/config.py, backend/app/api/medical_endpoints.py]
last-updated: 2026-05-25
---

# Auth Middleware

## What exists

- `python-jose` (JWT encode/decode) and `passlib`/`bcrypt` are declared in `requirements.txt`.
- `config.py` has `SECRET_KEY` and related env-var settings.
- CORS middleware (`allow_origins=["*"]`) and TrustedHost middleware (`allowed_hosts=["*"]`) are applied in `main.py`.

## Key invariants

- **Auth is NOT enforced on medical routes.** `medical_endpoints.py` has zero `Depends(get_current_user)` or equivalent guards on any of its POST handlers.
- CORS wildcard means any origin can call any endpoint — including `/api/v1/medical/*`.
- TrustedHost wildcard (`*`) provides no actual host restriction.

## Known gaps

- No `get_current_user` dependency injected into medical route handlers.
- No role-based access control (clinician vs patient vs anonymous).
- PHI/PII can be submitted by unauthenticated callers with no logging of who made the request.
- Auth scaffolding exists at library level but is wired to zero routes in the audited codebase.

## What would be needed

Inject a FastAPI `Depends(verify_token)` on every medical router; restrict CORS to known origins; implement audit logging per authenticated user per request.

## See also

- `compliance-gaps` — HIPAA implications
- `api-routes` — which routes are exposed without auth
