---
concept: compliance-gaps
tags: [hipaa, compliance, security, gaps, phi, pii]
related: [auth-middleware, data-storage, medical-ai-engine]
files: [backend/app/api/medical_endpoints.py, backend/app/main.py, backend/app/core/config.py]
last-updated: 2026-05-25
---

# Compliance Gaps

## What the README / docs claim

- HIPAA compliance
- End-to-end encryption
- Secure patient data handling

## What the codebase actually has

| Claim | Reality |
|---|---|
| HIPAA compliance | Not implemented. No HIPAA controls are present. |
| Encryption at rest | SQLite file is plaintext. No field encryption. |
| End-to-end encryption | HTTPS is transport-layer only (uvicorn config). No application-level encryption. |
| Audit trail for patient data | No audit log on any medical endpoint. |
| Auth on medical routes | Zero authentication enforced on `/api/v1/medical/*`. |
| Input validation | Only JSON parse errors caught. Free-text symptoms are not sanitized. |
| Medical disclaimer | No "not a substitute for professional advice" in API responses. |
| No-refusal path | Engine always returns a result; no "insufficient data" branch. |
| Model versioning | N/A — the "AI" is static dicts. No versioning or drift detection applies. |

## HIPAA rule mapping (gaps)

| HIPAA Rule | Requirement | Gap |
|---|---|---|
| §164.312(a)(1) | Access control | No auth on medical routes |
| §164.312(b) | Audit controls | No audit log for PHI access |
| §164.312(e)(2)(ii) | Encryption at rest | No encryption |
| §164.308(a)(1) | Risk analysis | Not performed or documented |

## Key invariant

This system is appropriate for **internal prototyping and demos only.** It must not handle real patient data or be deployed to clinical users without significant safety, compliance, and clinical validation work.

## See also

- `auth-middleware` — auth scaffolding that exists but is unenforced
- `data-storage` — where data would be stored and encryption status
- `medical-ai-engine` — no-refusal design implications
