---
concept: medical-ai-engine
tags: [ai, medical, scoring, hardcoded]
related: [api-routes, compliance-gaps, scaffold-origins]
files: [backend/app/services/real_medical_ai_simplified.py, backend/app/services/medical_ai_engine.py, backend/app/services/real_medical_ai.py, backend/app/services/ai_engine_mock.py]
last-updated: 2026-05-25
---

# Medical AI Engine

## What it actually does

`SimplifiedMedicalAI` in `real_medical_ai_simplified.py` is the live serving path for all `/api/v1/medical/*` routes. It is:

- A Python class with three `dict` attributes built in `__init__` via private `_initialize_*()` methods.
- The dicts encode conditions (symptoms, risk factors, treatments, severity) and drug interaction pairs — all hardcoded Python literals.
- Scoring is deterministic NumPy arithmetic over those dicts. No trained weights, no embeddings, no external model call.
- Every query returns a result. There is no "unknown" or "insufficient data" path.

## Key invariants

- **Not a trained model.** No weights file, no training loop, no inference call.
- **Deterministic and static.** Same input always returns same output. Adding a condition requires editing the source dict.
- **No hallucination guard.** Unrecognised symptoms are silently matched against whatever the dict contains; the engine never refuses.

## Known gaps (gap to real ML)

- No probabilistic calibration — confidence scores are arithmetic proxies, not calibrated probabilities.
- No drug interaction coverage outside the hardcoded pairs list.
- No ICD-10 / SNOMED grounding — condition names are plain strings.
- `real_medical_ai.py` imports torch/transformers but its load status is unaudited; not on the live request path.
- `medical_ai_engine.py` purpose and relationship to simplified version is TBD.

## See also

- `api-routes` — which endpoints call this class
- `compliance-gaps` — safety implications of the no-refusal design
