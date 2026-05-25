# Active context — medmind-ai

## Current state

MedMind AI is a FastAPI backend targeting clinical decision support (symptom differential,
drug-interaction checks, outcome risk scores). The "ML engine" is `SimplifiedMedicalAI`
in `real_medical_ai_simplified.py` — a hardcoded condition dict with deterministic NumPy
scoring, not a trained model. The FastAPI title now reads "MedMind AI" following the
2026-05-25 rename sweep that removed all NeuralVerse/MindForge brand residue from 25
files. Status: demo-only — NOT compliant for real patient data.

## What's covered by the KG

- `medical-ai-engine` — what actually serves predictions (hardcoded dict + NumPy)
- `auth-middleware` — JWT scaffolding present but unenforced on medical routes
- `api-routes` — full backend route inventory
- `data-storage` — storage locations and encryption status
- `compliance-gaps` — explicit HIPAA aspiration vs reality gap list
- `scaffold-origins` — NeuralVerse/MindForge rename history and what changed

## Known gaps (be explicit — this is a medical product)

- No trained ML model — predictions come from a hardcoded condition dict + numpy arithmetic
- Auth middleware exists but is NOT enforced on medical routes (no `Depends(...)` guards)
- SQLite plaintext, no encryption at rest, no audit trail
- HIPAA compliance is aspirational — not implemented (§164.312(a)(1), §164.312(b),
  §164.312(e)(2)(ii), §164.308(a)(1) all unaddressed)
- Engine has no "unknown / insufficient data" path — always returns a result
- No medical disclaimer in API responses
- Status: demo-only, do not deploy with real patient data

## In flight

_(none yet — populate on session start)_

## Recent change

- 2026-05-25: NeuralVerse/MindForge brand residue removed (25 files renamed, py_compile clean)

## Pick up here next session

1. (placeholder)
