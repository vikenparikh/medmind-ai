# MedMind AI — Architecture

## What It Is

MedMind AI is a Python/FastAPI backend that provides AI-assisted clinical decision support via REST and WebSocket APIs. It targets healthcare professionals needing symptom differential, drug-interaction checks, and patient outcome risk scores. The AI layer is **rule-based and statistical** (hardcoded knowledge bases, NumPy/scikit-learn scoring functions); there is no trained neural model or live clinical database in the current codebase. The frontend is not yet present in this repository.

> **Honest note:** The application title strings in source code switch between "MedMind AI", "MedMind AI", and "MedMind AI" — this repo appears to have been scaffolded from a shared template. The medical-specific logic lives in `backend/app/services/real_medical_ai_simplified.py` and `medical_endpoints.py`.

---

## Module Map

| Path | Purpose |
|---|---|
| `start_medmind.py` | Entry point — launches uvicorn via subprocess, opens browser |
| `start_clinical_platform.py` | Alternate startup script (TBD — not audited) |
| `backend/app/main.py` | FastAPI app factory, CORS/host middleware, WebSocket manager, `/health` `/status` `/` routes |
| `backend/app/core/config.py` | Pydantic `Settings` — env vars, DB URLs, AI API keys |
| `backend/app/api/endpoints.py` | Main API router; mounts knowledge + medical sub-routers; general AI routes (models, training, NLP, vision, CrewAI, LlamaIndex, audio) |
| `backend/app/api/medical_endpoints.py` | `/api/v1/medical/*` — symptom analysis, drug interactions, outcome prediction, clinical decision support |
| `backend/app/api/knowledge_endpoints.py` | `/api/v1/knowledge/*` — knowledge graph queries (TBD — not fully audited) |
| `backend/app/services/real_medical_ai_simplified.py` | `SimplifiedMedicalAI` — hardcoded condition/drug knowledge base, NumPy-based risk scoring |
| `backend/app/services/ai_engine_mock.py` | Mock AI engine used at startup; reports initialized models/agents/indices |
| `backend/app/services/ai_engine.py` | Full AI engine (CrewAI, LlamaIndex integrations) — active use TBD |
| `backend/app/services/medical_ai_engine.py` | Alternate medical engine — relationship to simplified version TBD |
| `backend/app/services/real_medical_ai.py` | Heavier version of medical AI (torch/transformers imports) — load status TBD |
| `backend/app/models/schemas.py` | Pydantic request/response schemas |
| `tests/` | pytest + manual test scripts (`test_medmind_production.py`, `test_e2e_medical.py`, etc.) |
| `docker-compose.yml` | Containerized deployment definition |
| `requirements.txt` | Full dependency list |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.8+ |
| **Framework** | FastAPI 0.104, uvicorn |
| **Schemas** | Pydantic v2 |
| **Core AI logic** | NumPy, pandas, scikit-learn, SciPy — statistical scoring |
| **NLP** | spaCy, NLTK, textblob (declared in requirements; actual import path TBD) |
| **Knowledge graphs** | NetworkX, StellarGraph, neo4j driver (declared; no live DB wired by default) |
| **Vector stores** | ChromaDB, FAISS, Pinecone (declared; keys optional) |
| **LLM frameworks** | LangChain 0.0.340, LlamaIndex 0.9, CrewAI 0.1 |
| **LLM providers** | OpenAI, Anthropic, HuggingFace (all key-optional) |
| **ML extras** | XGBoost, PyTorch 2.1, Transformers 4.35 (heavy; not loaded in simplified path) |
| **DB (configured)** | SQLite (default dev), PostgreSQL (prod env var), Redis (caching) |
| **Auth** | python-jose JWT, passlib/bcrypt |
| **Monitoring** | prometheus-client, structlog |
| **Testing** | pytest, pytest-asyncio, httpx |
| **Containerization** | Docker + docker-compose |

---

## Data Flow

```
Client request
    │
    ▼
FastAPI app  (main.py — CORS, TrustedHost middleware)
    │
    ▼
/api/v1 router  (endpoints.py)
    │
    ├── /medical/*  →  medical_endpoints.py
    │       │  Parses JSON query params (symptoms, patient_context, medications)
    │       ▼
    │   SimplifiedMedicalAI  (real_medical_ai_simplified.py)
    │       │  Looks up hardcoded knowledge base dicts
    │       │  Applies NumPy probability/risk scoring
    │       │  Returns differential list + risk score + recommendations
    │       ▼
    │   JSONResponse  →  client
    │
    ├── /knowledge/*  →  knowledge_endpoints.py
    │
    └── /crew/*, /index/*, /nlp/*, /generate/*  →  ai_engine_mock or ai_engine
            (CrewAI / LlamaIndex if keys present; mock stubs otherwise)

WebSocket /ws/{client_id}
    │  Accepts ping/pong + ai_request (nlp | prediction)
    └── ai_engine_mock.process_natural_language / predict_with_model
```

**No request passes through a trained clinical model at runtime.** All medical scoring in the simplified path is deterministic arithmetic over a hardcoded Python dict.

---

## Storage

| Store | What lives there | Status |
|---|---|---|
| SQLite (`medmind.db`) | Default dev DB — schema TBD | Configured, not audited |
| PostgreSQL | Prod patient data (env `DATABASE_URL`) | Optional; not provisioned in repo |
| Redis (`redis://localhost:6379`) | Session cache | Optional; declared, not verified in use |
| ChromaDB (`./chroma_db/`) | Vector embeddings for knowledge queries | Optional; keys not required |
| `./models/` | Trained model artifacts | Directory path declared; no artifacts present |
| `./data/` | Data storage path | Directory path declared; no datasets present |

**No conversation history persistence is implemented** in the current codebase. Each request is stateless.

---

## Safety / Compliance Considerations

**What exists:**
- JWT-based auth scaffolding (python-jose, passlib) — configured but not enforced on medical endpoints in the audited code
- CORS and TrustedHost middleware (currently `allow_origins=["*"]`, `allowed_hosts=["*"]`)
- Structured logging and a `/health` endpoint

**What does not exist (gaps):**
- No disclaimer or "not a substitute for professional medical advice" messaging in API responses
- No hallucination guard — the hardcoded knowledge base silently returns a result for any input; there is no "I don't know" path
- No HIPAA-compliant data handling — PII/PHI passes through query params (not body), no field-level encryption, no audit trail for patient data
- No input validation beyond JSON parse errors — free-text symptom strings are not sanitized
- No model versioning or drift detection (the "AI" is static dicts)
- README claims HIPAA compliance and end-to-end encryption; **neither is implemented in the codebase**

This system is appropriate for internal prototyping / demos. It must not be deployed to real patients without significant safety, compliance, and clinical validation work.

---

## Key Entry Points

| Route | File | Purpose |
|---|---|---|
| `GET /` | `main.py` | Platform info |
| `GET /health` | `main.py` | Service health + model counts |
| `GET /status` | `main.py` | Detailed metrics (static strings) |
| `POST /api/v1/medical/symptom-analysis` | `medical_endpoints.py` | Differential diagnosis |
| `POST /api/v1/medical/drug-interactions` | `medical_endpoints.py` | Drug safety check |
| `POST /api/v1/medical/outcome-prediction` | `medical_endpoints.py` | Patient risk score |
| `POST /api/v1/medical/clinical-decision-support` | `medical_endpoints.py` | Combined CDS |
| `POST /api/v1/crew/create` | `endpoints.py` | Instantiate CrewAI crew |
| `POST /api/v1/index/query` | `endpoints.py` | LlamaIndex knowledge query |
| `WS /ws/{client_id}` | `main.py` | Real-time NLP / prediction |
| `GET /docs` | FastAPI auto | OpenAPI UI |

---

## How to Add a New Feature

1. **Add Pydantic schemas** in `backend/app/models/schemas.py`.
2. **Implement logic** in a new or existing service under `backend/app/services/`. Use `SimplifiedMedicalAI` as the pattern for stateless, sync-wrapped async methods.
3. **Add router** in `backend/app/api/` (or extend `medical_endpoints.py` / `knowledge_endpoints.py`).
4. **Register the router** in `backend/app/api/endpoints.py` via `router.include_router(...)`.
5. **Write a test** in `tests/` using httpx + `pytest-asyncio`.
6. If the feature touches the medical knowledge base, update the dicts in `SimplifiedMedicalAI._initialize_medical_knowledge()` or `_initialize_drug_interactions()`.

---

## Run Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Start server (auto-opens browser)
python start_medmind.py

# Start directly with uvicorn
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

# Docker
docker-compose up -d

# Tests
python -m pytest tests/ -v
python tests/test_medmind_production.py

# Environment variables (required for prod)
export DATABASE_URL="postgresql://user:pass@localhost/medmind"
export REDIS_URL="redis://localhost:6379"
export OPENAI_API_KEY="..."     # optional — mock used if absent
export ANTHROPIC_API_KEY="..."  # optional
```
