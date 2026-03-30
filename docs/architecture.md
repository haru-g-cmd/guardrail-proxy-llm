# Architecture

## Goal
Intercept every LLM-bound prompt, score it for malicious intent and PII exposure, sanitize or block it, and persist a full audit trail before the model sees the text.

## Request Flow

```
Client
  │
  ▼
POST /v1/guardrail/check  ─────────────────────────────────────────────────┐
POST /v1/guardrail/proxy                                                    │
  │                                                                         │
  ▼                                                                         │
GuardrailService.check()                                                    │
  │                                                                         │
  ├─► Redis cache lookup ───────────────► HIT: return cached response       │
  │                                                                         │
  ▼  MISS                                                                   │
PromptAnalyzer.analyze()                                                    │
  ├─► DistilBERTClassifier.score()   (local model or heuristic fallback)   │
  └─► pii.detect() + pii.risk_score()                                      │
         │                                                                  │
         ▼                                                                  │
  Verdict logic (3 tiers)                                                   │
  ├─ maliciousness ≥ threshold  ──► BLOCK                                  │
  ├─ pii_risk ≥ block_threshold ──► BLOCK                                  │
  ├─ pii_risk ≥ sanitize_threshold ► SANITIZE (redacted prompt)            │
  └─ else                        ──► ALLOW                                 │
         │                                                                  │
         ├─► Redis: cache result (5 min TTL)                               │
         ├─► Postgres: write audit record (hash only, no raw prompt)       │
         │                                                                  │
         └─► (proxy endpoint only) DownstreamAdapter.call()               │
               │                                                            │
               ▼                                                            │
         Downstream LLM (OpenAI-compat)◄──────────────────────────────────┘
```

## Ports (all +10 offset)

| Service | Internal | Host |
|---------|----------|------|
| Proxy API | N/A | 8010 |
| Postgres | 5432 | 5442 |
| Redis | 6379 | 6389 |

## Component Map

```
src/guardrail_proxy/
├── main.py                     FastAPI app factory
├── config/
│   └── settings.py             Pydantic-settings config (env + .env file)
├── models/
│   └── contracts.py            Request/response Pydantic models
├── core/
│   ├── classifier.py           DistilBERT scorer + heuristic fallback
│   ├── pii.py                  Regex PII detector, sanitizer, risk scorer
│   ├── analyzer.py             3-tier verdict engine
│   └── service.py              Cache + audit orchestrator
├── api/
│   ├── routes.py               /check /proxy /healthz /statusz
│   └── dependencies.py         FastAPI DI (lru_cache singletons)
├── integrations/
│   ├── langchain_guardrail.py  GuardrailRunnable (LangChain middleware)
│   └── downstream.py          HTTP adapter for downstream LLM
├── storage/
│   ├── database.py             SQLAlchemy engine + session factory
│   ├── entities.py             AuditRecord ORM entity
│   └── cache.py                Redis JSON cache (sha256-keyed)
└── training/
    └── train_distilbert.py     Fine-tune script for local model artifact
```

## Security Design Decisions

| Decision | Rationale |
|----------|-----------|
| Prompts stored as SHA-256 hash only | Prevents audit log from becoming a PII source |
| Storage failures never block screening | Safer to analyse twice than allow unscreened |
| Heuristic fallback uses weighted markers | Proportional to compound adversarial intent; single ambiguous word does not trigger |
| PII block threshold = 0.8 > sanitize threshold = 0.3 | Clear gap prevents threshold confusion under load |
| All ports +10 offset from defaults | Avoids clashes with common local dev stacks |

