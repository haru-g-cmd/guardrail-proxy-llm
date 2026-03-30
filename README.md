# Guardrail Proxy

A prompt screening proxy that sits between the application and the LLM. You send a prompt to `/v1/guardrail/check`, and the service scores it for malicious intent and PII exposure and returns one of three verdicts: allow (prompt clears all thresholds and passes through unchanged), sanitize (PII spans are redacted in place and the cleaned text is forwarded), or block (request is rejected and metadata pushed to a quarantine queue). A second endpoint, `/v1/guardrail/proxy`, does the same screening and then calls the downstream LLM if not blocked, returning both the verdict and the model response in one payload. Per-tenant threshold overrides are accepted on every request via three headers (`X-Maliciousness-Threshold`, `X-PII-Block-Threshold`, `X-PII-Sanitize-Threshold`), so different callers can operate at different sensitivity levels without separate deployments.

The backend runs on FastAPI. Maliciousness scoring uses a fine-tuned DistilBERT classifier that returns a 0 to 1 score; if no trained artifact is found at the configured path, a weighted heuristic fallback runs instead, scanning for jailbreak markers, override phrases, and role confusion patterns and accumulating a score proportional to compound adversarial intent so a single ambiguous word does not trigger a block. PII detection is regex based and covers credit card numbers, SSNs, email addresses, phone numbers, API keys, and IP addresses; each match carries a per-type severity weight and a composite risk score is computed across all spans. The verdict logic compares both scores against the active thresholds: maliciousness or PII risk above the block threshold returns BLOCK, PII risk above the lower sanitize threshold returns SANITIZE with spans replaced by typed tokens like `[REDACTED_EMAIL]`, and everything else returns ALLOW. Results are cached in Redis keyed by SHA-256 of the prompt concatenated with the active threshold configuration (5 minute TTL), so a per-tenant override never returns a verdict cached under a different sensitivity. Every decision is written to a PostgreSQL audit log that stores the prompt hash, verdict, scores, and PII count, never the raw text. Blocked prompts also push a hash and reason to a Redis list capped at 1000 entries as a quarantine queue. API key auth and a sliding window rate limiter are enforced per identity. A `GuardrailRunnable` wraps any LangChain `RunnableSerializable` so the proxy can drop into an existing LangChain pipeline without changes to the chain definition.

The frontend is a dark UI built around the check endpoint. The nav bar shows live p95 latency, block rate, and a Redis health indicator. The prompt card has four example queries you can click (injection attack, heavy PII, light PII, clean prompt) and a character counter. Results render below the input: a colored verdict badge, a cached tag when the result was served from Redis, the classifier reason string, two score bars for maliciousness and PII risk, a PII findings list showing the entity type and replacement token for each span, and the sanitized output with redacted tokens highlighted. A history list below the result card lets you click any past prompt to replay it. A footer bar shows rolling session metrics: total requests, block rate, allowed and sanitized counts, p95 and p99 latency.

```bash
git clone https://github.com/haru-g-cmd/guardrail-proxy-llm.git
cd guardrail-proxy-llm
```

Copy the env template:

```bash
cp .env.example .env
```

Inside `.env` you can set:

- `OPENAI_API_KEY` -- needed only if you want `/v1/guardrail/proxy` to forward to a downstream model, not required for screening
- `MALICIOUSNESS_MODEL_PATH` -- path to a trained DistilBERT artifact, defaults to heuristic fallback if not set

Run setup and start services:

```
py -3.12 run.py setup
py -3.12 run.py start
```

Opens on `http://127.0.0.1:8010`. Run the test suite (unit tests pass without Docker):

```
py -3.12 run.py test
```

62 tests, all pass without API keys or Docker.
