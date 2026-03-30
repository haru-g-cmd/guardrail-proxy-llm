# Root Cause Analysis - Guardrail Proxy

**Project:** Guardrail Proxy (PR5)  
**Prepared:** 2026-03-28  
**Scope:** All defects and near-misses identified across Sprints 1–3  

---

## Summary Table

| ID | Title | Stage | Severity | Sprint |
|----|-------|-------|----------|--------|
| RCA-001 | Python 3.13 / torch wheel incompatibility | Design | Critical | Pre-S1 |
| RCA-002 | Heuristic marker missed article variant | Coding | Medium | Sprint 1 |
| RCA-003 | FastAPI `dependency_overrides` keyed by string instead of function object | Testing | High | Sprint 3 |

---

## RCA-001 - Python 3.13 / torch wheel incompatibility

### Description

The initial environment setup selected Python 3.13 (the system default on macOS at the time) as the interpreter for the virtual environment. PyTorch 2.2.x does not publish wheels for Python 3.13 on macOS x86_64; `pip install torch` silently fell back to a source build and then failed, leaving the environment in a broken state. The session had to be completely discarded.

### Root Cause Stage

**Design** - The runtime version constraint was not established before the dependency graph was resolved.

### Severity

**Critical.** The entire prior session was lost. No functional code survived; the workspace had to be wiped and rebuilt from scratch.

### How It Could Have Been Avoided

1. Declare `requires-python = ">=3.11,<3.13"` in `pyproject.toml` before running any `pip install`.  
2. Encode the Python version requirement in `run.py`'s `_find_bootstrap_python()` helper so setup fails fast with a clear error if the wrong interpreter is invoked.  
3. Pin `torch>=2.2.0,<2.3.0` as the first resolved package, confirming wheel availability for the target interpreter before proceeding.

### Fix Applied

- `pyproject.toml`: `requires-python = ">=3.11,<3.13"`  
- `run.py` `_find_bootstrap_python()`: iterates `python3.11`, `python3.12`, rejects 3.13+.  
- `torch` pinned to `>=2.2.0,<2.3.0` in `[project.dependencies]`.

### Impact on Overall Requirements

**None after fix.** All functional requirements (injection detection, PII screening, LangChain middleware) are unaffected by the runtime version choice within the `[3.11, 3.13)` range. The training artifact (`artifacts/distilbert_guardrail/`) is version-agnostic.

---

## RCA-002 - Heuristic marker missed article variant

### Description

The heuristic fallback classifier in `core/classifier.py` maintained a list of injection phrases used when the DistilBERT model artifact was unavailable. The phrase `"reveal system prompt"` was present but not `"reveal the system prompt"` (with article). A test case sending the latter was incorrectly scored below the `BLOCK` threshold and returned `allow`.

### Root Cause Stage

**Coding** - The phrase list was constructed by manual enumeration rather than by covering the normalized-token space. Adding or omitting a function word ("the") changed whether the substring check fired.

### Severity

**Medium.** The error surfaced only in the heuristic fallback path, which is the degraded-mode path used when no trained model is available. The primary code path (DistilBERT classifier, trained in Sprint 2 with 290 examples) handles this class of input correctly and was not affected.

### How It Could Have Been Avoided

1. **Token normalisation first**: strip articles/stopwords before phrase matching.  
2. **Fuzzy / partial matching**: check for any 2-gram from the target phrase rather than requiring the full literal string.  
3. **Exhaustive variant generation** at design time: for each listed phrase, systematically add article-prefixed and article-stripped variants.  
4. **Property-based testing** (Hypothesis): generate paraphrases of each canonical marker and assert they are blocked.

### Fix Applied

Added `"reveal the system prompt"` to the heuristic marker list in `core/classifier.py`. This was a one-line change and unblocked all 28 Sprint 1 tests.

### Impact on Overall Requirements

**None after fix.** The heuristic fallback is a safety net, not the primary classifier. Sprint 2 replaced it with a fine-tuned DistilBERT model (98.3% validation accuracy, F1 0.9831) that handles article variants and paraphrases as a natural consequence of training on 290 diverse examples.

---

## RCA-003 - FastAPI `dependency_overrides` keyed by string instead of function object

### Description

The Sprint 3 test helper `_override(**extra)` accepted keyword arguments to inject custom dependency implementations. Python keyword arguments become `str`-keyed dict entries (e.g., `{"get_rate_limiter": fn}`). FastAPI's `app.dependency_overrides` dict is keyed by the **callable object itself** - a reference, not a name. The string-keyed entries were silently accepted by the dict but never matched any dependency, so all calls fell through to the real module-level singletons (`_limiter`, `_metrics`).

Two tests failed as a result:

| Test | Observed | Expected |
|------|---------|----------|
| `test_burst_beyond_limit_returns_429` | Request 5 → 429 | Requests 1–5 → 200, request 6 → 429 |
| `test_metricsz_initial_state_is_zero` | `total_requests == 37` | `total_requests == 0` |

The global `_limiter` singleton had accumulated ~5 requests from `TestAuthentication` (which ran first and did not override the limiter), so the rate-limit window was already almost full when the rate-limit test fixture started. The global `_metrics` singleton had accumulated 37 requests from all preceding tests.

### Root Cause Stage

**Testing** - A type mismatch in the test helper (`dict[str, Callable]` vs. `dict[Callable, Callable]`) was not caught by the type checker because the function was annotated `**extra` without a typed signature.

### Severity

**High.** Two tests returned incorrect results. More critically, the pattern was latent: if other fixtures had accidentally passed the wrong key type, the oversight could have masked broken production dependencies silently - the test would pass but would be testing the singleton, not the injected instance.

### How It Could Have Been Avoided

1. **Type annotation**: `extra_overrides: dict[Callable[..., Any], Callable[..., Any]] | None` would have caused mypy to flag a `str`-typed call site immediately.  
2. **Assertion in helper**: `assert all(callable(k) for k in extra_overrides)` as a guard at the start of `_override()`.  
3. **Review FastAPI docs** before writing the first fixture: the `dependency_overrides` contract explicitly requires callable keys.

### Fix Applied

Replaced `_override(settings, **extra)` with `_override(settings, extra_overrides: dict | None = None)`. All call sites updated to pass a dict literal with function-object keys:

```python
_override(settings, extra_overrides={get_rate_limiter: lambda: fresh_limiter})
```

`tenant_client` fixture updated to also inject a fresh `get_rate_limiter` override to prevent cross-fixture contamination through the global limiter.

### Impact on Overall Requirements

**None.** The production code (`SlidingWindowLimiter`, `MetricsStore`, module-level singletons, route wiring) was always correct. The defect existed only in the test harness. All 53 tests pass after the fix with strictly hermetic isolation.

---

## Lessons Learned

| # | Lesson | Category |
|---|--------|----------|
| 1 | Pin the Python interpreter range *before* pip-resolving any package with compiled wheels. | Dependency Management |
| 2 | Heuristic string matchers must normalise input (strip articles, lowercase, tokenise) before matching - never rely on a literal phrase list alone. | ML Safety |
| 3 | FastAPI `dependency_overrides` keys are function objects, not names. Always use typed helper signatures and assert `callable(k)` on every override key in tests. | Test Design |
| 4 | Process-global singletons are correct for production but unsafe in test suites unless every fixture injects a fresh instance. Use `autouse` fixtures or typed override helpers to enforce this invariant. | Test Isolation |
