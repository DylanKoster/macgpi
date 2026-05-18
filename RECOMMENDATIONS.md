# MACGPi — Code Review Recommendations

This document lists concrete recommendations for improving the MACGPi repository, organised by category.

---

## 1. Bugs

### 1.8 Nested f-string syntax requires Python 3.12+
**File:** `macgpi/engine/main.py`

```python
f"{" and a schema.json file" if schema_required else ''}"
```

Nested quotes inside f-strings are only legal from Python 3.12. The README recommends Python 3.10+, which means this line is a syntax error on 3.10 and 3.11. Replace with a pre-computed variable or use different quote styles.

### 1.9 Project description directory only created conditionally
**File:** `macgpi/engine/main.py`

```python
if not os.path.exists(project_description_path):
    os.makedirs(os.path.dirname(project_description_path), exist_ok=True)
```

`os.makedirs` is only called when the file does not yet exist. On a fresh run the file doesn't exist, so the directory is created—this works. But on a re-run the file *does* exist so the directory creation is skipped; the subsequent `open(..., "w")` still succeeds because the directory is already there. However, the intent is unclear and the logic is easy to break. Replace with an unconditional `os.makedirs(..., exist_ok=True)` before the `open` call.

---

## 2. Security

### 2.1 No validation of `model_host` / `model_port`
**File:** `macgpi/engine/vllm.py`, `macgpi/engine/agent_manager.py`

Both the health-check URL and the `api_base` URL are constructed by string-interpolating user-supplied `host` and `port` values with no sanitisation. A malicious or misconfigured host string (e.g. `localhost/internal-service#`) could redirect requests to unintended endpoints (SSRF). Validate that `host` is a plain hostname or IP and that `port` is in the 1–65535 range before constructing URLs.

### 2.2 `api_key` hardcoded with no override
**File:** `macgpi/engine/agent_manager.py`

```python
"api_key": "EMPTY",
```

Hardcoding a sentinel API key is fine for a local vLLM server, but there is no path to supply a real key (e.g. for a remote OpenAI-compatible endpoint). Prefer reading from an environment variable (e.g. `MACGPI_API_KEY`) with `"EMPTY"` as the fallback.

---

## 3. Architecture & Design

### 3.1 Agent state bleeds between phases
**File:** `macgpi/engine/agent_manager.py`

`AgentManager` creates a **single** `DefaultAgent` instance and reuses it for every prompt across all phases. If `DefaultAgent` maintains an internal conversation history (which is typical for SWE-agent frameworks), earlier phases will pollute the context of later ones. Create a fresh agent instance for each prompt, or explicitly reset the agent state between calls.

### 3.2 JSON schema outputs are not validated programmatically
**Files:** `macgpi/engine/main.py`, prompts

The JSON schemas in `01_plan/schema.json` and `03_evaluate/schema.json` are injected into prompts as formatting guidance for the LLM, but the actual output files written by the agent are never validated against these schemas. If the LLM produces malformed JSON or violates the schema, the error only surfaces later (or never). Add a post-phase validation step using `jsonschema` to assert the output is schema-compliant before advancing to the next phase.

### 3.3 No pipeline resumability
**File:** `macgpi/engine/main.py`

If the pipeline fails mid-run (e.g. network error, LLM timeout), there is no checkpoint mechanism. The entire run must restart from scratch. Consider persisting `phase_visits` and the current phase to a state file in `output_dir`, and resuming from the last successfully completed phase on re-invocation.

### 3.4 Hardcoded PRD path inside the engine
**File:** `macgpi/engine/main.py`

The path `docs/project_description.md` is hardcoded in the engine and also appears in the default phases config. If a user customises the phases config to use a different input path, the engine will still write the PRD to the old location. Pass the PRD path through configuration or derive it consistently from the phases config.

### 3.5 No timeout on agent execution
**File:** `macgpi/engine/agent_manager.py`

`agentManager.run(template_output)` has no timeout. An agent that hangs (waiting for LLM, stuck in a loop, etc.) will block the pipeline indefinitely. Add a configurable timeout and surface a clear error if it is exceeded.

### 3.6 No way to install the package with `pip install .`
**Root directory**

The project has no `pyproject.toml` or `setup.py`. It can only be run by placing the source directory on `PYTHONPATH`. Add a `pyproject.toml` with a `[project]` table and `[project.scripts]` entry so the package is properly installable and the `macgpi` command is registered via an entry point.

### 3.7 Production and development dependencies mixed in one file
**File:** `requirements.txt`

`pre-commit`, `flake8`, `pytest`, `pytest-cov`, and `pytest-mock` are development tools, not runtime dependencies. Split into `requirements.txt` (production) and `requirements-dev.txt` (development) to keep deployments lean and to follow convention.

---

## 4. Code Style & Maintainability

### 4.1 `None` comparisons use `==` implicitly
**Multiple files**

Several places compare against `None` using truthiness (`if model_config_file == None`) instead of the idiomatic `is None`. Use `is None` / `is not None` consistently for singleton comparisons.

### 4.2 Prefer `pathlib.Path` over `os.path` string manipulation
**Multiple files**

Path construction throughout the codebase uses `os.path.join`, `os.path.dirname`, `os.path.exists`, etc. `pathlib.Path` provides a cleaner, object-oriented API that reduces concatenation errors and is the modern Python standard. Migrating would also make the hardcoded path issues more visible.

### 4.3 Empty `__init__.py`
**File:** `macgpi/__init__.py`

The package-level `__init__.py` is completely empty. At minimum it should expose the package version (`__version__`) and the public entry point (`macgpi` function) so users can do `from macgpi import macgpi`.

### 4.4 Bare `except Exception` swallows all errors
**File:** `macgpi/engine/main.py`

The top-level try/except in `macgpi()` catches every exception and returns `False` with a traceback string in the log. While this prevents crashes, it also hides programming errors (e.g. `AttributeError`, `TypeError`) that should propagate. Catch only the expected exception types (e.g. `FileNotFoundError`, `json.JSONDecodeError`) explicitly, and let unexpected ones bubble up.

---

## 5. Testing

### 5.1 Pytest markers defined but never used
**Files:** `pytest.ini`, all test files

`pytest.ini` declares `slow`, `integration`, and `unit` markers, but no test is annotated with any of them. Tag tests with `@pytest.mark.unit` and `@pytest.mark.integration` so they can be filtered (e.g. `pytest -m "not slow"` in CI).

### 5.2 `test_vllm.py` tests `ConnectionError`, not `requests.ConnectionError`
**File:** `tests/test_vllm.py`

The test mocks `side_effect = ConnectionError()`, which raises the Python built-in `ConnectionError`. The `vllm_health` function catches `Exception` (all exceptions) so the test passes, but in reality `requests` raises `requests.exceptions.ConnectionError`, which is a subclass. The test does not verify the exact exception path. Use `requests.exceptions.ConnectionError` in the test to be precise.

### 5.3 No end-to-end / integration tests
All tests mock out `AgentManager`, `TemplateManager`, or `vllm_health`. There are no integration tests that exercise actual template rendering combined with the phase state machine against real (or recorded) LLM responses. Even a single integration test that runs the full pipeline against a stub HTTP server would catch regressions in the phase-wiring logic.

### 5.4 No coverage threshold enforced
**File:** `pytest.ini`

`pytest-cov` is listed as a dependency but no `--cov` or `--cov-fail-under` option is set in `pytest.ini`. Add a minimum coverage threshold (e.g. `--cov=macgpi --cov-fail-under=80`) to prevent regressions.

---

## 6. Documentation

### 6.1 Duplicate step 3 in Getting Started
**File:** `README.md`

The "Quick local steps" section contains two steps numbered `3`:
- `3. Ensure a vLLM server is running…`
- `3. Run MACGPi`

Renumber to 1–4.

### 6.2 `--phases-config` missing from CLI reference table
**File:** `README.md`

The `--phases-config` argument is documented in the source code and accepted by the CLI, but it is absent from the "Optional parameters" section of the README. Add an entry consistent with the other parameters.

### 6.3 Typos in README
**File:** `README.md`

- "the schema is always taken from the phase directory, adn the output path…" → **and**
- "path to the produces code" → **produced**

### 6.4 Repository layout in README is inconsistent with actual structure
**File:** `README.md`

The layout section says:
```
macgpi/
  prompts/   — phase templates and JSON schemas
  configs/   — configuration files
```
But these directories live under `macgpi/macgpi/` (the package), not at the repo root. Update the layout description to reflect the actual directory structure.

### 6.5 No CONTRIBUTING or development setup guide
There is no guide explaining how to set up a development environment, run the test suite, use pre-commit hooks, or submit changes. A `CONTRIBUTING.md` (or a "Development" section in the README) covering these steps would lower the barrier for new contributors.

---

## 7. CI / Tooling

### 7.1 No CI pipeline
There is no GitHub Actions (or equivalent) configuration. At minimum, a CI workflow should install dependencies, run `flake8`, and run `pytest` on every push and pull request to catch regressions automatically.

### 7.2 Pre-commit is listed as a dependency but not configured
**File:** `requirements.txt`

`pre-commit >= 3.0.0` appears in the dependencies, but there is no `.pre-commit-config.yaml` file. Either add a config that runs `flake8` (and optionally `black`/`isort`) or remove `pre-commit` from the dependencies.

### 7.3 No `.gitignore`
The repository does not include a `.gitignore`. At minimum it should ignore `__pycache__/`, `.venv/`, `*.pyc`, `.pytest_cache/`, and any generated output directories to prevent accidental commits of artefacts.
